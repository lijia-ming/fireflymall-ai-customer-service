"""
AI 聊天路由：用户对话 + 人工客服协作（SSE 流式）

- POST /ai/chat            用户发送消息，SSE 流式返回（打字机 + 工具提示 + 完成事件）
- POST /ai/human/end       结束人工客服服务（流式返回 AI 转达的结语）
- GET  /ai/history/{thread_id}  查看 AI 阶段会话历史（供人工客服接手前了解上下文）

SSE 事件格式（data 均为 JSON）：
- token:    主Agent 模型流式 token，逐 chunk 即时发出，前端负责渲染（打字机效果在前端做）
- rollback: 上一段流式文本是内部思考（路由/工具调用的前导话术），前端应删除
            自"当前段起点"以来渲染的内容（段起点 = 上一个 turn/rollback 之后）
- turn:     上一段流式文本是最终回答，本段结束，前端可固定该段并更新段起点
- tool:     工具调用提示（AI客服调用了工具X，用户可见）
- done:     本次会话结束，data 含 out_msg / manual_intervention / thread_id
            （manual_intervention=True 时 out_msg 是固定话术而非 AI 回复，
              调用方应据此将用户转出到人工客服自己的渠道）
- error:    出错，data.msg 为错误描述

作为段边界信号：False → 发 rollback 让前端清除该段；True → 发 turn 固定该段。
"""
import json
import os
import signal
from typing import Any, AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk
from langgraph.errors import GraphDrained
from langgraph.runtime import RunControl
from pydantic import BaseModel, Field

from SPO.route_results import RouteResponse
from SPO.state import UserContext
from Tools.log_settings import LogSetting
from Tools.stream_route_text import RouteJsonStreamFilter
from main import graph
from model.injection_detector import detector
from routes.file import UPLOAD_DIR

logger = LogSetting.create(__name__)

# 注入防线拒绝提示（图外第一道）
_INJECTION_BLOCK_MSG = '检测到您的消息包含恶意指令，已拦截。如有正常咨询需求，请重新表述。'
_MUTE_MSG = '您的账号因多次恶意操作已被临时限制，请稍后再试，或联系人工客服。'

route = APIRouter()
control = RunControl()
signal.signal(signal.SIGTERM, lambda *_: control.request_drain("sigterm"))  # type: ignore[no-any]


class FileRef(BaseModel):
    """上传文件引用（upload 接口返回后由前端回传）"""
    filename: str = Field(..., description='upload 接口返回的 filename')
    original_name: str = Field(default='', description='upload 接口返回的原始文件名')


class ChatRequest(BaseModel):
    """用户聊天请求"""
    msg: str = Field(..., description='用户消息内容', min_length=1)
    files: list[FileRef] = Field(default_factory=list,
                                 description='本轮随消息携带的上传文件（upload 接口返回的引用列表）')


def _resolve_files(files: list[FileRef]) -> list[dict]:
    """把前端回传的 filename 解析为 upload_dir 内的绝对路径。

    安全校验：拒绝路径穿越（abspath 后必须仍在 upload_dir 内）与不存在的文件，
    防止客户端传任意路径让专家工具读取服务器文件。
    """
    abs_upload = os.path.abspath(UPLOAD_DIR)
    resolved = []
    for ref in files:
        candidate = os.path.abspath(os.path.join(abs_upload, ref.filename))
        if not candidate.startswith(abs_upload + os.sep):
            raise ValueError(f"非法文件路径: {ref.filename}")
        if not os.path.isfile(candidate):
            raise ValueError(f"文件不存在: {ref.filename}")
        resolved.append({'name': ref.original_name or ref.filename, 'path': candidate})
    return resolved


class HumanEndRequest(BaseModel):
    """结束人工客服服务请求"""
    msg: str = Field(..., description='人工客服结束服务时的结语，将由 AI 转达给用户', min_length=1)
    thread_id: str | None = Field(default=None, description='会话线程ID，缺省时使用用户ID')


def _sse(event: str, data: dict) -> str:
    """序列化一条 SSE 事件"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_graph(payload: dict, thread_id: str) -> AsyncIterator[str]:
    """
    SSE 事件生成器：消费图流并转发为前端可见事件。

    基于 graph.astream（默认 v1 协议）：
    - stream_mode='messages'：主Agent 模型流式 token（AIMessageChunk），按 langgraph_node 过滤
    - stream_mode='custom'：工具调用提示（AI客服调用了工具X）
    - stream_mode='values'：仅用于取最终态的 out_msg / manual_intervention，不转发给前端
    """
    last_state: dict[str, Any] = {}
    stream_filter = RouteJsonStreamFilter()
    # ---- 注入检测（第一道防线）----
    check = await detector.check(thread_id, payload.get('msg', ''))
    if check['action'] in ('hard', 'banned'):
        yield _sse('error', {'msg': _MUTE_MSG if check['action'] == 'banned' else _INJECTION_BLOCK_MSG})
        return
    injection_ctx = None
    if check['action'] in ('suspect', 'warn'):
        injection_ctx = {'level': check['action'], 'score': round(check['score'], 3)}
    try:
        async for mode, data in graph.astream(
            payload,
            config={'configurable': {'thread_id': thread_id}},
            context=UserContext(user_id=thread_id, injection=injection_ctx),
            stream_mode=['messages', 'custom', 'values'],
            control=control,
        ):
            if mode == 'messages':
                chunk, metadata = data
                if metadata.get('langgraph_node') != 'chat_node':
                    continue
                if isinstance(chunk, AIMessageChunk):
                    # 模型流式增量：经过滤器转发（打字机保留，逐 chunk 发出）
                    if isinstance(chunk.content, str) and chunk.content:
                        visible_text = stream_filter.push(chunk.content)
                        if visible_text:
                            yield _sse('token', {'text': visible_text})
                else:
                    # 模型调用结束的整条重放消息（chat_node 包装消息）：作为段边界信号。
                    # stream_visible=False → 本段是思考文本，前端回滚清除；
                    # True → 本段是最终回答，前端固定并更新段起点。
                    visible = chunk.additional_kwargs.get('stream_visible')
                    if visible is not None:
                        # 段结算：补发过滤器兜底内容（畸形 JSON 场景），再发段边界
                        leftover = stream_filter.flush()
                        if leftover:
                            yield _sse('token', {'text': leftover})
                        if visible:
                            yield _sse('turn', {})
                        else:
                            yield _sse('rollback', {})
            elif mode == 'custom':
                yield _sse('tool', {'text': str(data)})
            elif mode == 'values':
                last_state = data
        yield _sse('done', {
            'out_msg': last_state.get('out_msg'),
            'manual_intervention': last_state.get('manual_intervention', False),
            'thread_id': thread_id,
        })
    except GraphDrained as e:
        logger.error(f"图执行被终止（thread_id={thread_id}）：{e}")
        yield _sse('error', {'msg': '服务暂时不可用，请稍后再试'})
    except Exception as e:
        logger.error(f"用户聊天失败（thread_id={thread_id}）：{e}")
        yield _sse('error', {'msg': '服务暂时不可用，请稍后再试'})


def _streaming_response(thread_id: str, payload: dict) -> StreamingResponse:
    return StreamingResponse(
        _stream_graph(payload, thread_id),
        media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@route.get('/health')
async def health_check():
    """
    健康检查端点（供 Docker/K8s 探针使用）
    """
    from main import graph
    return {
        'status': 'ok',
        'graph_ready': graph is not None,
        'service': 'intelligent-customer-service',
    }


@route.post('/chat')
async def chat(body: ChatRequest, request: Request):
    """
    用户聊天入口（SSE 流式）。

    manual_intervention=True（done 事件）时：out_msg 是固定话术而非 AI 回复，
    调用方应将用户引导至人工客服自己的渠道，不要展示给用户当作 AI 回复。
    """
    user_id = str(request.state.login_info.get('id'))
    if not user_id:
        return RouteResponse.error(msg="不允许匿名用户聊天")
    try:
        file_urls = _resolve_files(body.files) if body.files else None
    except ValueError as e:
        logger.warning(f"文件引用校验失败（thread_id={user_id}）：{e}")
        return RouteResponse.error(msg=str(e))
    payload: dict = {'msg': body.msg}
    if file_urls:
        payload['file_urls'] = file_urls  # type: ignore
    return _streaming_response(user_id, payload)


@route.get('/retry')
async def retry_graph(request: Request):
    user_id = str(request.state.login_info.get('id'))
    if not user_id:
        return RouteResponse.error(msg="不允许匿名用户聊天")
    try:
        graph.invoke(None, {'configurable': {'thread_id': user_id}})
        return RouteResponse.ok()
    except Exception as e:
        return RouteResponse.error(msg=f"重试失败（thread_id={user_id}）：{e}")


@route.post('/human/end')
async def human_end(body: HumanEndRequest, request: Request):
    """
    结束人工客服服务（SSE 流式）：注入 human_reply=True 走图，解除人工接管状态，
    AI 将结语整理转达给用户并恢复正常服务。
    调用方（人工客服方）应确认该线程当前处于接管状态（manual_intervention=True）。
    """
    user_id = str(request.state.login_info.get('id'))
    if not user_id:
        return RouteResponse.error(msg="不允许匿名用户聊天")
    return _streaming_response(user_id, {'msg': body.msg, 'human_reply': True})


@route.get('/history')
async def history(request: Request):
    """获取 AI 阶段会话历史（人工客服接手前查看上下文）"""
    thread_id = str(request.state.login_info.get('id'))
    if not thread_id:
        return RouteResponse.error(msg="不允许匿名用户聊天")
    try:
        snapshot = await graph.aget_state({'configurable': {'thread_id': thread_id}})
        messages = [
            # files 取自消息 metadata（系统写入通道），前端据此渲染文件卡片，无需解析文本
            {'type': type(m).__name__, 'content': str(m.content),
             'files': m.additional_kwargs.get('files') or []}
            for m in snapshot.values.get('messages', [])
        ]
        return RouteResponse.ok(data={
            'thread_id': thread_id,
            'manual_intervention': snapshot.values.get('manual_intervention', False),
            'messages': messages,
        })
    except Exception as e:
        logger.error(f"获取会话历史失败（thread_id={thread_id}）：{e}")
        return RouteResponse.error(msg="获取会话历史失败")
