"""
项目主文件
"""
from contextlib import asynccontextmanager
from typing import Literal, Optional

import redis
from anthropic import BaseModel
from fastapi import FastAPI, Request
import jwt
from langgraph.cache.sqlite import SqliteCache
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import AsyncPostgresStore
from langgraph.types import Checkpointer
from pydantic import Field
from starlette.middleware.cors import CORSMiddleware
from starlette.datastructures import State
from starlette.responses import JSONResponse

from SPO.route_results import RouteResponse, ResultCode
from Tools.jwt_key_manage import KeyManage
from Tools.log_settings import LogSetting
from agent.main_agent import builder
from load_config.config import ROOT_BASE_DIR_PATH, config
from routes.file import route as file_route
from routes.ai_chat import route as ai_chat_route
from routes.manager import route as manager_route

logger = LogSetting.create(__name__)

# 探针端点豁免鉴权：Docker HEALTHCHECK 与 K8s liveness/readiness/startup 探针不携带 token
AUTH_EXEMPT_PATHS = frozenset({'/ai/health'})

# 必须走 config 而非硬编码 localhost：容器内 Redis 是独立服务（compose 服务名 / K8s Service 名）
_redis_conf = config.get('redis')
redis_pool = redis.ConnectionPool(
    host=_redis_conf.get('host', 'localhost'),
    port=_redis_conf.get('port', 6379),
    db=_redis_conf.get('db', 0),
    password=_redis_conf.get('password'),
    max_connections=_redis_conf.get('max_connections', 20),
    decode_responses=True
)
redis_con = redis.Redis(connection_pool=redis_pool)

_postgres_conf = config.get('databases').get('postgres')
_postgres_url = f"postgresql://{_postgres_conf.get('user')}:{_postgres_conf.get('password')}@{_postgres_conf.get('host')}:{_postgres_conf.get('port')}/{_postgres_conf.get('db')}?sslmode=disable"

postgres_check = AsyncPostgresSaver.from_conn_string(_postgres_url)
postgres_store = AsyncPostgresStore.from_conn_string(_postgres_url)
graph = None


class ParsedTokenData(BaseModel):
    type: Literal['user', 'admin', 'unverified'] = Field(default='unverified', description="状态类型")
    id: str = Field(default='', description="用户/管理员ID")
    is_old: bool = Field(default=False, description="是否为旧密钥")
    permission: Optional[int] = Field(default=None, description="管理员权限")


def _normalize_id(raw_id) -> str:
    """token 里的 id 可能是数字或字符串，统一转成字符串（ParsedTokenData.id 是 str 类型，
    pydantic v2 不会把 int 隐式转 str，数字 id 会抛 ValidationError 导致 401）"""
    return '' if raw_id is None else str(raw_id)


def verify_token(token: str) -> ParsedTokenData | bool:
    """
    验证token是否有效
    :param token: token字符串
    :return: 是否有效
    """
    if not token.startswith("Bearer "):
        return False
    token = token[7:]
    if not token:
        return False
    old_key = redis_con.hget('jwt_key', 'old')
    new_key = redis_con.hget('jwt_key', 'new')
    # 捕获 Exception：PyJWT 的 ExpiredSignatureError 不是 DecodeError 的子类，
    # 只捕获 DecodeError 会让过期 token 抛未捕获异常导致 500；key 为 None 时还会抛 TypeError
    try:
        payload = jwt.decode(jwt=token, key=new_key, algorithms=['HS256'])
        return ParsedTokenData(
            type=payload.get('type', 'unverified'),
            id=_normalize_id(payload.get('id')),
            is_old=False,
            permission=payload.get('permission', None)
        )
    except Exception:
        try:
            payload = jwt.decode(jwt=token, key=old_key, algorithms=['HS256'])
            return ParsedTokenData(
                type=payload.get('type', 'unverified'),
                id=_normalize_id(payload.get('id')),
                is_old=True,
                permission=payload.get('permission', None)
            )
        except Exception:
            return False


async def create_graph(
        graph_build: StateGraph,
        store: Optional[BaseStore] = None,
        check: Optional[Checkpointer] = None
):
    """
    当 store 与 check 均为None时，默认使用InMemoryStore和InMemorySaver
    """
    global graph
    if (not store or isinstance(store, InMemoryStore)) and (not check or isinstance(check, InMemorySaver)):
        graph = graph_build.compile(
            checkpointer=InMemorySaver(),
            store=InMemoryStore(),
            cache=SqliteCache(path=ROOT_BASE_DIR_PATH / 'cache/agent_cache.db')
        )
    async with store, check:
        if hasattr(check, 'setup'):
            await check.setup()
        graph = graph_build.compile(
            checkpointer=check,
            store=store,
            cache=SqliteCache(path=ROOT_BASE_DIR_PATH / 'cache/agent_cache.db')
        )


@asynccontextmanager
async def lifespan(api: FastAPI):
    global graph
    await create_graph(builder, postgres_store, postgres_check)
    key_manage = KeyManage()
    key_manage.start()
    from Tools.middleware.memory.memory_rag import start_consumers
    # 启动RabbitMQ消费者
    await start_consumers()
    logger.info("=" * 60)
    logger.info("FastAPI 应用已【完全启动成功】！")
    logger.info("主页: http://127.0.0.1:8000")
    logger.info("=" * 60)
    yield
    key_manage.shutdown()
    graph = None
    logger.info("应用已关闭")


def start_app():
    app = FastAPI(lifespan=lifespan)

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        # 健康检查端点直接放行：探针不带 token，若走鉴权会返回 401，
        # 导致容器永远 unhealthy、K8s 探针失败反复重启 Pod
        if request.url.path in AUTH_EXEMPT_PATHS:
            return await call_next(request)
        # 在请求前验证token是否有效
        authorization = request.headers.get("Authorization")
        if not authorization:
            resp = RouteResponse.error(code=ResultCode.UNAUTHORIZED, msg="token不能为空")
            return JSONResponse(
                content=resp.model_dump(),
                status_code=ResultCode.UNAUTHORIZED.value
            )
        res = verify_token(authorization)

        if not res:
            resp = RouteResponse.error(code=ResultCode.UNAUTHORIZED, msg="token无效")
            return JSONResponse(
                content=resp.model_dump(),
                status_code=ResultCode.UNAUTHORIZED.value
            )
        # 转字典
        request.state.login_info = res.model_dump()
        response = await call_next(request)
        response.headers['replace_jwt'] = str(res.is_old)
        return response

    app.add_middleware(
        CORSMiddleware,  # type: ignore
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        # 自定义响应头
        expose_headers=["replace_jwt"],
    )
    app.state = State()
    app.include_router(file_route, prefix='/files', tags=['文件操作'])
    app.include_router(ai_chat_route, prefix='/ai', tags=['AI聊天'])
    app.include_router(manager_route, prefix='/manager', tags=['管理员操作界面'])
    return app
