"""从仓库根目录加载 .env，并提供 OpenAI 兼容客户端与模型名。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import SecretStr


def _load_env() -> Path:
    """从本包所在目录向上查找第一个包含 .env 的目录并加载。"""
    start = Path(__file__).resolve().parent
    for p in [start, *start.parents]:
        env_file = p / ".env"
        if env_file.is_file():
            load_dotenv(env_file)
            return p
    load_dotenv()
    return Path.cwd()


REPO_ROOT = _load_env()


def _get_env(name: str) -> str:
    return os.getenv(name) or ""


def create_chat_openai(
    *,
    api_key: str,
    base_url: str,
    model: str,
    **kwargs: Any,
) -> ChatOpenAI:
    """创建显式传入凭据的 ChatOpenAI，避免依赖 os.environ 副作用。"""
    return ChatOpenAI(
        model=model,
        api_key=SecretStr(api_key),
        base_url=base_url,
        **kwargs,
    )


def create_deepseek_chat(**kwargs: Any) -> ChatOpenAI:
    return create_chat_openai(
        api_key=get_deepseek_api_key(),
        base_url=get_deepseek_api_host(),
        model=get_deepseek_chat_model(),
        **kwargs,
    )


def create_volcano_chat(**kwargs: Any) -> ChatOpenAI:
    return create_chat_openai(
        api_key=get_volcano_api_key(),
        base_url=get_volcano_api_host(),
        model=get_volcano_chat_model(),
        **kwargs,
    )


def configure_huggingface_token() -> str:
    """将 HF token 写入 HuggingFace Hub 所需的环境变量。"""
    token = get_huggingface_api_token()
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
    return token


def get_qwen_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("QIANWEN_API_KEY"),
        base_url=os.getenv("QIANWEN_API_HOST"),
    )


def get_openai_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_HOST"),
    )


def get_openai_chat_model(default: str = "gpt-5-mini") -> str:
    return os.getenv("OPENAI_MODEL") or default


def get_deepseek_client(use_beta: bool = False) -> OpenAI:
    if use_beta:
        return OpenAI(
            api_key=get_deepseek_api_key(),
            base_url=_get_env("DEEPSEEK_BETA_API_HOST"),
        )
    return OpenAI(
        api_key=get_deepseek_api_key(),
        base_url=get_deepseek_api_host(),
    )


def get_deepseek_chat_model(default: str = "deepseek-v4-flash") -> str:
    return os.getenv("DEEPSEEK_MODEL") or default


def get_deepseek_api_key() -> str:
    return _get_env("DEEPSEEK_API_KEY")


def get_deepseek_api_host() -> str:
    return _get_env("DEEPSEEK_API_HOST")


def get_huggingface_api_token() -> str:
    return _get_env("HF_TOKEN")


def get_minimax_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("MINIMAX_API_KEY"),
        base_url=os.getenv("MINIMAX_API_HOST"),
    )


def get_minimax_api_key() -> str:
    return os.getenv("MINIMAX_API_KEY") or ""


def get_minimax_chat_model(default: str = "MiniMax-M2.7") -> str:
    return os.getenv("MINIMAX_MODEL") or default


def get_minimax_api_host() -> str:
    return os.getenv("MINIMAX_API_HOST") or ""


def get_volcano_client() -> OpenAI:
    return OpenAI(
        api_key=get_volcano_api_key(),
        base_url=get_volcano_api_host(),
    )


def get_volcano_api_key() -> str:
    return _get_env("VOLCANO_API_KEY")


def get_volcano_api_host() -> str:
    return _get_env("VOLCANO_API_HOST")


def get_volcano_chat_model(default: str = "deepseek-v4-flash") -> str:
    return os.getenv("VOLCANO_MODEL") or default
