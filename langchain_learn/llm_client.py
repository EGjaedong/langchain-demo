"""从仓库根目录加载 .env，并提供 OpenAI 兼容客户端与模型名。"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


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
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BETA_API_HOST"),
        )
    else:
        return OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_API_HOST"),
        )


def get_deepseek_chat_model(default: str = "deepseek-v4-flash") -> str:
    return os.getenv("DEEPSEEK_MODEL") or default


def get_deepseek_api_key() -> str:
    return os.getenv("DEEPSEEK_API_KEY") or ""


def get_deepseek_api_host() -> str:
    return os.getenv("DEEPSEEK_API_HOST") or ""

def get_huggingface_api_token() -> str:
    return os.getenv("HF_TOKEN") or ""

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