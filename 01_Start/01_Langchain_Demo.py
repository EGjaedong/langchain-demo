import os

from pydantic import SecretStr

from langchain_learn.llm_client import get_deepseek_chat_model, get_deepseek_api_key, get_deepseek_api_host
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=get_deepseek_chat_model(),
    api_key=SecretStr(get_deepseek_api_key()),
    base_url=get_deepseek_api_host(),
    max_completion_tokens=200,
)

response = llm.invoke("请给我写一句情人节红玫瑰的中文宣传语")
print(response.content)