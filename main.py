import os

from pydantic import SecretStr
from llm_client import get_deepseek_chat_model
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=get_deepseek_chat_model(),
    api_key=SecretStr(os.environ["DEEPSEEK_API_KEY"]),
    base_url=os.environ["DEEPSEEK_API_HOST"],
    max_completion_tokens=200,
)

response = llm.invoke("请给我写一句情人节红玫瑰的中文宣传语")
print(response.content)