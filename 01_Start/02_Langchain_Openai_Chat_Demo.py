import os

from langchain_learn.llm_client import get_deepseek_api_key, get_deepseek_chat_model, get_deepseek_api_host
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

os.environ["OPENAI_API_KEY"] = get_deepseek_api_key()
chat = ChatOpenAI(
    model = get_deepseek_chat_model(),
    base_url = get_deepseek_api_host(),
    max_completion_tokens = 60,
    temperature = 0.8,
)

messages = [
    SystemMessage(content="你是一个很棒的智能助手"),
    HumanMessage(content="请给我的花店起个名"),
]
response = chat.invoke(messages)
print(response.content)
