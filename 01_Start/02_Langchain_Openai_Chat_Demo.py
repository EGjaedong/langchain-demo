from langchain_core.messages import HumanMessage, SystemMessage

from langchain_learn.llm_client import create_deepseek_chat

chat = create_deepseek_chat(
    max_completion_tokens=60,
    temperature=0.8,
)

messages = [
    SystemMessage(content="你是一个很棒的智能助手"),
    HumanMessage(content="请给我的花店起个名"),
]
response = chat.invoke(messages)
print(response.content)
