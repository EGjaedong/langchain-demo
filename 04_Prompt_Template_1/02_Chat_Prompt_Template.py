from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# 模版定义
template = "你是一位专业顾问，负责为专注于{product}的公司起名。"
system_message_promtp = SystemMessagePromptTemplate.from_template(template)
human_template = "公司主打产品是{product_detail}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
prompt_template = ChatPromptTemplate.from_messages([system_message_promtp, human_message_prompt])

prompt = prompt_template.format_prompt(product="鲜花装饰", product_detail="创新的鲜花设计。").to_messages()

import os
from llm_client import get_deepseek_api_key, get_deepseek_api_host, get_deepseek_chat_model

os.environ["OPENAI_API_KEY"] = get_deepseek_api_key()

from langchain_openai.chat_models import ChatOpenAI
chat = ChatOpenAI(
    model=get_deepseek_chat_model(),
    base_url=get_deepseek_api_host(),
    max_completion_tokens = 200,
    temperature=0.5
)
result = chat.invoke(prompt)
print(result)