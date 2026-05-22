from langchain_core.prompts.prompt import PromptTemplate # String Prompt Template
from langchain_core.prompts import FewShotPromptTemplate # Few-shot Prompt Template, 少样本提示模版，通过示例来教模型如何回答
from langchain_core.prompts import ChatPromptTemplate # Chat Prompt Template, 聊天提示模版，用于聊天模型
from langchain_core.prompts import (
    ChatMessagePromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
