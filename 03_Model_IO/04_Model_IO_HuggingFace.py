import os
from llm_client import get_deepseek_api_key, get_huggingface_api_token

os.environ["OPENAI_API_KEY"] = get_deepseek_api_key()

from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

template = """您是一位专业的鲜花店文案撰写员。
对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？
"""

prompt = PromptTemplate.from_template(template)
print(prompt)

# 设置 HuggingFace API Token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = get_huggingface_api_token()

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# HF Inference API 上可用的开源模型（中文表现好）
# 备选：meta-llama/Llama-3.2-1B-Instruct（英文更强，中文一般）
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=get_huggingface_api_token(),
    max_new_tokens=128,
    temperature=0.7,
)
model = ChatHuggingFace(llm=llm)

input_text = prompt.format(flower_name="玫瑰", price=50)
output = model.invoke([HumanMessage(content=input_text)])
print(output.content)
