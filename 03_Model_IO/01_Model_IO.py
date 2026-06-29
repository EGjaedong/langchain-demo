from langchain_core.prompts import PromptTemplate

from langchain_learn.llm_client import create_deepseek_chat

template = """您是一位专业的鲜花店文案撰写员。
对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？
"""

prompt = PromptTemplate.from_template(template)
print(prompt)

model = create_deepseek_chat()
input = prompt.format(price=50, flower_name=["玫瑰"])
output = model.invoke(input)
print(output.content)
