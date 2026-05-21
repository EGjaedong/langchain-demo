import os
from llm_client import get_deepseek_api_key, get_deepseek_chat_model, get_deepseek_api_host

os.environ["OPENAI_API_KEY"] = get_deepseek_api_key()

from langchain_core.prompts import PromptTemplate

template = """您是一位专业的鲜花店文案撰写员。
对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？
"""

prompt = PromptTemplate.from_template(template)

from langchain_openai import ChatOpenAI
model = ChatOpenAI(model=get_deepseek_chat_model(), base_url = get_deepseek_api_host())

flowers = ["玫瑰", "康乃馨", "百合", "郁金香", "向日葵"]
prices = [50, 30, 20, 10, 15]

for flower, price in zip(flowers, prices):
    input = prompt.format(price=price, flower_name=flower)
    output = model.invoke(input)
    print(output.content)

