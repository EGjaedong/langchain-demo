from langchain_core.prompts import PromptTemplate

from langchain_learn.llm_client import create_deepseek_chat

template = """您是一位专业的鲜花店文案撰写员。
对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？
"""

prompt = PromptTemplate.from_template(template)
model = create_deepseek_chat()

flowers = ["玫瑰", "康乃馨", "百合", "郁金香", "向日葵"]
prices = [50, 30, 20, 10, 15]

for flower, price in zip(flowers, prices):
    input = prompt.format(price=price, flower_name=flower)
    output = model.invoke(input)
    print(output.content)
