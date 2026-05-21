import os
from llm_client import get_deepseek_api_key, get_deepseek_chat_model, get_deepseek_api_host
os.environ["OPENAI_API_KEY"] = get_deepseek_api_key()

from langchain_core.prompts import PromptTemplate

template = """您是一位专业的鲜花店文案撰写员。
对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？
{format_instructions}"""

prompt = PromptTemplate.from_template(template)
print(prompt)

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

class FlowerCopy(BaseModel):
    description: str = Field(description="鲜花的描述文案")
    reason: str = Field(description="为什么要这样写这个文案")

output_parser = PydanticOutputParser(pydantic_object=FlowerCopy)
format_instructions = output_parser.get_format_instructions()

prompt = PromptTemplate.from_template(template, partial_variables={"format_instructions": format_instructions})

flowers = ["玫瑰", "百合", "康乃馨"]
prices = ["50", "30", "20"]

import pandas as pd
df = pd.DataFrame(columns=["flower_name", "price", "description", "reason"])

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model=get_deepseek_chat_model(), base_url=get_deepseek_api_host())

for flower, price in zip(flowers, prices):
    input = prompt.format(price=price, flower_name=flower)
    output = llm.invoke(input)
    parsed_output = output_parser.parse(output.content).model_dump()
    parsed_output["flower_name"] = flower
    parsed_output["price"] = price
    df.loc[len(df)] = parsed_output

print(df.to_dict(orient="records"))
output_path = os.path.join(os.path.dirname(__file__), "flowers_with_descriptions.csv")
df.to_csv(output_path, index=False)