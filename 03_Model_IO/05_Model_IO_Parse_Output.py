import os

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
import pandas as pd

from langchain_learn.llm_client import create_deepseek_chat

template = """您是一位专业的鲜花店文案撰写员。
对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？
{format_instructions}"""


class FlowerCopy(BaseModel):
    description: str = Field(description="鲜花的描述文案")
    reason: str = Field(description="为什么要这样写这个文案")


output_parser = PydanticOutputParser(pydantic_object=FlowerCopy)
format_instructions = output_parser.get_format_instructions()

prompt = PromptTemplate.from_template(
    template,
    partial_variables={"format_instructions": format_instructions},
)
print(prompt)

flowers = ["玫瑰", "百合", "康乃馨"]
prices = ["50", "30", "20"]

df = pd.DataFrame(columns=["flower_name", "price", "description", "reason"])

llm = create_deepseek_chat()

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
