import json

import pandas as pd
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from langchain_learn.llm_client import create_volcano_chat

# -- part 1，init model
llm = create_volcano_chat()

# -- part 2
# Create a empty dataframe to save the results
df = pd.DataFrame(columns=["flower_type", "price", "description", "reason"])

# prepare data
flowers = ["玫瑰", "百合", "康乃馨"]
prices = ["50", "30", "20"]

# Define the data structure we want
class FlowerDescription(BaseModel):
    flower_type: str = Field(description="鲜花的种类")
    price: int = Field(description="鲜花的价格")
    description: str = Field(description="鲜花的描述文案")
    reason: str = Field(description="为什么要这样写这个文案")

# -- part 3
# Create output parser
output_parser = PydanticOutputParser(pydantic_object=FlowerDescription)

# Get output format instructions
format_instructions = output_parser.get_format_instructions()
print(f"format_instructions: {format_instructions}")

# -- part 4
# Create prompt template
prompt_template = """您是一位专业的鲜花店文案撰写员。对于售价为 {price} 元的 {flower} ，您能提供一个吸引人的简短中文描述吗？{format_instructions}"""

# Create prompt by prompt template
prompt = PromptTemplate.from_template(
    prompt_template,
    partial_variables={"format_instructions": format_instructions},
)
print(f"prompt: {prompt}")

# -- part 5
for flower, price in zip(flowers, prices):
    # create input
    input = prompt.format(flower=flower, price=price)
    # print input
    print(f"input: {input}")

    # get llm output
    output = llm.invoke(input)
    print(f"output: {output}")

    # parse output
    parsed_output = output_parser.parse(output.content)
    print(f"parsed_output: {parsed_output}")

    # Add to dataframe
    df.loc[len(df)] = parsed_output.model_dump()

print("Data output:")
print(json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2))
