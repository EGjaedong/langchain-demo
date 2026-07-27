from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# Create a data structure to save the flowers
class Flower(BaseModel):
    name: str = Field(description="The name of the flower")
    colors: List[str] = Field(description="The colors of the flower")

# Define a query flower colors query
flower_query = "Generate the charaters for a random flower."

# Define invalid formatted output
misformatted = "{'name': '康乃馨', 'colors': ['粉红色', '白色', '红色', '紫色', '黄色']}"

# Create a output parser
parser = PydanticOutputParser(pydantic_object=Flower)
# Use the output parser to parse the invalid formatted output
# parser.parse(misformatted) # This line will raise an error

from langchain_learn.llm_client import create_volcano_chat
llm = create_volcano_chat()

from langchain_classic.output_parsers.fix import OutputFixingParser

new_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

result = new_parser.parse(misformatted)
print(result)