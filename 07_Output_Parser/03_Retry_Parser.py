template = """Based on the user question, provide an Action and Action Input for what step should be taken.
{format_instructions}
Question: {query}
Response"""

# Define a Pydantic data struct, define a Action class and it's fields
from langchain_classic.output_parsers.fix import OutputFixingParser
from pydantic import BaseModel, Field
class Action(BaseModel):
    action: str = Field(description="action to take")
    action_input: str = Field(description="input to the action")

from langchain_core.output_parsers import PydanticOutputParser
parser = PydanticOutputParser(pydantic_object=Action)

from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
prompt_value = prompt.format_prompt(query="What are the colors of Orchid?")

bad_response = '{"action": "search"}'
# parser.parse(bad_response) # Will error here because missing a field

from langchain_learn.llm_client import create_volcano_chat
llm = create_volcano_chat()

fix_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
parser_result = fix_parser.parse(bad_response)
print('The parse result of OutputFixingParser: ', parser_result)

from langchain_classic.output_parsers import RetryWithErrorOutputParser
retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=llm)
parse_result = retry_parser.parse_with_prompt(bad_response, prompt_value)
print('The parse result of RetryWithErrorOutputParser: ', parse_result)