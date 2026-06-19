import os

from llm_client import get_huggingface_api_token

os.environ["HUGGINGFACEHUB_API_TOKEN"] = get_huggingface_api_token()

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# 支持 text-generation 的 HF 模型（走 text_generation，非 conversational）
# 实测 mistralai/Mistral-7B-v0.1 在 HF Serverless API 上可用，问答补全效果较好
# 注意：多数 *-Instruct / *-it 模型在 HF API 上只支持 conversational，需用 ChatHuggingFace
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-v0.1",
    huggingfacehub_api_token=get_huggingface_api_token(),
    max_new_tokens=128,
    temperature=0.7,
)

template = """
Question: {question}
Answer:
"""

prompt = PromptTemplate.from_template(template)
chain = prompt | llm | StrOutputParser()

question = "Rose is which type of flower?"

print(chain.invoke({"question": question}))
