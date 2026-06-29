from langchain_learn.llm_client import create_deepseek_chat

llm = create_deepseek_chat(max_completion_tokens=200)

response = llm.invoke("请给我写一句情人节红玫瑰的中文宣传语")
print(response.content)
