from langchain_learn.llm_client import get_deepseek_client, get_deepseek_chat_model


#---- completions demo ----
# print("**** completions demo ****")

# completion_client = get_deepseek_client(use_beta=True)
# completion_response = completion_client.completions.create(
#     model = get_deepseek_chat_model(),
#     prompt = "请给我的花店起个名",
#     max_tokens = 128,
# )
# print(completion_response.choices[0].text.strip())

#---- chat demo ----
print("**** chat demo ****")

chat_client = get_deepseek_client(use_beta=False)
chat_response = chat_client.chat.completions.create(
    model=get_deepseek_chat_model(),
    temperature=0.5,
    messages=[
        {"role": "system", "content": "You are a creative AI."},
        {"role": "user", "content": "请给我的花店起个名"},
    ]
)
print(chat_response.choices[0].message.content)
print(chat_response)