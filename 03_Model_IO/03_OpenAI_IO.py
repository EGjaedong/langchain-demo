import os
from llm_client import get_deepseek_api_key, get_deepseek_client, get_deepseek_chat_model

os.environ["OPENAI_API_KEY"] = get_deepseek_api_key()

chat_client = get_deepseek_client(use_beta=False)

prompt_text = "您是一位专业的鲜花店文案撰写员。对于售价为{}元的{}，您能提供一个吸引人的简短描述吗？" # 设置提示

flowers = ["玫瑰", "百合", "康乃馨"]
prices = ["50", "30", "20"]

# 循环调用Text模型的Completion方法，生成文案
for flower, price in zip(flowers, prices):
    prompt = prompt_text.format(price, flower)
    response = chat_client.chat.completions.create(
        model=get_deepseek_chat_model(),
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=256
    )
    print(response.choices[0].message.content) # 输出文案