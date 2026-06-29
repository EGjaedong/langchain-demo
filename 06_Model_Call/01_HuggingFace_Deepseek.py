from langchain_learn.llm_client import configure_huggingface_token

configure_huggingface_token()

from transformers import AutoTokenizer, AutoModelForCausalLM

# 不要真正去运行，这个真的会下载一个deepseek-ai/DeepSeek-V4-Pro的模型，大概100GB左右
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V4-Pro")
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-V4-Pro", device_map="auto"
)

prompt = "请给我讲一个玫瑰的爱情故事"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(inputs["input_ids"], max_new_tokens=2000)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
