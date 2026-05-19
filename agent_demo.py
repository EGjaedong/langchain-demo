import os
from pydantic import SecretStr
import requests
from PIL import Image
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.tools import BaseTool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from langchain_learn.llm_client import get_deepseek_chat_model

load_dotenv()

#---- Part I 初始化图像字幕生成模型# 指定要使用的工具模型（HuggingFace中的image-caption模型）
# 指定要使用的工具模型（HuggingFace中的image-caption模型）
hf_model = "Salesforce/blip-image-captioning-base"

# 初始化处理器和工具模型
# 预处理器将准备图像供模型使用
processor = BlipProcessor.from_pretrained(hf_model)
# 然后初始化工具模型本身
model = BlipForConditionalGeneration.from_pretrained(hf_model)

#---- Part II 定义图像字幕生成工具类
class ImageCapTool(BaseTool):
    name: str = "image_captioner"
    description: str = "为图片创作说明文案."

    def _run(self, url: str):
        # 下载图像并将其转换为PIL对象
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        # 预处理图像
        inputs = processor(image, return_tensors="pt")
        # 生成字幕
        out = model.generate(**inputs, max_new_tokens=20)
        # 获取字幕
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async execution")

#---- PartIII 初始化并运行 LangChain 智能代理
# 设置OpenAi的API迷药并初始化大语言模型(DeepSeek的Text模型)
llm = ChatOpenAI(
    model=get_deepseek_chat_model(),
    api_key=SecretStr(os.environ["DEEPSEEK_API_KEY"]),
    base_url=os.environ["DEEPSEEK_API_HOST"],
    max_completion_tokens=200,
    temperature=0.2,
)

# 创建工具初始化智能代理并运行它
tools = [
    ImageCapTool(
        name="image_captioner",
        description="为图片创作说明文案.",
    )
]
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="你是一个中文营销文案助手。需要理解图片内容时，先调用工具生成图片说明，再基于说明创作推广文案。",
    debug=True,
)

img_url = 'https://mir-s3-cdn-cf.behance.net/project_modules/hd/eec79e20058499.563190744f903.jpg'
agent.invoke({"messages": [HumanMessage(content=f"{img_url}\n请创作合适的中文推广文案")]})