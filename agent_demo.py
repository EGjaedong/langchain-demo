import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.agents import create_agent
from langchain.tools import BaseTool
from langchain_core.messages import HumanMessage

from langchain_learn.llm_client import create_deepseek_chat

# ---- Part I 初始化图像字幕生成模型
# 指定要使用的工具模型（HuggingFace中的image-caption模型）
hf_model = "Salesforce/blip-image-captioning-base"

# 初始化处理器和工具模型
processor = BlipProcessor.from_pretrained(hf_model)
model = BlipForConditionalGeneration.from_pretrained(hf_model)

# ---- Part II 定义图像字幕生成工具类
class ImageCapTool(BaseTool):
    name: str = "image_captioner"
    description: str = "为图片创作说明文案."

    def _run(self, url: str):
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=20)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async execution")

# ---- Part III 初始化并运行 LangChain 智能代理
llm = create_deepseek_chat(
    max_completion_tokens=200,
    temperature=0.2,
)

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

img_url = "https://mir-s3-cdn-cf.behance.net/project_modules/hd/eec79e20058499.563190744f903.jpg"
agent.invoke({"messages": [HumanMessage(content=f"{img_url}\n请创作合适的中文推广文案")]})
