# 1. 创建一些示例
samples = [
  {
    "flower_type": "玫瑰",
    "occasion": "爱情",
    "ad_copy": "玫瑰，浪漫的象征，是你向心爱的人表达爱意的最佳选择。"
  },
  {
    "flower_type": "康乃馨",
    "occasion": "母亲节",
    "ad_copy": "康乃馨代表着母爱的纯洁与伟大，是母亲节赠送给母亲的完美礼物。"
  },
  {
    "flower_type": "百合",
    "occasion": "庆祝",
    "ad_copy": "百合象征着纯洁与高雅，是你庆祝特殊时刻的理想选择。"
  },
  {
    "flower_type": "向日葵",
    "occasion": "鼓励",
    "ad_copy": "向日葵象征着坚韧和乐观，是你鼓励亲朋好友的最好方式。"
  }
]

# 2. 创建一个提示模版
from langchain_core.prompts import PromptTemplate
template="鲜花类型:{flower_type}\n场合:{occasion}\n文案:{ad_copy}"
prompt_sample = PromptTemplate(
    input_variables=["flower_type", "occasion", "ad_copy"],
    template=template
)
print("prompt_sample:")
print(prompt_sample.format(**samples[0]))
print("-" * 20)

# 3. 创建一个FewShotPromptTemplate对象
from langchain_core.prompts import FewShotPromptTemplate
few_shot_prompt_template = FewShotPromptTemplate(
    examples=samples,
    example_prompt=prompt_sample,
    suffix="鲜花类型：{flower_type}\n场合：{occasion}",
    input_variables=["flower_type", "occasion"],
)
print("few_shot_prompt_template:")
print(few_shot_prompt_template.format(flower_type="野玫瑰", occasion="爱情"))
print("-" * 20)

# 4. 把提示传递给大模型
print("**** Chat with LLM using FewShotPromptTemplate ****")
from langchain_learn.llm_client import create_deepseek_chat

chat = create_deepseek_chat()
result = chat.invoke(few_shot_prompt_template.format(flower_type="野玫瑰", occasion="爱情"))
print(result.content)
print("-" * 20)

# 5. 使用示例选择器
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
embeddings = DashScopeEmbeddings(model="text-embedding-v4")
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=samples,
    embeddings=embeddings,
    vectorstore_cls=Chroma,
    k=1
)
# 创建一个使用示例选择器的 FewShotPrometTemplate 对象
# example selector 会根据embeddings相似度选择最相似的示例
few_shot_prompt_template = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=prompt_sample,
    suffix="鲜花类型：{flower_type}\n场合：{occasion}",
    input_variables=["flower_type", "occasion"],
)
print("few_shot_prompt_template:")
print(few_shot_prompt_template.format(flower_type="红玫瑰", occasion="爱情"))
print("-" * 20)

print("****Chat with LLM using FewShotPromptTemplate with SemanticSimilarityExampleSelector ****")
result = chat.invoke(few_shot_prompt_template.format(flower_type="红玫瑰", occasion="爱情"))
print(result.content)
print("-" * 20)