import os
from llm_client import get_deepseek_api_key

os.environ["OPENAI_API_KEY"] = get_deepseek_api_key()

# 1. Load 导入Document Loaders
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_qdrant import QdrantVectorStore

# 加载Documents（相对脚本目录，不依赖当前工作目录）
print("[1/5] Loading documents...")
base_dir = os.path.join(os.path.dirname(__file__), "OneFlower")
documents = []
for file in os.listdir(base_dir):
    # 构建完整文件路径
    file_path = os.path.join(base_dir, file)
    if file.endswith('.pdf'):
        loaded = PyPDFLoader(file_path).load()
        documents.extend(loaded)
    elif file.endswith('.docx'):
        loaded = Docx2txtLoader(file_path).load()
        documents.extend(loaded)
    elif file.endswith('.txt'):
        loaded = TextLoader(file_path).load()
        documents.extend(loaded)

print(f"\nTotal: {len(documents)} document chunk(s) from {base_dir}")
if documents:
    print(f"Preview: {documents[0].page_content[:100].strip()}...")

# 2. Split 将Documents切分成块以便后续进行嵌入和向量存储
print("[2/5] Splitting documents...")

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=10,
)
chunked_documents = text_splitter.split_documents(documents)
print(f"\nTotal: {len(chunked_documents)} document chunk(s) from {base_dir}")
print(f"Preview: {chunked_documents[0].page_content[:100].strip()}...")

# 3. Embed 用 DashScope 的 text-embedding-v4 做向量嵌入
print("[3/5] Creating embeddings and vector store...")
embeddings = DashScopeEmbeddings(model="text-embedding-v4")
sample_vector = embeddings.embed_query(chunked_documents[0].page_content)
print(f"Embedding dim: {len(sample_vector)}")
vectorstore = QdrantVectorStore.from_documents(
    documents=chunked_documents,
    embedding=embeddings,
    location=":memory:",
    collection_name="my_documents",
)
print("[3/5] Vector store ready")

# 4. Retrieval 准备模型和Retrieval链
import logging
from langchain_openai.chat_models import ChatOpenAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.chains import RetrievalQA # RetrievalQA链，已经过时，后面写个新的

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger('langchain_classic.retrievers.multi_query').setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.info("[4/5] Building LLM, retriever, and RetrievalQA chain...")

# 实例化一个model
from langchain_learn.llm_client import get_deepseek_chat_model, get_deepseek_api_host
llm = ChatOpenAI(
    model = get_deepseek_chat_model(),
    base_url = get_deepseek_api_host(),
    max_completion_tokens = 60,
    temperature = 0,
)

# 实例化一个MultiQueryRetriever
retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=llm)

# 实例化一个RetrievalQA链
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever_from_llm)
logger.info("[4/5] QA chain ready")

# 5. Output 问答系统的UT实现
from flask import Flask, request, render_template
app = Flask(__name__) 

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':

        # 接收用户输入作为问题
        question = request.form.get('question')
        logger.info("Question received: %s", question)
        
        # RetrievalQA链 - 读入问题，生成答案
        result = qa_chain({"query": question})
        
        # 把大模型的回答结果返回网页进行渲染
        return render_template('index.html', result=result)
    
    return render_template('index.html')

if __name__ == "__main__":
    logger.info("[5/5] Starting Flask server on http://0.0.0.0:15000")
    app.run(host='0.0.0.0',debug=True,port=15000)
