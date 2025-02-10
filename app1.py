import getpass
import os
from langchain.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings  # 更新导入路径
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 导入文本分割器

# 设置 API 密钥
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

# 载入 Markdown 文档
loader = DirectoryLoader("docs", glob="**/*.md")
documents = loader.load()

# 使用 RecursiveCharacterTextSplitter 将文档分块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_documents = text_splitter.split_documents(documents)

# 使用 OpenAI Embedding 模型为每个分块文档生成向量
embeddings = OpenAIEmbeddings()
doc_embeddings = embeddings.embed_documents([doc.page_content for doc in split_documents])

# 使用 FAISS 建立一个向量数据库
vector_store = FAISS.from_documents(split_documents, embeddings)

# 获取问题的嵌入
question = "OpCode.NEWBUFFER 需要多少Datoshi手续费"
question_embedding = embeddings.embed_query(question)

# 检索与问题最相关的文档
similar_docs = vector_store.similarity_search_by_vector(question_embedding, k=3)  # k=3表示检索最相关的3个文档

# 拼接相关文档的内容
docs_text = "\n".join([doc.page_content for doc in similar_docs])

# 初始化 OpenAI 模型
model = ChatOpenAI(model="gpt-4o-mini")

# 设置提示模板
prompt = PromptTemplate(template="Given the following information: {docs}\nAnswer this question: {question}", input_variables=["docs", "question"])

# 创建链条
chain = LLMChain(llm=model, prompt=prompt)

# 获取模型响应
response = chain.run(docs=docs_text, question=question)

# 输出响应
print(response)
