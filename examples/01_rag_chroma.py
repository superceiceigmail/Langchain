"""
A minimal RAG example using OpenAI embeddings + Chroma.
- load some texts (here we synthesize)
- create embeddings, store in Chroma
- do a simple retrieval + LLM answer
Requires OPENAI_API_KEY in env.
"""
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.schema import Document

def build_vectorstore():
    # For demo we create a few simple documents. In langchain_practice load files with DocumentLoaders.
    docs = [
        Document(page_content="LangChain 是一个用于构建基于 LLM 应用的框架。它把 prompt、chains、agents、memory 组合在一起。"),
        Document(page_content="Chroma 是一个轻量的向量数据库，常用于本地开发与测试。"),
        Document(page_content="Embeddings 将文本转换为向量，常用 OpenAI 的 embedding 接口或本地模型。"),
    ]

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(docs, embedding=embeddings, collection_name="demo_collection")
    return vectordb

def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Please set OPENAI_API_KEY")

    vectordb = build_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    llm = ChatOpenAI(temperature=0)
    prompt = PromptTemplate.from_template(
        "Use the retrieved documents to answer the question succinctly.\n\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs={"prompt": prompt})

    query = "什么是 Chroma？"
    res = qa(query)
    print("Answer:", res["result"].strip())
    print("\nSource documents:")
    for doc in res.get("source_documents", []):
        print("-", doc.page_content[:120])

if __name__ == "__main__":
    main()