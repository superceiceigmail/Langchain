"""
A minimal RAG example using OpenAI embeddings + Chroma.
Supports loading key from openai.json -> .env -> environment.
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv

def load_key_from_openai_json(path: str = "openai.json"):
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    for k in ("key", "api_key", "OPENAI_API_KEY"):
        if isinstance(data.get(k), str) and data.get(k).startswith("sk-"):
            return data.get(k)
    return None

# Prefer openai.json
key = load_key_from_openai_json()
if key:
    os.environ["OPENAI_API_KEY"] = key

# fallback to .env
load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

def build_vectorstore(persist_dir=None):
    docs = [
        Document(page_content="LangChain 是一个用于构建基于 LLM 应用的框架。它把 prompt、chains、agents、memory 组合在一起。"),
        Document(page_content="Chroma 是一个轻量的向量数据库，常用于本地开发与测试。"),
        Document(page_content="Embeddings 将文本转换为向量，常用 OpenAI 的 embedding 接口或本地模型。"),
    ]

    embeddings = OpenAIEmbeddings()
    if persist_dir:
        vectordb = Chroma.from_documents(docs, embedding=embeddings, collection_name="demo_collection", persist_directory=persist_dir)
        try:
            vectordb.persist()
        except Exception:
            pass
    else:
        vectordb = Chroma.from_documents(docs, embedding=embeddings, collection_name="demo_collection")
    return vectordb

def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Please set OPENAI_API_KEY (openai.json, .env or env variable)")

    persist_dir = os.environ.get("CHROMA_PERSIST_PATH")  # optional
    vectordb = build_vectorstore(persist_dir)
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    model_name = os.environ.get("MODEL_NAME")
    llm_kwargs = {"temperature": 0}
    if model_name:
        llm_kwargs["model"] = model_name
    llm = ChatOpenAI(**llm_kwargs)

    prompt = PromptTemplate.from_template(
        "Use the retrieved documents to answer the question succinctly.\n\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs={"prompt": prompt})

    query = "什么是 Chroma？"
    res = qa(query)
    print("Answer:", res.get("result", "").strip())
    print("\nSource documents:")
    for doc in res.get("source_documents", []):
        print("-", doc.page_content[:200])

if __name__ == "__main__":
    main()