"""
Basic quickstart for LangChain (LLMChain + PromptTemplate + ChatOpenAI)
Run:
  python -m src.langchain_practice.quickstart

This script supports loading OpenAI key from:
  1) openai.json at project root with format {"key": "sk-..."} (preferred)
  2) .env (via python-dotenv)
  3) environment variable OPENAI_API_KEY
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
    # support a few common keys
    for k in ("key", "api_key", "OPENAI_API_KEY"):
        if isinstance(data.get(k), str) and data.get(k).startswith("sk-"):
            return data.get(k)
    return None

# 1) Try openai.json first
key_from_file = load_key_from_openai_json()
if key_from_file:
    os.environ["OPENAI_API_KEY"] = key_from_file

# 2) Load .env (if present). This will not override openai.json result above.
load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY (openai.json, .env or env variable)")

    model_name = os.environ.get("MODEL_NAME")  # optional, e.g. "gpt-4o-mini"
    llm_kwargs = {"temperature": 0.2}
    if model_name:
        llm_kwargs["model"] = model_name

    # Construct a chat LLM
    llm = ChatOpenAI(**llm_kwargs)

    # A simple prompt template
    template = """You are a helpful assistant.
Given the following user question, provide a concise answer.

Question: {user_input}

Answer:"""
    prompt = PromptTemplate.from_template(template)
    chain = LLMChain(llm=llm, prompt=prompt)

    # Example run
    question = "用一句话解释区块链的核心思想。"
    result = chain.run(user_input=question)
    print("Question:", question)
    print("Answer:", result.strip())

if __name__ == "__main__":
    main()