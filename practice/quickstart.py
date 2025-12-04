"""
Basic quickstart for LangChain (LLMChain + PromptTemplate + ChatOpenAI)
Run:
  python -m src.langchain_practice.quickstart
Requires OPENAI_API_KEY in env.
"""
import os

from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY in your environment")

    # Construct a chat LLM (temperature small for reproducible outputs)
    llm = ChatOpenAI(temperature=0.2)

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