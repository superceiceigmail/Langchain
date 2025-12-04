"""
Basic quickstart for LangChain (LLMChain + PromptTemplate + ChatOpenAI)

Behavior changes:
- Loads OpenAI key from openai.json (preferred) -> .env -> env var OPENAI_API_KEY.
- If no key and USE_FAKE_LLM != "0", it will use a FakeChain so you can run demos without OpenAI.
- Catches OpenAI quota/rate errors and prints a friendly message.
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
        v = data.get(k)
        if isinstance(v, str) and v.startswith("sk-"):
            return v
    return None

# 1) Try openai.json first
key_from_file = load_key_from_openai_json()
if key_from_file:
    os.environ["OPENAI_API_KEY"] = key_from_file

# 2) Load .env (if present). This will not override openai.json result above.
load_dotenv()

# Attempt normal imports (may produce deprecation warnings depending on langchain version)
try:
    from langchain.chat_models import ChatOpenAI  # type: ignore
except Exception:
    ChatOpenAI = None  # may be provided via langchain-community or other wrapper

# Import LLMChain / PromptTemplate from langchain; tests will monkeypatch LLMChain where needed
try:
    from langchain import LLMChain, PromptTemplate
except Exception:
    # fallback: try common alternative imports (some langchain versions warn about root imports)
    try:
        from langchain.chains import LLMChain  # type: ignore
    except Exception:
        LLMChain = None
    try:
        from langchain_core.prompts import PromptTemplate  # type: ignore
    except Exception:
        PromptTemplate = None

# A minimal FakeChain that mimics LLMChain.run(user_input=...)
class FakeChain:
    def __init__(self, llm=None, prompt=None):
        self.prompt = prompt

    def run(self, **kwargs):
        # kwargs might contain 'user_input' (our quickstart uses that)
        user_input = kwargs.get("user_input") or kwargs.get("input") or ""
        # simple deterministic mapping for demo/test purposes
        if "区块链" in user_input:
            return "区块链的核心思想是去中心化与不可篡改的账本。"
        if "Chroma" in user_input or "chroma" in user_input:
            return "Chroma 是一个轻量的本地向量数据库。"
        return "这是一个本地模拟回答（FakeChain）"

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    use_fake = os.environ.get("USE_FAKE_LLM", "1") != "0"

    # Build LLM chain or fake chain depending on key presence / env
    if not api_key and use_fake:
        # Use FakeChain (no network calls)
        chain = FakeChain()
    else:
        if LLMChain is None or PromptTemplate is None:
            raise RuntimeError("LangChain classes not found; consider installing a compatible langchain version.")
        # Construct a chat LLM (temperature small for reproducible outputs)
        llm_kwargs = {"temperature": 0.2}
        model_name = os.environ.get("MODEL_NAME")
        if model_name:
            # some langchain versions use model or model_name param; pass model if accepted
            if ChatOpenAI is not None:
                try:
                    llm = ChatOpenAI(model=model_name, **llm_kwargs)
                except TypeError:
                    llm = ChatOpenAI(model_name=model_name, **llm_kwargs)
            else:
                # If ChatOpenAI isn't available, we'll try to construct via other means (best-effort)
                llm = None
        else:
            llm = ChatOpenAI(**llm_kwargs) if ChatOpenAI is not None else None

        template = """You are a helpful assistant.
Given the following user question, provide a concise answer.

Question: {user_input}

Answer:"""
        prompt = PromptTemplate.from_template(template)
        chain = LLMChain(llm=llm, prompt=prompt)

    # Example run
    question = "用一句话解释区块链的核心思想。"
    try:
        result = chain.run(user_input=question)
    except Exception as e:
        # Friendly handling for OpenAI quota/rate errors or other runtime issues
        try:
            # openai lib historically exposes RateLimitError or raises a generic exception with code 429
            from openai.error import RateLimitError
            if isinstance(e, RateLimitError):
                print("OpenAI RateLimitError：配额或速率受限，请检查你的账号配额与计费信息。")
                return
        except Exception:
            pass
        # Generic fallback message (print the error for debugging)
        msg = str(e)
        if "quota" in msg.lower() or "insufficient_quota" in msg.lower() or "429" in msg:
            print("OpenAI 返回配额错误（insufficient_quota / 429），请检查计费或换 key。")
            return
        print("运行 LLM 时发生错误：", e)
        return

    print("Question:", question)
    print("Answer:", result.strip())

if __name__ == "__main__":
    main()