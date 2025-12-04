import importlib
import importlib.util
from pathlib import Path
from langchain.schema import Document

def load_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_rag_main_with_mocks(monkeypatch, capsys, tmp_path):
    """
    Load examples/01_rag_chroma.py by file path and monkeypatch its dependencies:
    - Replace build_vectorstore with a fake that returns an object with as_retriever().
    - Replace RetrievalQA.from_chain_type with a fake that returns a callable QA function.
    This avoids calling embeddings/Chroma/OpenAI while testing the main flow.
    """
    repo_root = Path.cwd()
    example_path = repo_root / "examples" / "01_rag_chroma.py"
    assert example_path.exists(), f"{example_path} not found"

    mod = load_module_from_path(example_path, "example_rag")

    # Ensure main will not error on missing key; set a dummy key
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    # Fake vector DB with as_retriever
    class FakeVectorDB:
        def __init__(self, docs):
            self._docs = docs
        def as_retriever(self, **kwargs):
            class R:
                def get_relevant_documents(self_inner, query):
                    return self._docs[:2]
            return R()

    # Monkeypatch build_vectorstore in module to return FakeVectorDB
    fake_docs = [
        Document(page_content="LangChain 是一个用于构建 LLM 应用的框架。"),
        Document(page_content="Chroma 是一个向量数据库。"),
    ]
    monkeypatch.setattr(mod, "build_vectorstore", lambda persist_dir=None: FakeVectorDB(fake_docs))

    # Monkeypatch RetrievalQA.from_chain_type to return a callable that returns predictable result
    def fake_from_chain_type(llm, chain_type, retriever, return_source_documents, chain_type_kwargs):
        def qa_callable(query):
            return {"result": "这是模拟的 RAG 回答", "source_documents": fake_docs}
        return qa_callable
    monkeypatch.setattr(mod, "RetrievalQA", type("R", (), {"from_chain_type": staticmethod(fake_from_chain_type)}))

    # Run main() — should use our fakes and print output
    mod.main()
    captured = capsys.readouterr()
    assert "Answer:" in captured.out
    assert "模拟" in captured.out
    assert "Source documents:" in captured.out