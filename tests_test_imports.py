def test_basic_imports():
    # smoke test to ensure langchain can be imported in CI
    import importlib
    importlib.import_module("langchain")
    # Try to import commonly used modules
    importlib.import_module("langchain.chat_models")
    importlib.import_module("langchain.embeddings")