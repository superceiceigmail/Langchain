import importlib

def test_quickstart_runs_with_fake_chain(monkeypatch, capsys):
    """
    Replace the module's LLMChain with a FakeChain to avoid network calls,
    then run quickstart.main() and assert expected output is printed.
    """
    mod_name = "src.langchain_practice.quickstart"
    if mod_name in importlib.sys.modules:
        importlib.reload(importlib.import_module(mod_name))
    qs = importlib.import_module(mod_name)

    # Ensure script sees a key or uses fake chain; here we force Fake behavior
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("USE_FAKE_LLM", "1")

    # Define a FakeChain to replace LLMChain if needed by module-level code
    class FakeChainLocal:
        def __init__(self, llm=None, prompt=None):
            self.prompt = prompt
        def run(self, **kwargs):
            # simulate expected answer
            return "这是测试回答"

    # Patch the LLMChain symbol used in the module to our fake
    monkeypatch.setattr(qs, "LLMChain", FakeChainLocal, raising=False)
    # Ensure PromptTemplate exists in module (some versions may require it)
    if not hasattr(qs, "PromptTemplate"):
        class DummyPrompt:
            @staticmethod
            def from_template(t): return t
        monkeypatch.setattr(qs, "PromptTemplate", DummyPrompt, raising=False)

    # Run main (should use FakeChain and not call OpenAI)
    qs.main()
    captured = capsys.readouterr()
    assert "Answer:" in captured.out
    assert "这是测试回答" in captured.out