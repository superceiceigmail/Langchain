# Replace the module's FakeChain (not LLMChain) so quickstart.main() uses our fake implementation.
import importlib

def test_quickstart_runs_with_fake_chain(monkeypatch, capsys):
    """
    Replace the module's FakeChain to avoid network calls,
    then run quickstart.main() and assert expected output is printed.
    """
    mod_name = "src.langchain_practice.quickstart"
    # Reload module if already imported to ensure a clean state
    if mod_name in importlib.sys.modules:
        importlib.reload(importlib.import_module(mod_name))
    qs = importlib.import_module(mod_name)

    # Ensure script uses fake chain path
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("USE_FAKE_LLM", "1")

    # Define a FakeChain to replace the module's FakeChain
    class FakeChainLocal:
        def __init__(self, llm=None, prompt=None):
            self.prompt = prompt
        def run(self, **kwargs):
            # simulate expected answer
            return "这是测试回答"

    # Patch the module's FakeChain symbol to our fake
    monkeypatch.setattr(qs, "FakeChain", FakeChainLocal, raising=False)

    # Ensure PromptTemplate exists in module (some langchain versions may require it)
    if not hasattr(qs, "PromptTemplate"):
        class DummyPrompt:
            @staticmethod
            def from_template(t): return t
        monkeypatch.setattr(qs, "PromptTemplate", DummyPrompt, raising=False)

    # Run main (should use patched FakeChain and not call OpenAI)
    qs.main()
    captured = capsys.readouterr()
    assert "Answer:" in captured.out
    assert "这是测试回答" in captured.out