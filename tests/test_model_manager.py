import pytest
from assist.model_manager import ModelManager


def test_openai_mapping(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    mm = ModelManager()
    planner, executor = mm.get_llms("gpt-4o", 0.1)
    assert planner.model_name == "gpt-4o"
    assert executor.model_name == "gpt-4o-mini"


def test_ollama_mapping(monkeypatch):
    class DummyManager(ModelManager):
        def _load_ollama_models(self):
            return ["llama3.2:1b", "llama3.2:3b", "llama3.2:8b"]

    mm = DummyManager()
    planner, executor = mm.get_llms("llama3.2:8b", 0.1)
    assert planner.model == "llama3.2:8b"
    assert executor.model == "llama3.2:3b"
