import pytest


@pytest.fixture(autouse=True)
def _isolated_logs(tmp_path, monkeypatch):
    # keep test runs out of logs/runs.jsonl
    import agent.graph

    monkeypatch.setattr(agent.graph, "LOG_DIR", tmp_path)
