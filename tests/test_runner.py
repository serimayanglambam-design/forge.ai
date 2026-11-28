import json
from pathlib import Path
from typing import Any

import pytest

from forgeai import runner
from forgeai.runner import AgentResponse, _extract_cited_sources, load_agent

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"


def test_load_agent_from_sample_files() -> None:
    agent = load_agent(str(SAMPLE_DIR / "agent.json"), str(SAMPLE_DIR))

    assert agent.definition.name == "Sample Document Assistant"
    assert agent.tools == []
    assert agent.retriever.chunks
    assert all(chunk.doc_id == "leave_policy.txt" for chunk in agent.retriever.chunks)


def test_load_agent_rejects_unknown_tool(tmp_path: Path) -> None:
    definition_path = SAMPLE_DIR / "agent.json"
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    definition["tools"] = ["not_registered"]
    agent_path = tmp_path / "agent.json"
    agent_path.write_text(json.dumps(definition), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown tool 'not_registered'"):
        load_agent(str(agent_path), str(SAMPLE_DIR))


def test_answer_detects_refusal_and_only_reports_cited_retrieved_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refusal = "I CAN'T ANSWER THAT FROM THE PROVIDED DOCUMENTS."

    def fake_chat(**kwargs: Any) -> dict[str, object]:
        assert kwargs["model"] == runner.config.MODEL
        assert "tools" not in kwargs
        return {"message": {"content": refusal, "tool_calls": []}}

    monkeypatch.setattr(runner.ollama, "chat", fake_chat)
    agent = load_agent(str(SAMPLE_DIR / "agent.json"), str(SAMPLE_DIR))

    response = agent.answer("What is the annual leave allowance?")

    assert isinstance(response, AgentResponse)
    assert response.answer == refusal
    assert response.refused is True
    assert response.sources == []
    assert response.retrieved_chunk_ids


def test_answer_reports_only_cited_retrieved_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_chat(**kwargs: Any) -> dict[str, object]:
        messages = kwargs["messages"]
        assert "[source: leave_policy.txt]" in messages[-1]["content"]
        return {
            "message": {
                "content": "The annual allowance is 20 days [leave_policy.txt].",
                "tool_calls": [],
            }
        }

    monkeypatch.setattr(runner.ollama, "chat", fake_chat)
    agent = load_agent(str(SAMPLE_DIR / "agent.json"), str(SAMPLE_DIR))

    response = agent.answer("How much annual leave is provided?")

    assert response.sources == ["leave_policy.txt"]
    assert response.refused is False


@pytest.mark.parametrize(
    "citation",
    [
        "[LEAVE_POLICY.TXT]",
        "[doc_id: leave_policy.txt]",
        "[source: LEAVE_POLICY.TXT]",
    ],
)
def test_extract_cited_sources_accepts_supported_single_id_forms(
    citation: str,
) -> None:
    assert _extract_cited_sources(
        f"See {citation}.", ["leave_policy.txt"]
    ) == ["leave_policy.txt"]


def test_extract_cited_sources_accepts_multiple_ids_and_preserves_appearance_order() -> None:
    answer = (
        "See [source: second.txt, policy.pdf; unknown.csv], "
        "then [doc_id: SECOND.TXT]."
    )

    assert _extract_cited_sources(
        answer, ["policy.pdf", "second.txt"]
    ) == ["second.txt", "policy.pdf"]


def test_extract_cited_sources_ignores_documents_not_retrieved() -> None:
    assert _extract_cited_sources(
        "See [unretrieved.pdf].", ["leave_policy.txt"]
    ) == []


def test_answer_runs_attached_tool_and_appends_tool_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    definition = json.loads(
        (SAMPLE_DIR / "agent.json").read_text(encoding="utf-8")
    )
    definition["tools"] = ["current_date"]
    agent_path = tmp_path / "agent.json"
    agent_path.write_text(json.dumps(definition), encoding="utf-8")
    responses: list[dict[str, object]] = [
        {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "current_date",
                            "arguments": {},
                        }
                    }
                ],
            }
        },
        {"message": {"content": "Today is the date returned by the tool.", "tool_calls": []}},
    ]

    call_count = 0

    def fake_chat(**kwargs: Any) -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        assert kwargs["tools"][0]["function"]["name"] == "current_date"
        if call_count == 2:
            assert any(
                message["role"] == "tool" for message in kwargs["messages"]
            )
        response = responses.pop(0)
        if call_count == 2:
            tool_messages = [
                message for message in kwargs["messages"] if message["role"] == "tool"
            ]
            assert len(tool_messages) == 1
            assert "date" in tool_messages[0]["content"]
        return response

    monkeypatch.setattr(runner.ollama, "chat", fake_chat)
    agent = load_agent(str(agent_path), str(SAMPLE_DIR))

    response = agent.answer("What is today's date?")

    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["name"] == "current_date"
