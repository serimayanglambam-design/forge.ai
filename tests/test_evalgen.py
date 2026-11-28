from pathlib import Path

import pytest

from forgeai import evalgen, llm
from forgeai.loader import Doc
from forgeai.schemas import GenItem, GenItems, OOSItem, OOSItems

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "forgeai" / "prompts"


def _documents() -> list[Doc]:
    return [
        Doc(
            doc_id=f"doc_{index}.txt",
            kind="text",
            text=(f"Document {index} describes its subject in detail. " * 12),
            path=f"doc_{index}.txt",
        )
        for index in range(3)
    ]


def _mock_call_json(
    prompt_file: str,
    variables: dict[str, str | int],
    schema: type[GenItems] | type[OOSItems],
) -> GenItems | OOSItems:
    if schema is OOSItems:
        return OOSItems(
            items=[
                OOSItem(question=f"Unsupported question {index}?", kind="near_miss")
                for index in range(int(variables["n"]))
            ]
        )
    return GenItems(
        items=[
            GenItem(
                question=f"Question for {variables.get('doc_id', variables.get('doc_id_a', 'source'))}?",
                expected_key_facts=["explicit detail"],
            )
        ]
    )


def test_generates_unique_ids_sources_and_refusals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evalgen, "call_json", _mock_call_json)

    cases, provenance = evalgen.generate_evals_with_provenance(
        _documents(),
        requirements=None,
        n_target=12,
        seed=7,
    )

    assert len({case.id for case in cases}) == len(cases)
    assert all(
        any(doc.doc_id in case.expected_sources for case in cases)
        for doc in _documents()
    )
    assert all(
        case.should_refuse
        for case in cases
        if case.category == "out_of_scope"
    )
    assert all(case.id in provenance for case in cases)
    assert all(
        provenance[case.id] == []
        for case in cases
        if case.category == "out_of_scope"
    )


def test_failed_generation_call_is_logged_and_skipped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    original_call_json = _mock_call_json
    failed = False

    def fail_once(
        prompt_file: str,
        variables: dict[str, str | int],
        schema: type[GenItems] | type[OOSItems],
    ) -> GenItems | OOSItems:
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("temporary generation failure")
        return original_call_json(prompt_file, variables, schema)

    monkeypatch.setattr(evalgen, "call_json", fail_once)

    cases = evalgen.generate_evals(_documents(), requirements=None, n_target=12)

    assert cases
    assert failed
    assert "Evaluation generation call failed" in caplog.text


def test_raises_when_more_than_half_of_generation_calls_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def always_fail(
        prompt_file: str,
        variables: dict[str, str | int],
        schema: type[GenItems] | type[OOSItems],
    ) -> GenItems | OOSItems:
        raise RuntimeError("first generation error")

    monkeypatch.setattr(evalgen, "call_json", always_fail)

    with pytest.raises(RuntimeError, match="first generation error"):
        evalgen.generate_evals(_documents(), requirements=None, n_target=12)


def test_evalgen_prompts_render_without_unresolved_placeholders() -> None:
    prompt_variables: dict[str, dict[str, str | int]] = {
        "evalgen_answerable.md": {
            "n": 1,
            "doc_id": "policy.txt",
            "chunk_text": "The passage text.",
        },
        "evalgen_multidoc.md": {
            "doc_id_a": "a.txt",
            "chunk_a": "Passage A.",
            "doc_id_b": "b.txt",
            "chunk_b": "Passage B.",
        },
        "evalgen_ambiguous.md": {
            "doc_id": "policy.txt",
            "chunk_text": "The passage text.",
        },
        "evalgen_out_of_scope.md": {
            "scope": "a document assistant",
            "audience": "general users",
            "doc_overview": "Document overview.",
            "n": 3,
        },
    }

    for prompt_file, variables in prompt_variables.items():
        assert (PROMPTS_DIR / prompt_file).is_file()
        assert "{{" not in llm.render(prompt_file, variables)
