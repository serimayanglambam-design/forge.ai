from pathlib import Path

import pytest

from forgeai.loader import Chunk
from forgeai.retrieval import Retriever
from forgeai.schemas import RetrievalConfig


def _topic_chunks() -> list[Chunk]:
    topics = [
        ("astronomy", "Stars form galaxies and planets orbit distant suns."),
        ("cooking", "Fresh basil and tomatoes make a flavorful pasta sauce."),
        ("quantum", "Quantum entanglement links photon states across distance."),
        ("ocean", "Coral reefs support diverse fish and marine ecosystems."),
        ("football", "The striker scored a goal during the championship match."),
    ]
    return [
        Chunk(
            chunk_id=f"{doc_id}::0",
            doc_id=doc_id,
            text=text,
        )
        for doc_id, text in topics
    ]


def test_bm25_ranks_the_relevant_chunk_first() -> None:
    retriever = Retriever(
        _topic_chunks(),
        RetrievalConfig(method="bm25", top_k=3),
    )

    results = retriever.search("quantum entanglement photon")

    assert results[0][0].doc_id == "quantum"
    assert len(results) == 3
    assert retriever.retrieved_doc_ids("quantum entanglement photon")[0] == "quantum"


def test_embedding_failure_falls_back_to_bm25(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail_embedding(**kwargs: object) -> None:
        raise ConnectionError("Ollama is unavailable")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("forgeai.retrieval.ollama.embed", fail_embedding)
    retriever = Retriever(
        _topic_chunks(),
        RetrievalConfig(method="embedding", top_k=1),
    )

    results = retriever.search("quantum entanglement photon")

    assert results[0][0].doc_id == "quantum"
    assert "falling back to BM25" in caplog.text


@pytest.mark.parametrize("method", ["embedding", "hybrid"])
@pytest.mark.ollama
def test_ollama_retrieval_methods(
    method: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    retriever = Retriever(
        _topic_chunks(),
        RetrievalConfig(method=method, top_k=3),
    )

    results = retriever.search("quantum entanglement photon")

    assert results[0][0].doc_id == "quantum"
    assert len(results) == 3
