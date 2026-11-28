"""Generate evaluation questions from document chunks."""

import json
import logging
import random
from pathlib import Path

from forgeai.loader import Doc, Chunk, chunk
from forgeai.llm import call_json
from forgeai.schemas import (
    EvalCase,
    GenItem,
    GenItems,
    OOSItem,
    OOSItems,
    Requirements,
)

logger = logging.getLogger(__name__)

_PROMPT_ANSWERABLE = "evalgen_answerable.md"
_PROMPT_MULTIDOC = "evalgen_multidoc.md"
_PROMPT_AMBIGUOUS = "evalgen_ambiguous.md"
_PROMPT_OUT_OF_SCOPE = "evalgen_out_of_scope.md"


def _allocate_counts(n_target: int, available_doc_count: int) -> dict[str, int]:
    if available_doc_count == 0:
        return {
            "answerable": 0,
            "multidoc": 0,
            "ambiguous": 0,
            "out_of_scope": n_target,
        }

    weights = [
        ("answerable", 50),
        ("multidoc", 10 if available_doc_count > 1 else 0),
        ("ambiguous", 15),
        ("out_of_scope", 25),
    ]
    weight_total = sum(weight for _, weight in weights)
    exact_counts = {
        name: n_target * weight / weight_total for name, weight in weights
    }
    counts = {name: int(value) for name, value in exact_counts.items()}
    remainder = n_target - sum(counts.values())
    allocation_order = sorted(
        (name for name, weight in weights if weight),
        key=lambda name: (-(exact_counts[name] - counts[name]), name),
    )
    for name in allocation_order[:remainder]:
        counts[name] += 1

    if counts["answerable"] < available_doc_count:
        additional = available_doc_count - counts["answerable"]
        counts["answerable"] = available_doc_count
        for donor in ("out_of_scope", "ambiguous", "multidoc"):
            reduction = min(additional, counts[donor])
            counts[donor] -= reduction
            additional -= reduction
            if additional == 0:
                break

    return counts


def generate_evals_with_provenance(
    docs: list[Doc],
    requirements: Requirements | None,
    n_target: int = 35,
    seed: int = 0,
) -> tuple[list[EvalCase], dict[str, list[str]]]:
    """Generate evaluation cases and map each case to its source chunk texts."""
    if n_target < 0:
        raise ValueError("n_target must be non-negative")

    rng = random.Random(seed)
    chunks_by_doc: list[tuple[str, list[Chunk]]] = []
    for doc in docs:
        eligible_chunks = [
            chunk_item
            for chunk_item in chunk(doc, size=700, overlap=0)
            if len(chunk_item.text) >= 200
        ]
        if eligible_chunks:
            chunks_by_doc.append((doc.doc_id, eligible_chunks))

    counts = _allocate_counts(n_target, len(chunks_by_doc))
    cases: list[EvalCase] = []
    provenance: dict[str, list[str]] = {}
    id_counts = {"ans": 0, "multi": 0, "amb": 0, "oos": 0}
    total_calls = 0
    failed_calls = 0
    first_error: str | None = None

    def generate(
        prompt_file: str,
        variables: dict[str, str | int],
        schema: type[GenItems] | type[OOSItems],
    ) -> GenItems | OOSItems | None:
        nonlocal total_calls, failed_calls, first_error
        total_calls += 1
        try:
            result = call_json(prompt_file, variables, schema)
            if not result.items:
                raise ValueError(f"{prompt_file} returned no items")
            return result
        except Exception as error:
            failed_calls += 1
            if first_error is None:
                first_error = str(error) or repr(error)
            logger.exception("Evaluation generation call failed for %s", prompt_file)
            return None

    def add_case(
        prefix: str,
        category: str,
        item: GenItem | OOSItem,
        expected_sources: list[str],
        source_texts: list[str],
        should_refuse: bool = False,
    ) -> None:
        id_counts[prefix] += 1
        case_id = f"{prefix}_{id_counts[prefix]:03d}"
        cases.append(
            EvalCase(
                id=case_id,
                category=category,
                question=item.question,
                expected_sources=expected_sources,
                expected_key_facts=(
                    item.expected_key_facts if isinstance(item, GenItem) else []
                ),
                should_refuse=should_refuse,
            )
        )
        provenance[case_id] = source_texts

    if chunks_by_doc:
        doc_order = list(range(len(chunks_by_doc)))
        rng.shuffle(doc_order)
        shuffled_chunks: list[list[Chunk]] = []
        for _, doc_chunks in chunks_by_doc:
            shuffled = list(doc_chunks)
            rng.shuffle(shuffled)
            shuffled_chunks.append(shuffled)
        doc_chunk_positions = [0] * len(chunks_by_doc)

        for sample_index in range(counts["answerable"]):
            doc_index = doc_order[sample_index % len(doc_order)]
            doc_id, _ = chunks_by_doc[doc_index]
            doc_chunks = shuffled_chunks[doc_index]
            if doc_chunk_positions[doc_index] >= len(doc_chunks):
                rng.shuffle(doc_chunks)
                doc_chunk_positions[doc_index] = 0
            chunk_item = doc_chunks[doc_chunk_positions[doc_index]]
            doc_chunk_positions[doc_index] += 1

            generated = generate(
                _PROMPT_ANSWERABLE,
                {"n": 1, "doc_id": doc_id, "chunk_text": chunk_item.text},
                GenItems,
            )
            if generated is not None:
                add_case(
                    "ans",
                    "answerable",
                    generated.items[0],
                    [doc_id],
                    [chunk_item.text],
                )

        if counts["multidoc"] and len(chunks_by_doc) > 1:
            for _ in range(counts["multidoc"]):
                doc_index_a, doc_index_b = rng.sample(range(len(chunks_by_doc)), 2)
                doc_id_a, chunks_a = chunks_by_doc[doc_index_a]
                doc_id_b, chunks_b = chunks_by_doc[doc_index_b]
                chunk_a = rng.choice(chunks_a)
                chunk_b = rng.choice(chunks_b)
                generated = generate(
                    _PROMPT_MULTIDOC,
                    {
                        "doc_id_a": doc_id_a,
                        "chunk_a": chunk_a.text,
                        "doc_id_b": doc_id_b,
                        "chunk_b": chunk_b.text,
                    },
                    GenItems,
                )
                if generated is not None:
                    add_case(
                        "multi",
                        "edge_multidoc",
                        generated.items[0],
                        [doc_id_a, doc_id_b],
                        [chunk_a.text, chunk_b.text],
                    )

        all_chunks = [
            (doc_id, chunk_item)
            for doc_id, doc_chunks in chunks_by_doc
            for chunk_item in doc_chunks
        ]
        for _ in range(counts["ambiguous"]):
            doc_id, chunk_item = rng.choice(all_chunks)
            generated = generate(
                _PROMPT_AMBIGUOUS,
                {"doc_id": doc_id, "chunk_text": chunk_item.text},
                GenItems,
            )
            if generated is not None:
                add_case(
                    "amb",
                    "edge_ambiguous",
                    generated.items[0],
                    [doc_id],
                    [chunk_item.text],
                )

    doc_overview = "\n".join(
        f"{doc_id}: {chunk_item.text[:150]}"
        for doc_id, doc_chunks in chunks_by_doc
        for chunk_item in doc_chunks
    )
    if counts["out_of_scope"]:
        generated = generate(
            _PROMPT_OUT_OF_SCOPE,
            {
                "scope": requirements.scope if requirements else "a document assistant",
                "audience": requirements.audience if requirements else "general users",
                "doc_overview": doc_overview,
                "n": counts["out_of_scope"],
            },
            OOSItems,
        )
        if generated is not None:
            for item in generated.items[: counts["out_of_scope"]]:
                add_case(
                    "oos",
                    "out_of_scope",
                    item,
                    [],
                    [],
                    should_refuse=True,
                )

    if total_calls and failed_calls > total_calls / 2:
        raise RuntimeError(
            f"More than half of evaluation generation calls failed. "
            f"First error: {first_error}"
        )

    return cases, provenance


def generate_evals(
    docs: list[Doc],
    requirements: Requirements | None,
    n_target: int = 35,
    seed: int = 0,
) -> list[EvalCase]:
    """Generate evaluation cases without returning their provenance mapping."""
    cases, _ = generate_evals_with_provenance(
        docs,
        requirements,
        n_target=n_target,
        seed=seed,
    )
    return cases


def save_evals(
    cases: list[EvalCase],
    provenance: dict[str, list[str]],
    out_dir: str | Path,
) -> None:
    """Write generated evaluation cases and provenance as JSON files."""
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "evals_raw.json").open("w", encoding="utf-8") as output_file:
        json.dump(
            [case.model_dump(mode="json") for case in cases],
            output_file,
            ensure_ascii=False,
            indent=2,
        )
    with (output_path / "provenance.json").open("w", encoding="utf-8") as output_file:
        json.dump(provenance, output_file, ensure_ascii=False, indent=2)
