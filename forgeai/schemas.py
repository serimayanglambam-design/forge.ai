from pydantic import BaseModel, Field
from typing import Literal, Optional

class Requirements(BaseModel):
    scope: str
    audience: str
    tone: str
    must_answer_topics: list[str]
    must_refuse_topics: list[str]
    unsupported_topics: list[str] = []
    groundability_notes: str = ""

class RetrievalConfig(BaseModel):
    method: Literal["bm25", "embedding", "hybrid"] = "hybrid"
    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k: int = 4

class OutputContract(BaseModel):
    require_citations: bool = True
    refusal_phrase: str = "I can't answer that from the provided documents."
    max_words: int = 150

class AgentDefinition(BaseModel):
    name: str
    system_prompt: str
    retrieval: RetrievalConfig
    tools: list[str]
    tool_justification: dict[str, str]
    output_contract: OutputContract
    version: int = 1

class EvalCase(BaseModel):
    id: str
    category: Literal["answerable", "out_of_scope", "edge_ambiguous", "edge_multidoc"]
    question: str
    expected_sources: list[str] = []
    expected_key_facts: list[str] = []
    should_refuse: bool = False

class GenItem(BaseModel):
    question: str
    expected_key_facts: list[str]

class GenItems(BaseModel):
    items: list[GenItem]

class OOSItem(BaseModel):
    question: str
    kind: Literal["near_miss", "off_topic"]

class OOSItems(BaseModel):
    items: list[OOSItem]

class CaseResult(BaseModel):
    case_id: str
    answer: str
    retrieved_sources: list[str]
    grounded: bool
    correct: bool
    refusal_ok: bool
    failure_category: Optional[Literal[
        "retrieval_miss", "prompt_defect", "hallucination", "wrong_refusal"
    ]] = None

class Change(BaseModel):
    iteration: int
    field_changed: str
    old_value: str
    new_value: str
    reason: str
    failures_addressed: list[str]
    score_before: float
    score_after: Optional[float] = None