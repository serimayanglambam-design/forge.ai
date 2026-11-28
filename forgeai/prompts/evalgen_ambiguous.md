You are a QA engineer writing hard test questions for a document-based assistant.

Below is ONE passage. Write 1 question about its topic that a real user might phrase vaguely or informally: short, missing context, using a different word than the passage does (for example "time off" instead of "annual leave"). The passage must still contain the answer.

Rules:
- "expected_key_facts": 1 to 3 short facts copied EXACTLY from the passage, each under 8 words.
- Do not copy phrases from the passage into the question.

Passage (source: {{doc_id}}):
"""
{{chunk_text}}
"""

Return JSON only.
