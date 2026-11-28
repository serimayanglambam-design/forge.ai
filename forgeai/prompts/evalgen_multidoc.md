You are a QA engineer writing test questions for a document-based assistant.

Below are TWO passages from different documents. Write 1 question that can only be answered completely by combining information from BOTH passages.

Rules:
- Each passage must contribute at least one key fact.
- "expected_key_facts": 2 to 4 short facts, each copied EXACTLY from one of the passages, each under 8 words.
- The question must make sense on its own. Do not mention "passage" or "document".

Passage A (source: {{doc_id_a}}):
"""
{{chunk_a}}
"""

Passage B (source: {{doc_id_b}}):
"""
{{chunk_b}}
"""

Return JSON only.
