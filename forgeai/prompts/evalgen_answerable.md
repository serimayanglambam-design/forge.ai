You are a QA engineer writing test questions for a document-based assistant.

Below is ONE passage from a document. Write {{n}} question(s) that a real user might ask and that this passage answers directly.

Rules:
- The answer must be stated explicitly in the passage. Do not use outside knowledge.
- For each question, give "expected_key_facts": 1 to 3 short facts (numbers, names, durations, conditions) that a correct answer must contain.
- Each key fact must be copied EXACTLY as it appears in the passage (same digits, same wording). Keep each under 8 words.
- Questions must make sense on their own. Never write "according to the passage" or "in this document".
- Do not ask yes/no questions.

Passage (source: {{doc_id}}):
"""
{{chunk_text}}
"""

Return JSON only.
