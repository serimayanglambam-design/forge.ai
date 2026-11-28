You are a QA engineer writing test questions for a document-based assistant.

The assistant is scoped as follows:
Scope: {{scope}}
Audience: {{audience}}

Its documents cover these topics (excerpt of headings and first lines):
{{doc_overview}}

Write {{n}} questions that this audience might plausibly ask, but that the documents CANNOT answer. Mix two kinds:
1. Near-miss: related to the scope, but asks for a detail the documents do not contain (for example a specific number, name or rule that is not there).
2. Off-topic: unrelated to the scope.

Rules:
- Do not ask anything the overview above clearly answers.
- Questions must be specific and realistic, not "what is the meaning of life".
- Label each question "kind": "near_miss" or "off_topic".

Return JSON only.
