# ForgeAI project rules
- Python 3.11+, type hints everywhere, Pydantic v2 for ALL LLM outputs.
- Every LLM call goes through forgeai/llm.py::call_json(prompt_file, vars, schema, retries=3). It validates against the Pydantic schema and retries with the error appended.
- Prompts live ONLY in forgeai/prompts/*.md. Never inline prompts in code.
- Nothing is hardcoded to a specific domain (HR, ships, etc). The code must work on unseen briefs and document folders.
- Every pipeline stage writes its output as JSON to runs/<run_name>/.
- Deterministic checks come before LLM-judge checks.
- Model names come from forgeai/config.py (env vars), never hardcoded elsewhere.
- Never use eval() or exec().
