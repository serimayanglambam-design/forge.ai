import json, time
from pathlib import Path
from typing import Type, TypeVar
import ollama
from pydantic import BaseModel, ValidationError
from . import config

T = TypeVar("T", bound=BaseModel)
PROMPTS = Path(__file__).parent / "prompts"
LOG_PATH: Path | None = None

def set_log(path: Path):
    global LOG_PATH
    LOG_PATH = path
    path.parent.mkdir(parents=True, exist_ok=True)

def render(prompt_file: str, vars: dict) -> str:
    text = (PROMPTS / prompt_file).read_text(encoding="utf-8")
    for k, v in vars.items():
        text = text.replace("{{" + k + "}}", str(v))
    return text

def _log(entry: dict):
    if LOG_PATH:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def call_json(prompt_file: str, vars: dict, schema: Type[T],
              retries: int = 3, temperature: float | None = None) -> T:
    prompt = render(prompt_file, vars)
    messages = [{"role": "user", "content": prompt}]
    last_err = ""
    for attempt in range(1, retries + 1):
        t0 = time.time()
        resp = ollama.chat(
            model=config.MODEL,
            messages=messages,
            format=schema.model_json_schema(),
            options={"temperature": temperature if temperature is not None else config.TEMPERATURE},
        )
        raw = resp["message"]["content"]
        try:
            obj = schema.model_validate_json(raw)
            _log({"prompt_file": prompt_file, "attempt": attempt, "ok": True,
                  "seconds": round(time.time() - t0, 2), "raw": raw})
            return obj
        except ValidationError as e:
            last_err = str(e)[:800]
            _log({"prompt_file": prompt_file, "attempt": attempt, "ok": False, "error": last_err, "raw": raw})
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"That JSON failed validation:\n{last_err}\nReturn corrected JSON only."},
            ]
    raise RuntimeError(f"{prompt_file}: failed after {retries} attempts. Last error: {last_err}")

def call_text(prompt: str, temperature: float = 0.2) -> str:
    resp = ollama.chat(model=config.MODEL,
                       messages=[{"role": "user", "content": prompt}],
                       options={"temperature": temperature})
    return resp["message"]["content"]