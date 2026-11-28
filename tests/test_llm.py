from pydantic import BaseModel
from forgeai.llm import call_json

class Person(BaseModel):
    name: str
    age: int

def test_call_json():
    p = call_json("smoke.md", {"text": "Priya is 21 years old."}, Person)
    assert p.name == "Priya" and p.age == 21