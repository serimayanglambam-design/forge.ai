"""Generic retrieval-augmented chat runner."""

import json
import re
from pathlib import Path
from typing import Any

import ollama
import typer
from pydantic import BaseModel

from forgeai import config
from forgeai.loader import Chunk, chunk, load_folder
from forgeai.retrieval import Retriever
from forgeai.schemas import AgentDefinition
from forgeai.tools import TOOLS, ToolSpec
from forgeai.tools.csv_lookup import set_docs_dir


def _extract_cited_sources(answer: str, retrieved_doc_ids: list[str]) -> list[str]:
    retrieved_names = {
        doc_id.casefold(): doc_id for doc_id in retrieved_doc_ids
    }
    sources: list[str] = []
    seen: set[str] = set()
    for bracket_content in re.findall(r"\[([^\[\]]+)\]", answer):
        content = re.sub(
            r"^\s*(?:doc_id|source)\s*:\s*",
            "",
            bracket_content,
            flags=re.IGNORECASE,
        )
        for citation in re.split(r"[,;]", content):
            citation = re.sub(
                r"^\s*(?:doc_id|source)\s*:\s*",
                "",
                citation,
                flags=re.IGNORECASE,
            ).strip()
            doc_id = retrieved_names.get(citation.casefold())
            if doc_id is not None and doc_id not in seen:
                seen.add(doc_id)
                sources.append(doc_id)
    return sources


class AgentResponse(BaseModel):
    answer: str
    sources: list[str]
    retrieved_chunk_ids: list[str]
    tool_calls: list[dict[str, Any]]
    refused: bool


class Agent:
    """Answer user questions from an agent definition and document collection."""

    def __init__(
        self,
        definition: AgentDefinition,
        retriever: Retriever,
        tools: list[ToolSpec],
    ) -> None:
        self.definition = definition
        self.retriever = retriever
        self.tools = tools

    def _system_prompt(self) -> str:
        contract = self.definition.output_contract
        instructions = [
            "Follow the output contract exactly.",
        ]
        if contract.require_citations:
            instructions.append(
                "Cite sources exactly like [filename.ext], for example "
                "[policy.pdf]. Do not write 'doc_id:' inside the brackets."
            )
        instructions.append(
            "If the supplied documents do not contain the answer, reply with "
            f"exactly: {contract.refusal_phrase}"
        )
        instructions.append(
            f"Keep the answer under {contract.max_words} words."
        )
        return f"{self.definition.system_prompt.rstrip()}\n\n" + "\n".join(instructions)

    @staticmethod
    def _member(value: object, key: str, default: object = None) -> object:
        if isinstance(value, dict):
            return value.get(key, default)
        return getattr(value, key, default)

    @staticmethod
    def _serializable(value: object) -> object:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return {str(key): Agent._serializable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [Agent._serializable(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def _tool_schemas(self) -> list[dict[str, object]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.tools
        ]

    def _run_tool(self, name: str, arguments: object) -> object:
        tool = next((candidate for candidate in self.tools if candidate.name == name), None)
        if tool is None:
            return {"error": f"Tool '{name}' is not attached to this agent"}
        try:
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be a JSON object")
            return self._serializable(tool.fn(**arguments))
        except Exception as error:
            return {"error": str(error)}

    @staticmethod
    def _message_text(message: object) -> str:
        content = Agent._member(message, "content", "")
        return content if isinstance(content, str) else str(content)

    @staticmethod
    def _message_tool_calls(message: object) -> list[object]:
        tool_calls = Agent._member(message, "tool_calls", [])
        return tool_calls if isinstance(tool_calls, list) else []

    def answer(
        self, question: str, history: list[dict[str, Any]] | None = None
    ) -> AgentResponse:
        results: list[tuple[Chunk, float]] = []
        recorded_tool_calls: list[dict[str, Any]] = []
        try:
            results = self.retriever.search(question)
            retrieved_chunk_ids = [chunk_item.chunk_id for chunk_item, _ in results]
            retrieved_doc_ids = [chunk_item.doc_id for chunk_item, _ in results]
            sources_block = "\n\n".join(
                f"[source: {chunk_item.doc_id}]\n{chunk_item.text}"
                for chunk_item, _ in results
            )
            user_content = (
                f"{sources_block}\n\nQuestion: {question}"
                if sources_block
                else question
            )
            messages: list[dict[str, object]] = [
                {"role": "system", "content": self._system_prompt()},
            ]
            messages.extend(dict(item) for item in history or [])
            messages.append({"role": "user", "content": user_content})

            tool_schemas = self._tool_schemas()
            tool_count = 0
            while True:
                request: dict[str, object] = {
                    "model": config.MODEL,
                    "messages": messages,
                }
                if tool_schemas and tool_count < 3:
                    request["tools"] = tool_schemas
                response = ollama.chat(**request)
                assistant_message = self._member(response, "message", response)
                tool_requests = self._message_tool_calls(assistant_message)

                if not tool_requests or not tool_schemas or tool_count >= 3:
                    answer = self._message_text(assistant_message)
                    break

                allowed_requests = tool_requests[: 3 - tool_count]
                messages.append(
                    {
                        "role": "assistant",
                        "content": self._message_text(assistant_message),
                        "tool_calls": self._serializable(allowed_requests),
                    }
                )
                for tool_request in allowed_requests:
                    function = self._member(tool_request, "function", {})
                    name_value = self._member(function, "name", "")
                    name = name_value if isinstance(name_value, str) else str(name_value)
                    arguments = self._member(function, "arguments", {})
                    result = self._run_tool(name, arguments)
                    recorded_tool_calls.append(
                        {
                            "name": name,
                            "arguments": self._serializable(arguments),
                            "result": self._serializable(result),
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "name": name,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
                    tool_count += 1

            ordered_sources = _extract_cited_sources(answer, retrieved_doc_ids)

            refusal_phrase = self.definition.output_contract.refusal_phrase
            return AgentResponse(
                answer=answer,
                sources=ordered_sources,
                retrieved_chunk_ids=retrieved_chunk_ids,
                tool_calls=recorded_tool_calls,
                refused=refusal_phrase.casefold() in answer.casefold(),
            )
        except Exception as error:
            return AgentResponse(
                answer=f"Error: {error}",
                sources=[],
                retrieved_chunk_ids=[chunk_item.chunk_id for chunk_item, _ in results],
                tool_calls=recorded_tool_calls,
                refused=False,
            )


def load_agent(def_path: str, docs_dir: str) -> Agent:
    """Load an agent definition, its documents, retriever, and configured tools."""
    definition_path = Path(def_path)
    with definition_path.open("r", encoding="utf-8") as definition_file:
        definition = AgentDefinition.model_validate_json(definition_file.read())

    selected_tools: list[ToolSpec] = []
    for tool_name in definition.tools:
        if tool_name not in TOOLS:
            raise ValueError(
                f"Unknown tool '{tool_name}' in agent definition; "
                f"available tools: {', '.join(sorted(TOOLS))}"
            )
        selected_tools.append(TOOLS[tool_name])

    set_docs_dir(docs_dir)
    documents = load_folder(docs_dir)
    chunks = [
        chunk_item
        for document in documents
        for chunk_item in chunk(
            document,
            size=definition.retrieval.chunk_size,
            overlap=definition.retrieval.chunk_overlap,
        )
    ]
    return Agent(
        definition=definition,
        retriever=Retriever(chunks, definition.retrieval),
        tools=selected_tools,
    )


app = typer.Typer(help="Chat with a document-grounded agent.")


@app.command()
def chat(
    agent: str = typer.Option(..., "--agent", help="Path to the agent JSON definition."),
    docs: str = typer.Option(..., "--docs", help="Directory containing agent documents."),
) -> None:
    """Start an interactive chat loop."""
    loaded_agent = load_agent(agent, docs)
    history: list[dict[str, Any]] = []
    while True:
        question = typer.prompt("You")
        if question.strip().casefold() == "exit":
            break
        response = loaded_agent.answer(question, history)
        typer.echo(response.answer)
        if response.sources:
            typer.echo(f"Sources: {', '.join(response.sources)}")
        if response.refused:
            typer.echo("REFUSED")
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": response.answer})


@app.command()
def ask(
    agent: str = typer.Option(..., "--agent", help="Path to the agent JSON definition."),
    docs: str = typer.Option(..., "--docs", help="Directory containing agent documents."),
    question: str = typer.Option(..., "--question", help="Question to answer."),
) -> None:
    """Answer a single question and print the response as JSON."""
    response = load_agent(agent, docs).answer(question)
    typer.echo(response.model_dump_json())


if __name__ == "__main__":
    app()
