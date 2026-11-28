"""Tools available to ForgeAI."""

from collections.abc import Callable
from dataclasses import dataclass

from forgeai.tools.calculator import calculator as _calculator
from forgeai.tools.csv_lookup import csv_lookup as _csv_lookup
from forgeai.tools.date_tool import current_date as _current_date
from forgeai.tools.weather import get_weather as _get_weather


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, object]
    fn: Callable[..., dict[str, object]]


TOOLS: dict[str, ToolSpec] = {
    "calculator": ToolSpec(
        name="calculator",
        description=(
            "Use to safely evaluate a user-provided arithmetic expression when "
            "they need a numerical calculation."
        ),
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The arithmetic expression to evaluate.",
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
        fn=_calculator,
    ),
    "current_date": ToolSpec(
        name="current_date",
        description=(
            "Use when the user asks for today's date, the current weekday, or "
            "the current timestamp."
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        fn=_current_date,
    ),
    "get_weather": ToolSpec(
        name="get_weather",
        description=(
            "Current weather for a latitude/longitude; only use when the user "
            "explicitly asks about weather conditions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "Latitude in decimal degrees.",
                },
                "longitude": {
                    "type": "number",
                    "description": "Longitude in decimal degrees.",
                },
            },
            "required": ["latitude", "longitude"],
            "additionalProperties": False,
        },
        fn=_get_weather,
    ),
    "csv_lookup": ToolSpec(
        name="csv_lookup",
        description=(
            "Use to find rows containing a value in a named column of a "
            "configured project CSV file."
        ),
        parameters={
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": "A CSV file path relative to the configured documents directory.",
                },
                "column": {
                    "type": "string",
                    "description": "The CSV column to search.",
                },
                "value": {
                    "type": "string",
                    "description": "The text to find, ignoring letter case.",
                },
                "max_rows": {
                    "type": "integer",
                    "description": "Maximum number of matching rows to return.",
                    "default": 5,
                    "minimum": 0,
                },
            },
            "required": ["file", "column", "value"],
            "additionalProperties": False,
        },
        fn=_csv_lookup,
    ),
}
