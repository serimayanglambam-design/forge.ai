import json
import importlib
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse

import pytest

from forgeai.tools import TOOLS, ToolSpec
from forgeai.tools.calculator import calculator
from forgeai.tools.csv_lookup import csv_lookup

csv_lookup_module = importlib.import_module("forgeai.tools.csv_lookup")
date_tool = importlib.import_module("forgeai.tools.date_tool")
weather = importlib.import_module("forgeai.tools.weather")


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3 * (4 - 1)", 11),
        ("-7 // 2", -4),
        ("10 % 3", 1),
        ("2 ** 5", 32),
        ("8 / 4", 2),
    ],
)
def test_calculator_evaluates_allowed_arithmetic(
    expression: str, expected: int | float
) -> None:
    assert calculator(expression) == {"result": expected}


@pytest.mark.parametrize(
    "expression",
    ["__import__('os')", "2**9999999", "10**13", "1 + True"],
)
def test_calculator_rejects_unsafe_or_out_of_range_expressions(
    expression: str,
) -> None:
    assert "error" in calculator(expression)


def test_csv_lookup_searches_case_insensitively_and_rejects_escapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(csv_lookup_module, "DOCS_DIR", csv_lookup_module.DOCS_DIR)
    csv_lookup_module.set_docs_dir(tmp_path)
    (tmp_path / "places.csv").write_text(
        "name,city\nAlice,New York\nBob,Seattle\nCal,Newark\n",
        encoding="utf-8",
    )

    result = csv_lookup("places.csv", "city", "NEW", max_rows=5)

    assert result == {
        "rows": [
            {"name": "Alice", "city": "New York"},
            {"name": "Cal", "city": "Newark"},
        ]
    }
    assert "error" in csv_lookup("../places.csv", "city", "new")
    assert "error" in csv_lookup(str(tmp_path / "places.csv"), "city", "new")
    missing_column = csv_lookup("places.csv", "country", "US")
    assert "name, city" in missing_column["error"]


def test_current_date_returns_expected_fields() -> None:
    result = date_tool.current_date()

    assert set(result) == {"date", "weekday", "iso_datetime"}
    today = date.fromisoformat(result["date"])
    timestamp = datetime.fromisoformat(result["iso_datetime"])
    assert result["weekday"] == today.strftime("%A")
    assert timestamp.date() == today


def test_weather_uses_mocked_open_meteo_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "current": {
            "temperature_2m": 18.5,
            "wind_speed_10m": 7.2,
            "weather_code": 2,
        }
    }

    def fake_urlopen(url: str, timeout: int) -> BytesIO:
        assert timeout == 10
        query = parse_qs(urlparse(url).query)
        assert query["current"] == ["temperature_2m,wind_speed_10m,weather_code"]
        return BytesIO(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(weather, "urlopen", fake_urlopen)

    assert weather.get_weather(45.0, -122.0) == payload


def test_weather_returns_errors_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_urlopen(url: str, timeout: int) -> BytesIO:
        raise URLError("service unavailable")

    monkeypatch.setattr(weather, "urlopen", fail_urlopen)

    assert "error" in weather.get_weather(45.0, -122.0)


def test_tool_registry_exposes_specs() -> None:
    assert set(TOOLS) == {"calculator", "current_date", "get_weather", "csv_lookup"}
    assert all(isinstance(spec, ToolSpec) and callable(spec.fn) for spec in TOOLS.values())
    assert all(spec.description.endswith(".") for spec in TOOLS.values())
