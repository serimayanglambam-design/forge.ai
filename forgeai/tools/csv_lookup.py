"""CSV lookup tool for ForgeAI."""

import csv
from pathlib import Path

DOCS_DIR: Path = (Path(__file__).resolve().parents[2] / "data").resolve()


def set_docs_dir(path: str | Path) -> None:
    """Set the root directory from which CSV files can be looked up."""
    global DOCS_DIR
    DOCS_DIR = Path(path).resolve()


def csv_lookup(
    file: str,
    column: str,
    value: str,
    max_rows: int = 5,
) -> dict[str, object]:
    """Return CSV rows whose named column contains the requested text."""
    relative_path = Path(file)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return {"error": "file must be a relative path that stays within DOCS_DIR"}

    docs_dir = DOCS_DIR.resolve()
    file_path = (docs_dir / relative_path).resolve()
    try:
        file_path.relative_to(docs_dir)
    except ValueError:
        return {"error": "file must stay within DOCS_DIR"}

    if max_rows < 0:
        return {"error": "max_rows must be zero or greater"}

    try:
        with file_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            columns = reader.fieldnames or []
            if column not in columns:
                valid_columns = ", ".join(columns) if columns else "(none)"
                return {
                    "error": (
                        f"Column '{column}' does not exist. "
                        f"Valid columns: {valid_columns}"
                    )
                }
            if max_rows == 0:
                return {"rows": []}

            matches: list[dict[str, str | None]] = []
            needle = value.casefold()
            for row in reader:
                cell = row.get(column) or ""
                if needle in cell.casefold():
                    matches.append({name: row.get(name) for name in columns})
                    if len(matches) >= max_rows:
                        break
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        return {"error": f"Could not read CSV file: {error}"}

    return {"rows": matches}
