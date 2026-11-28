"""Date tool for ForgeAI."""

from datetime import datetime


def current_date() -> dict[str, object]:
    """Return the current local date, weekday, and ISO timestamp."""
    now = datetime.now().astimezone()
    return {
        "date": now.strftime("%Y-%m-%d"),
        "weekday": now.strftime("%A"),
        "iso_datetime": now.isoformat(timespec="seconds"),
    }
