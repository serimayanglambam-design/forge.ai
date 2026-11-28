"""Weather tool for ForgeAI."""

import json
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen


def get_weather(latitude: float, longitude: float) -> dict[str, object]:
    """Return current weather for the requested latitude and longitude."""
    query = urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,wind_speed_10m,weather_code",
        }
    )
    url = f"https://api.open-meteo.com/v1/forecast?{query}"
    try:
        with urlopen(url, timeout=10) as response:
            data = json.loads(response.read())
    except (URLError, TimeoutError, OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        return {"error": f"Weather request failed: {error}"}

    if not isinstance(data, dict):
        return {"error": "Weather service returned an unexpected response"}
    return data
