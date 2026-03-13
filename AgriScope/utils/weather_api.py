"""
AgriScope - Weather API Module
Fetches real-time weather data for Gujarat districts using Open-Meteo API.
"""

import requests
from typing import Optional

# Coordinates for Gujarat districts
DISTRICT_COORDS = {
    "Ahmedabad":        {"lat": 23.0225, "lon": 72.5714},
    "Vadodara":         {"lat": 22.3072, "lon": 73.1812},
    "Surat":            {"lat": 21.1702, "lon": 72.8311},
    "Rajkot":           {"lat": 22.3039, "lon": 70.8022},
    "Bhavnagar":        {"lat": 21.7645, "lon": 72.1519},
    "Junagadh":         {"lat": 21.5222, "lon": 70.4579},
    "Jamnagar":         {"lat": 22.4707, "lon": 70.0577},
    "Mehsana":          {"lat": 23.5971, "lon": 72.3714},
    "Kutch":            {"lat": 23.7337, "lon": 69.8597},
    "Banaskantha":      {"lat": 24.1779, "lon": 72.4375},
    "Patan":            {"lat": 23.8493, "lon": 72.1266},
    "Sabarkantha":      {"lat": 23.6264, "lon": 73.0122},
    "Aravalli":         {"lat": 23.7000, "lon": 73.0500},
    "Gandhinagar":      {"lat": 23.2156, "lon": 72.6369},
    "Surendranagar":    {"lat": 22.7274, "lon": 71.6389},
    "Morbi":            {"lat": 22.8174, "lon": 70.8376},
    "Devbhumi Dwarka":  {"lat": 22.2428, "lon": 68.9686},
    "Porbandar":        {"lat": 21.6417, "lon": 69.6293},
    "Gir Somnath":      {"lat": 20.9070, "lon": 70.3676},
    "Amreli":           {"lat": 21.6032, "lon": 71.2210},
    "Botad":            {"lat": 22.1736, "lon": 71.6670},
    "Anand":            {"lat": 22.5645, "lon": 72.9289},
    "Kheda":            {"lat": 22.7429, "lon": 72.6887},
    "Panchmahal":       {"lat": 22.7597, "lon": 73.5199},
    "Mahisagar":        {"lat": 23.0900, "lon": 73.4700},
    "Dahod":            {"lat": 22.8340, "lon": 74.2500},
    "Chhota Udaipur":   {"lat": 22.3000, "lon": 74.0100},
    "Narmada":          {"lat": 21.8740, "lon": 73.7100},
    "Bharuch":          {"lat": 21.7051, "lon": 72.9959},
    "Navsari":          {"lat": 20.9467, "lon": 72.9520},
    "Valsad":           {"lat": 20.5992, "lon": 72.9342},
    "Tapi":             {"lat": 21.1200, "lon": 73.4100},
}

BASE_URL = "https://api.open-meteo.com/v1/forecast"


def get_weather(district: str) -> Optional[dict]:
    """
    Fetch current weather for the given Gujarat district using Open-Meteo.
    
    Returns a dict with keys:
        temperature (°C), humidity (%), rainfall (mm), wind_speed (km/h)
    Returns None on error.
    """
    if district not in DISTRICT_COORDS:
        print(f"[WARN] District '{district}' not found. Using Ahmedabad as fallback.")
        district = "Ahmedabad"

    coords = DISTRICT_COORDS[district]
    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
        ],
        "timezone": "Asia/Kolkata",
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        current = data.get("current", {})
        return {
            "temperature": round(current.get("temperature_2m", 28.0), 2),
            "humidity":    round(current.get("relative_humidity_2m", 65.0), 2),
            "rainfall":    round(current.get("precipitation", 0.0), 2),
            "wind_speed":  round(current.get("wind_speed_10m", 10.0), 2),
        }
    except requests.exceptions.Timeout:
        print(f"[ERROR] Weather API timeout for district: {district}")
        return _fallback_weather(district)
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Weather API error for '{district}': {e}")
        return _fallback_weather(district)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return _fallback_weather(district)


def _fallback_weather(district: str) -> dict:
    """Return reasonable default weather values when API is unavailable."""
    return {
        "temperature": 28.5,
        "humidity":    65.0,
        "rainfall":    0.0,
        "wind_speed":  12.0,
    }


def get_all_districts() -> list:
    """Return list of all supported Gujarat districts."""
    return sorted(DISTRICT_COORDS.keys())


if __name__ == "__main__":
    for d in ["Ahmedabad", "Vadodara", "Surat"]:
        result = get_weather(d)
        print(f"{d}: {result}")
