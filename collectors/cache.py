import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config


def load_cache(filename: str, max_hours: float | None) -> dict | None:
    """
    Lee data/{filename} y devuelve el campo 'data' si el cache es válido.

    max_hours=None significa sin límite de edad (útil en DEV_MODE).
    Devuelve None si el archivo no existe, está corrupto o ha expirado.
    """
    path = config.DATA_DIR / filename
    if not path.exists():
        return None

    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    if "collected_at" not in payload or "data" not in payload:
        return None

    if max_hours is not None:
        collected_at = datetime.fromisoformat(payload["collected_at"])
        age_hours = (datetime.now(timezone.utc) - collected_at).total_seconds() / 3600
        if age_hours > max_hours:
            return None

    return payload["data"]


def save_cache(filename: str, data: dict) -> None:
    """Guarda data en data/{filename} con timestamp UTC."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = config.DATA_DIR / filename

    payload = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def cache_age_str(filename: str) -> str:
    """Devuelve la edad del cache como string legible, o 'sin cache'."""
    path = config.DATA_DIR / filename
    if not path.exists():
        return "sin cache"

    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        collected_at = datetime.fromisoformat(payload["collected_at"])
        minutes = (datetime.now(timezone.utc) - collected_at).total_seconds() / 60
        if minutes < 60:
            return f"{int(minutes)}m"
        return f"{minutes / 60:.1f}h"
    except (KeyError, ValueError, OSError):
        return "cache inválido"
