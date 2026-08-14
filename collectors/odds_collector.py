import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).parent.parent))
import config
from collectors.cache import cache_age_str, load_cache

# Requests restantes devueltas por la API en los headers de cada respuesta
_requests_remaining: int | None = None
_requests_used: int | None = None


def get_odds(
    sport: str,
    regions: str = config.ODDS_API_REGIONS,
    markets: str = config.ODDS_API_MARKETS,
) -> list[dict]:
    """Llama a GET /v4/sports/{sport}/odds y devuelve la lista de partidos con cuotas."""
    if not config.ODDS_API_KEY:
        raise ValueError("ODDS_API_KEY no está configurada en el .env")

    url = f"{config.ODDS_API_BASE_URL}/sports/{sport}/odds"
    params = {
        "apiKey": config.ODDS_API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": config.ODDS_API_ODDS_FORMAT,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
    except requests.exceptions.ConnectionError:
        raise ConnectionError(f"No se pudo conectar a The Odds API para {sport}")
    except requests.exceptions.Timeout:
        raise TimeoutError(f"Timeout al consultar {sport} — intenta de nuevo")

    _update_quota_headers(response)

    if response.status_code == 401:
        raise PermissionError("ODDS_API_KEY inválida o sin permisos")
    if response.status_code == 404:
        raise ValueError(f"Deporte no encontrado: '{sport}'")
    if response.status_code == 422:
        raise ValueError(f"Deporte no reconocido por la API: '{sport}'")
    if response.status_code == 429:
        raise RuntimeError(
            "Límite de requests alcanzado. "
            f"Usados: {_requests_used} | Restantes: {_requests_remaining}"
        )
    if response.status_code != 200:
        raise RuntimeError(
            f"Error inesperado de la API [{response.status_code}]: {response.text}"
        )

    return response.json()


def _update_quota_headers(response: requests.Response) -> None:
    global _requests_remaining, _requests_used
    try:
        _requests_remaining = int(response.headers.get("x-requests-remaining", -1))
        _requests_used = int(response.headers.get("x-requests-used", -1))
    except (TypeError, ValueError):
        pass


def collect_all_leagues(
    regions: str = config.ODDS_API_REGIONS,
    markets: str = config.ODDS_API_MARKETS,
) -> dict:
    """
    Itera sobre TRACKED_LEAGUES, agrega todas las cuotas y guarda el JSON raw.
    Devuelve un dict {sport: [partidos]} con todo lo recopilado.
    """
    cache_file = "odds_raw.json"
    max_hours = None if config.DEV_MODE else config.CACHE_HOURS_ODDS
    cached = load_cache(cache_file, max_hours)

    if cached is not None:
        age = cache_age_str(cache_file)
        print(f"  [CACHE] Usando datos guardados (antigüedad: {age})")
        return cached

    all_odds: dict[str, list] = {}
    errors: list[str] = []

    for sport in config.TRACKED_LEAGUES:
        try:
            matches = get_odds(sport, regions, markets)
            all_odds[sport] = matches
            print(f"  {sport}: {len(matches)} partidos")
        except RuntimeError as e:
            # Solo el 429 lanza RuntimeError — no tiene sentido seguir
            print(f"  [CUOTA AGOTADA] {e}")
            errors.append(sport)
            break
        except (ConnectionError, TimeoutError, ValueError, PermissionError) as e:
            print(f"  [ERROR] {sport}: {e}")
            errors.append(sport)
            # Continúa con la siguiente liga

    _save_raw(all_odds)
    _print_summary(all_odds, errors)

    return all_odds


def _save_raw(all_odds: dict) -> None:
    """Guarda all_odds en data/odds_raw.json con timestamp UTC."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.DATA_DIR / "odds_raw.json"

    payload = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "requests_used": _requests_used,
        "requests_remaining": _requests_remaining,
        "data": all_odds,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n  Guardado en: {output_path}")


def _print_summary(all_odds: dict, errors: list[str]) -> None:
    total_matches = sum(len(v) for v in all_odds.values())

    print("\n--- Resumen ---")
    print(f"  Ligas consultadas : {len(config.TRACKED_LEAGUES)}")
    print(f"  Ligas con datos   : {len(all_odds)}")
    print(f"  Errores           : {len(errors)}")
    print(f"  Partidos totales  : {total_matches}")
    if _requests_remaining is not None:
        print(f"  Requests restantes: {_requests_remaining}")


if __name__ == "__main__":
    print("Recopilando cuotas de The Odds API...\n")
    collect_all_leagues()
