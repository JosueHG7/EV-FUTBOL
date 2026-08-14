import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).parent.parent))
import config
from collectors.cache import cache_age_str, load_cache

_requests_remaining: int | None = None
_requests_limit: int | None = None

_SESSION = requests.Session()
_SESSION.headers.update({"x-apisports-key": config.API_FOOTBALL_KEY})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_fixtures(league_id: int, season: int = config.API_FOOTBALL_SEASON) -> list[dict]:
    """Todos los partidos de una liga y temporada."""
    return _get("fixtures", league=league_id, season=season)


def get_team_statistics(
    league_id: int, season: int, team_id: int
) -> dict:
    """Estadísticas agregadas de un equipo en una liga/temporada."""
    results = _get("teams/statistics", league=league_id, season=season, team=team_id)
    return results[0] if results else {}


def get_standings(league_id: int, season: int = config.API_FOOTBALL_SEASON) -> list[dict]:
    """Tabla de posiciones de una liga/temporada."""
    return _get("standings", league=league_id, season=season)


# ---------------------------------------------------------------------------
# Batch collection
# ---------------------------------------------------------------------------

def collect_all_fixtures(season: int = config.API_FOOTBALL_SEASON) -> dict:
    """
    Trae fixtures de todas las ligas en LEAGUE_IDS y guarda en fixtures_{season}.json.
    Devuelve un dict {league_key: [fixtures]}.
    """
    cache_file = f"fixtures_{season}.json"
    max_hours = None if config.DEV_MODE else config.CACHE_HOURS_FIXTURES
    cached = load_cache(cache_file, max_hours)

    if cached is not None:
        age = cache_age_str(cache_file)
        print(f"  [CACHE] {cache_file} (antigüedad: {age})")
        return cached

    if not config.API_FOOTBALL_KEY:
        raise ValueError("API_FOOTBALL_KEY no está configurada en el .env")

    all_fixtures: dict[str, list] = {}
    errors: list[str] = []

    for league_key, league_id in config.LEAGUE_IDS.items():
        _check_quota()
        try:
            fixtures = get_fixtures(league_id, season)
            all_fixtures[league_key] = fixtures
            print(f"  {league_key} (id={league_id}): {len(fixtures)} partidos")
        except RuntimeError as e:
            print(f"  [CUOTA AGOTADA] {e}")
            errors.append(league_key)
            break
        except (ConnectionError, TimeoutError, ValueError, PermissionError) as e:
            print(f"  [ERROR] {league_key}: {e}")
            errors.append(league_key)

    _save_raw(all_fixtures, season)
    _print_summary(all_fixtures, errors)

    return all_fixtures


def collect_all_seasons(seasons: list[int] = [2022, 2023, 2024]) -> None:
    """
    Itera sobre las temporadas dadas, llama collect_all_fixtures para cada una
    y guarda fixtures_{season}.json. Se detiene si requests_remaining < 20.
    """
    if not config.API_FOOTBALL_KEY:
        raise ValueError("API_FOOTBALL_KEY no está configurada en el .env")

    requests_at_start = _requests_remaining
    completed: list[int] = []

    for season in seasons:
        if _requests_remaining is not None and 0 <= _requests_remaining < 20:
            print(f"\n  [STOP] Solo quedan {_requests_remaining} requests — abortando temporadas restantes")
            break

        print(f"\n--- Temporada {season} ---")
        collect_all_fixtures(season)
        completed.append(season)

    print("\n=== Resumen multi-temporada ===")
    print(f"  Temporadas completadas : {completed}")
    skipped = [s for s in seasons if s not in completed]
    if skipped:
        print(f"  Temporadas omitidas    : {skipped}")
    if _requests_remaining is not None and _requests_remaining >= 0:
        used = (requests_at_start - _requests_remaining) if requests_at_start is not None else "?"
        print(f"  Requests usados        : {used}")
        print(f"  Requests restantes     : {_requests_remaining} / {_requests_limit}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get(endpoint: str, **params) -> list[dict]:
    """Llamada genérica a la API. Devuelve el array 'response' del payload."""
    if not config.API_FOOTBALL_KEY:
        raise ValueError("API_FOOTBALL_KEY no está configurada en el .env")

    url = f"{config.API_FOOTBALL_BASE_URL}/{endpoint}"

    try:
        resp = _SESSION.get(url, params=params, timeout=15)
    except requests.exceptions.ConnectionError:
        raise ConnectionError(f"No se pudo conectar a api-football.com ({endpoint})")
    except requests.exceptions.Timeout:
        raise TimeoutError(f"Timeout en {endpoint} — intenta de nuevo")

    _update_quota_headers(resp)

    if resp.status_code == 401:
        raise PermissionError("API_FOOTBALL_KEY inválida o sin permisos")
    if resp.status_code == 429:
        raise RuntimeError(
            f"Límite diario alcanzado. Limit: {_requests_limit} | Restantes: {_requests_remaining}"
        )
    if resp.status_code != 200:
        raise RuntimeError(f"Error de la API [{resp.status_code}]: {resp.text}")

    body = resp.json()

    # La API devuelve errores lógicos con status 200 dentro del payload
    if body.get("errors"):
        raise ValueError(f"Error en la respuesta ({endpoint}): {body['errors']}")

    return body.get("response", [])


def _update_quota_headers(resp: requests.Response) -> None:
    global _requests_remaining, _requests_limit
    try:
        _requests_remaining = int(resp.headers.get("x-ratelimit-requests-remaining", -1))
        _requests_limit = int(resp.headers.get("x-ratelimit-requests-limit", -1))
    except (TypeError, ValueError):
        pass


def _check_quota() -> None:
    """Avisa si quedan pocas requests antes de cada llamada."""
    if _requests_remaining is not None and 0 < _requests_remaining <= 10:
        print(f"  [AVISO] Solo quedan {_requests_remaining} requests hoy")


def _save_raw(all_fixtures: dict, season: int) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.DATA_DIR / f"fixtures_{season}.json"

    payload = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "season": season,
        "requests_limit": _requests_limit,
        "requests_remaining": _requests_remaining,
        "data": all_fixtures,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n  Guardado en: {output_path}")


def _print_summary(all_fixtures: dict, errors: list[str]) -> None:
    total = sum(len(v) for v in all_fixtures.values())

    print("\n--- Resumen ---")
    print(f"  Ligas consultadas : {len(config.LEAGUE_IDS)}")
    print(f"  Ligas con datos   : {len(all_fixtures)}")
    print(f"  Errores           : {len(errors)}")
    print(f"  Partidos totales  : {total}")
    if _requests_remaining is not None and _requests_remaining >= 0:
        print(f"  Requests restantes: {_requests_remaining} / {_requests_limit}")


if __name__ == "__main__":
    print("Recopilando fixtures de API-Football...\n")
    collect_all_seasons()
