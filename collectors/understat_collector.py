"""
collectors/understat_collector.py

Fetches expected goals (xG) data from understat.com for the 5 main leagues.
No API key required — uses the same AJAX endpoint the browser calls.

API
---
GET https://understat.com/getLeagueData/<slug>/<start_year>
Required headers:
  X-Requested-With: XMLHttpRequest
  Referer:          https://understat.com/league/<slug>/<start_year>

Response JSON: {"dates": [...], "teams": {...}, "players": [...]}
The "dates" array contains per-match records:
  {"id", "isResult", "h": {"title": ...}, "a": {"title": ...},
   "goals": {"h": "2", "a": "1"}, "xG": {"h": "1.5", "a": "0.8"},
   "datetime": "YYYY-MM-DD HH:MM:SS"}

Cache: data/understat/<slug>_<season>.json, TTL 48 hours.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).parent.parent))
import config

_CACHE_HOURS = 48.0

# API-Football league_id → Understat slug
LEAGUE_SLUGS: dict[int, str] = {
    39:  "EPL",
    140: "La_liga",
    78:  "Bundesliga",
    135: "Serie_A",
    61:  "Ligue_1",
}

# Understat team name → DB team name (home_team_name in matches table).
# Only entries that differ between the two sources are needed.
_TEAM_NAME_MAP: dict[str, str] = {
    # Premier League
    "Wolverhampton Wanderers": "Wolves",
    "Sheffield United":        "Sheffield Utd",
    "Newcastle United":        "Newcastle",
    "Leeds United":            "Leeds",
    "Leicester City":          "Leicester",
    "Tottenham Hotspur":       "Tottenham",
    "West Ham United":         "West Ham",
    "Brighton & Hove Albion":  "Brighton",
    "Luton Town":              "Luton",
    "Ipswich Town":            "Ipswich",
    # La Liga
    "Deportivo Alavés":        "Alaves",
    "Alavés":                  "Alaves",
    "Almería":                 "Almeria",
    "Cádiz":                   "Cadiz",
    "Granada":                 "Granada CF",
    "Leganés":                 "Leganes",
    # Bundesliga
    "Heidenheim":              "1. FC Heidenheim",
    "FC Heidenheim":           "1. FC Heidenheim",
    "FC Köln":                 "1. FC Köln",
    "Hoffenheim":              "1899 Hoffenheim",
    "Borussia M'gladbach":     "Borussia Mönchengladbach",
    "Augsburg":                "FC Augsburg",
    "Schalke 04":              "FC Schalke 04",
    "St. Pauli":               "FC St. Pauli",
    "Mainz":                   "FSV Mainz 05",
    "Fortuna Düsseldorf":      "Fortuna Dusseldorf",
    "Freiburg":                "SC Freiburg",
    "Darmstadt":               "SV Darmstadt 98",
    "Elversberg":              "SV Elversberg",
    "Stuttgart":               "VfB Stuttgart",
    "Bochum":                  "VfL Bochum",
    "Wolfsburg":               "VfL Wolfsburg",
    # Serie A
    "Roma":                    "AS Roma",
    "Internazionale":          "Inter",
    "AC Milan":                "Milan",
    # Ligue 1
    "Paris Saint-Germain":     "Paris Saint Germain",
    "Brest":                   "Stade Brestois 29",
    "Clermont":                "Clermont Foot",
    "Troyes":                  "Estac Troyes",
    "Saint-Étienne":           "Saint Etienne",
    "Saint-Etienne":           "Saint Etienne",
}

_CACHE_DIR = config.DATA_DIR / "understat"

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":           "application/json, text/javascript, */*; q=0.01",
    "Accept-Language":  "en-US,en;q=0.9",
    "X-Requested-With": "XMLHttpRequest",
})


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(slug: str, season: int) -> Path:
    return _CACHE_DIR / f"{slug}_{season}.json"


def _load_cache(slug: str, season: int) -> list[dict] | None:
    path = _cache_path(slug, season)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        collected_at = datetime.fromisoformat(payload["collected_at"])
        age_hours = (datetime.now(timezone.utc) - collected_at).total_seconds() / 3600
        if age_hours > _CACHE_HOURS:
            return None
        return payload["data"]
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def _save_cache(slug: str, season: int, data: list[dict]) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }
    _cache_path(slug, season).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

def _fetch_raw(slug: str, season: int) -> list[dict]:
    """
    Fetch match list with xG from understat.com JSON API.

    Understat migrated from HTML-embedded JSON.parse() to a dedicated
    AJAX endpoint:  GET /getLeagueData/<slug>/<season>
    Response: {"dates": [...], "teams": {...}, "players": [...]}
    The "dates" array is the old datesData — same per-match structure.
    """
    url = f"https://understat.com/getLeagueData/{slug}/{season}"
    _SESSION.headers["Referer"] = f"https://understat.com/league/{slug}/{season}"
    try:
        resp = _SESSION.get(url, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise ConnectionError(
            f"understat.com no disponible ({slug}/{season}): {exc}"
        )

    try:
        payload = resp.json()
    except ValueError as exc:
        raise ValueError(
            f"Respuesta no es JSON válido ({slug}/{season}): {exc}"
        )

    if "dates" not in payload:
        raise ValueError(
            f"Campo 'dates' no encontrado en respuesta de {slug}/{season}. "
            f"Claves: {list(payload.keys())}"
        )

    return payload["dates"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_league_xg(league_id: int, season: int) -> list[dict]:
    """
    Fetch xG data for one league/season, using cache when available.

    Returns
    -------
    list of raw Understat match dicts (datesData entries)
    """
    if league_id not in LEAGUE_SLUGS:
        raise ValueError(
            f"league_id={league_id} no tiene slug Understat. "
            f"Disponibles: {list(LEAGUE_SLUGS.keys())}"
        )

    slug   = LEAGUE_SLUGS[league_id]
    cached = _load_cache(slug, season)
    if cached is not None:
        return cached

    data = _fetch_raw(slug, season)
    _save_cache(slug, season, data)
    return data


def _map_team(name: str) -> str:
    return _TEAM_NAME_MAP.get(name, name)


def to_dataframe(records: list[dict]) -> pd.DataFrame:
    """
    Convert raw Understat datesData to a clean DataFrame.

    Only completed matches (isResult=True with numeric xG values) are included.

    Returns
    -------
    DataFrame with columns:
        match_date  datetime (UTC-naive, as provided by Understat)
        home_team   str — mapped to DB team names via _TEAM_NAME_MAP
        away_team   str
        xg_h        float — home xG
        xg_a        float — away xG
        goals_h     int
        goals_a     int
    """
    rows = []
    for r in records:
        if not r.get("isResult"):
            continue
        try:
            xg_h = float(r["xG"]["h"])
            xg_a = float(r["xG"]["a"])
        except (KeyError, TypeError, ValueError):
            continue
        try:
            rows.append({
                "match_date": pd.to_datetime(r["datetime"]),
                "home_team":  _map_team(r["h"]["title"]),
                "away_team":  _map_team(r["a"]["title"]),
                "xg_h":       xg_h,
                "xg_a":       xg_a,
                "goals_h":    int(r["goals"]["h"]),
                "goals_a":    int(r["goals"]["a"]),
            })
        except (KeyError, TypeError, ValueError):
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("match_date").reset_index(drop=True)
    return df


def load_all_xg(
    league_ids: list[int] | None = None,
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """
    Load xG data for all leagues × seasons and return a single DataFrame.

    Parameters
    ----------
    league_ids : list of API-Football league IDs.  Defaults to all 5 main
                 leagues: EPL (39), La Liga (140), Bundesliga (78),
                 Serie A (135), Ligue 1 (61).
    seasons    : list of start-year integers.  Defaults to [2022, 2023, 2024].

    Returns
    -------
    Combined DataFrame with match_date, home_team, away_team, xg_h, xg_a,
    goals_h, goals_a, league_id, season columns — sorted chronologically.
    Empty DataFrame on total failure.
    """
    if league_ids is None:
        league_ids = list(LEAGUE_SLUGS.keys())
    if seasons is None:
        seasons = [2022, 2023, 2024, 2025]

    frames: list[pd.DataFrame] = []
    for league_id in league_ids:
        slug = LEAGUE_SLUGS[league_id]
        for season in seasons:
            print(f"  Understat {slug} {season}...", end=" ", flush=True)
            try:
                records = get_league_xg(league_id, season)
                df = to_dataframe(records)
                if df.empty:
                    print("0 partidos")
                    continue
                df["league_id"] = league_id
                df["season"]    = season
                frames.append(df)
                print(f"{len(df)} partidos")
            except (ConnectionError, ValueError) as exc:
                print(f"ERROR: {exc}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values("match_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Cargando xG de Understat (5 ligas × 3 temporadas)...\n")
    xg = load_all_xg()

    if xg.empty:
        print("No se pudo cargar xG data.")
    else:
        print(f"\nTotal: {len(xg)} partidos con xG")
        print(f"Ligas  : {sorted(xg['league_id'].unique())}")
        print(f"Rango  : {xg['match_date'].min().date()} → {xg['match_date'].max().date()}")
        print(f"\nxG medio: home={xg['xg_h'].mean():.3f}  away={xg['xg_a'].mean():.3f}")
        print(f"\nPrimeros 5 partidos:")
        print(xg[["match_date", "home_team", "away_team", "xg_h", "xg_a",
                   "goals_h", "goals_a"]].head().to_string(index=False))
