import json
import sys
from datetime import datetime, timezone, date
from pathlib import Path

from sqlalchemy import select

sys.path.append(str(Path(__file__).parent.parent))
import config
from database.db import get_session, init_db
from database.models import Match, Odds

# ---------------------------------------------------------------------------
# Status mapping
# ---------------------------------------------------------------------------

_STATUS_MAP = {
    "FT": "finished", "AET": "finished", "PEN": "finished",
    "NS": "scheduled", "TBD": "scheduled",
}


def _map_status(short: str) -> str:
    return _STATUS_MAP.get(short, "cancelled")


# ---------------------------------------------------------------------------
# Team name normalisation
# ---------------------------------------------------------------------------

# API-Football uses localised/variant spellings that differ from football-data.co.uk
# and from the names used in The Odds API.  Canonical form is the most common
# English name so all three data sources can resolve to the same string.
_TEAM_NAME_MAP: dict[str, str] = {
    "Bayern München":        "Bayern Munich",
    "Paris Saint-Germain":   "Paris Saint Germain",
    "Atlético Madrid":       "Atletico Madrid",
    "Internazionale":        "Inter",
    "AC Milan":              "Milan",
}


def _normalize(name: str) -> str:
    return _TEAM_NAME_MAP.get(name, name)


# ---------------------------------------------------------------------------
# Fixtures → Match
# ---------------------------------------------------------------------------

def ingest_fixtures(path: Path = config.DATA_DIR / "fixtures_raw.json") -> None:
    print(f"Ingesting fixtures from {path.name}...")

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    inserted = updated = errors = 0

    with get_session() as session:
        for league_key, fixtures in raw["data"].items():
            for fix in fixtures:
                try:
                    fix_id     = fix["fixture"]["id"]
                    status_short = fix["fixture"]["status"]["short"]
                    match_date = datetime.fromisoformat(fix["fixture"]["date"])

                    existing = session.get(Match, fix_id)
                    if existing is None:
                        match = Match(id=fix_id)
                        session.add(match)
                        inserted += 1
                    else:
                        match = existing
                        updated += 1

                    match.league_id      = fix["league"]["id"]
                    match.league_name    = fix["league"]["name"]
                    match.season         = fix["league"]["season"]
                    match.home_team_id   = fix["teams"]["home"]["id"]
                    match.home_team_name = _normalize(fix["teams"]["home"]["name"])
                    match.away_team_id   = fix["teams"]["away"]["id"]
                    match.away_team_name = _normalize(fix["teams"]["away"]["name"])
                    match.match_date     = match_date
                    match.status         = _map_status(status_short)
                    match.home_goals     = fix["goals"]["home"]
                    match.away_goals     = fix["goals"]["away"]

                except (KeyError, TypeError, ValueError) as e:
                    errors += 1
                    fix_id = fix.get("fixture", {}).get("id", "?")
                    print(f"  [ERROR] fixture {fix_id}: {e}")

    print(f"  Fixtures — insertados: {inserted} | actualizados: {updated} | errores: {errors}")


# ---------------------------------------------------------------------------
# Odds → Odds
# ---------------------------------------------------------------------------

def ingest_odds(path: Path = config.DATA_DIR / "odds_raw.json") -> None:
    print(f"Ingesting odds from {path.name}...")

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    collected_at = datetime.fromisoformat(raw["collected_at"])
    inserted = updated = skipped = errors = 0

    with get_session() as session:
        for league_key, matches in raw["data"].items():
            for match_data in matches:
                try:
                    db_match = _find_match(session, match_data)
                    if db_match is None:
                        skipped += 1
                        continue

                    for bookmaker in match_data.get("bookmakers", []):
                        h2h = next(
                            (m for m in bookmaker["markets"] if m["key"] == "h2h"),
                            None,
                        )
                        if h2h is None:
                            continue

                        home_win, draw, away_win = _parse_h2h(
                            h2h["outcomes"],
                            match_data["home_team"],
                            match_data["away_team"],
                        )
                        if None in (home_win, draw, away_win):
                            continue

                        bk_key = bookmaker["key"]
                        existing_odds = session.scalar(
                            select(Odds).where(
                                Odds.match_id == db_match.id,
                                Odds.bookmaker == bk_key,
                            )
                        )

                        bk_updated = datetime.fromisoformat(
                            bookmaker.get("last_update", raw["collected_at"])
                        )

                        if existing_odds is None:
                            session.add(Odds(
                                match_id=db_match.id,
                                bookmaker=bk_key,
                                home_win=home_win,
                                draw=draw,
                                away_win=away_win,
                                collected_at=bk_updated,
                            ))
                            inserted += 1
                        else:
                            existing_odds.home_win    = home_win
                            existing_odds.draw        = draw
                            existing_odds.away_win    = away_win
                            existing_odds.collected_at = bk_updated
                            updated += 1

                except (KeyError, TypeError, ValueError) as e:
                    errors += 1
                    print(f"  [ERROR] odds {match_data.get('id', '?')}: {e}")

    print(
        f"  Odds — insertadas: {inserted} | actualizadas: {updated} "
        f"| sin partido: {skipped} | errores: {errors}"
    )


def _find_match(session, match_data: dict) -> Match | None:
    """Busca un Match por nombre de equipos y fecha (día)."""
    commence = datetime.fromisoformat(match_data["commence_time"])
    target_date: date = commence.date()
    home = match_data["home_team"]
    away = match_data["away_team"]

    result = session.scalars(
        select(Match).where(
            Match.home_team_name == home,
            Match.away_team_name == away,
        )
    ).all()

    for m in result:
        if m.match_date.date() == target_date:
            return m

    return None


def _parse_h2h(
    outcomes: list[dict], home_team: str, away_team: str
) -> tuple[float | None, float | None, float | None]:
    """Extrae (home_win, draw, away_win) de los outcomes h2h."""
    prices: dict[str, float] = {o["name"]: o["price"] for o in outcomes}
    return (
        prices.get(home_team),
        prices.get("Draw"),
        prices.get(away_team),
    )


# ---------------------------------------------------------------------------
# Multi-season ingest
# ---------------------------------------------------------------------------

def ingest_all_seasons(seasons: list[int] = [2022, 2023, 2024]) -> None:
    """Lee fixtures_{season}.json para cada temporada y las ingesta con upsert."""
    for season in seasons:
        path = config.DATA_DIR / f"fixtures_{season}.json"
        if not path.exists():
            print(f"  [SKIP] {path.name} no encontrado — ejecuta collect_all_seasons() primero")
            continue
        ingest_fixtures(path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_db()
    print()
    ingest_all_seasons()
    print()
    ingest_odds()
