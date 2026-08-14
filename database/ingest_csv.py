"""
database/ingest_csv.py

Ingests match results + Pinnacle / Bet365 opening odds from
football-data.co.uk CSVs into the Match and Odds tables.

Two jobs in one pass:
  - Seasons already in the DB (2022-2025, from API-Football): finds each
    match via fuzzy name + date and upserts Pinnacle / Bet365 odds.
  - Seasons NOT in the DB (2020/21, 2021/22): creates Match records with
    synthetic IDs (≥ 2_000_000_000) and inserts odds.

Synthetic IDs are deterministic hashes of (league_id, season, home, away)
so re-running is idempotent.

Usage
-----
    python database/ingest_csv.py               # download missing CSVs + ingest all
    python database/ingest_csv.py --no-download # skip download step
"""

import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path

import requests
from sqlalchemy import case, distinct, func, select

sys.path.append(str(Path(__file__).parent.parent))
import config
from database.db import get_session, init_db
from database.models import Match, Odds

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DIV_MAP: dict[str, tuple[int, str]] = {
    "E0":  (39,  "Premier League"),
    "SP1": (140, "La Liga"),
    "D1":  (78,  "Bundesliga"),
    "I1":  (135, "Serie A"),
    "F1":  (61,  "Ligue 1"),
}

# API-Football fixture IDs top out around ~1.5 M; our synthetic IDs start at 2 B
_MATCH_ID_OFFSET = 2_000_000_000
_TEAM_ID_OFFSET  = 3_000_000_000

# football-data.co.uk short names → canonical API-Football names stored in DB
_TEAM_NAME_MAP: dict[str, str] = {
    # Bundesliga
    "Ein Frankfurt":    "Eintracht Frankfurt",
    "M'gladbach":       "Borussia Monchengladbach",
    "Monchengladbach":  "Borussia Monchengladbach",
    "Koln":             "1. FC Köln",
    "Mainz":            "FSV Mainz 05",
    "Dortmund":         "Borussia Dortmund",
    "Wolfsburg":        "VfL Wolfsburg",
    "Hertha":           "Hertha Berlin",
    "Schalke":          "Schalke 04",
    "Leverkusen":       "Bayer Leverkusen",
    "Hoffenheim":       "TSG Hoffenheim",
    "Stuttgart":        "VfB Stuttgart",
    "Augsburg":         "FC Augsburg",
    "Freiburg":         "SC Freiburg",
    "Bochum":           "VfL Bochum",
    "Heidenheim":       "1. FC Heidenheim",
    "Union Berlin":     "1. FC Union Berlin",
    "Greuther Furth":   "SpVgg Greuther Fürth",
    "Bielefeld":        "Arminia Bielefeld",
    "Hamburg":          "Hamburger SV",
    "St Pauli":         "FC St. Pauli",
    "Paderborn":        "SC Paderborn 07",
    "Dusseldorf":       "Fortuna Düsseldorf",
    "Nurnberg":         "1. FC Nürnberg",
    "Hannover":         "Hannover 96",
    # Premier League
    "Man United":       "Manchester United",
    "Man City":         "Manchester City",
    "Tottenham":        "Tottenham Hotspur",
    "Newcastle":        "Newcastle United",
    "Wolves":           "Wolverhampton Wanderers",
    "Leicester":        "Leicester City",
    "Leeds":            "Leeds United",
    "Norwich":          "Norwich City",
    "Sheff Utd":        "Sheffield United",
    "West Brom":        "West Bromwich Albion",
    "West Ham":         "West Ham United",
    "Nott'm Forest":    "Nottingham Forest",
    # Serie A
    "AC Milan":         "Milan",
    "Inter Milan":      "Inter",
    "Internazionale":   "Inter",
    "Verona":           "Hellas Verona",
    # La Liga
    "Ath Bilbao":       "Athletic Club",
    "Ath Madrid":       "Atletico Madrid",
    "Atletico Madrid":  "Atletico Madrid",
    "Alaves":           "Deportivo Alaves",
    "Celta":            "Celta Vigo",
    "Espanol":          "Espanyol",
    "Betis":            "Real Betis",
    "Sociedad":         "Real Sociedad",
    "Vallecano":        "Rayo Vallecano",
    # Ligue 1
    "Paris SG":         "Paris Saint Germain",
    "Paris Saint-Germain": "Paris Saint Germain",
    "St Etienne":       "Saint-Etienne",
    # ingest.py canonicals — pass through unchanged
    "Bayern München":   "Bayern Munich",
    "Atlético Madrid":  "Atletico Madrid",
}

# CSVs to download when missing (season_code, division)
_CSV_DOWNLOADS: list[tuple[str, str]] = [
    (code, div)
    for code in ("2021", "2122")
    for div in ("E0", "D1", "SP1", "I1", "F1")
]
_FD_BASE = "https://www.football-data.co.uk/mmz4281"


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------

def _synthetic_match_id(league_id: int, season: int, home: str, away: str) -> int:
    key = f"{league_id}|{season}|{home}|{away}"
    digest = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return _MATCH_ID_OFFSET + (digest % 900_000_000)


def _synthetic_team_id(name: str) -> int:
    digest = int(hashlib.md5(name.encode()).hexdigest(), 16)
    return _TEAM_ID_OFFSET + (digest % 900_000_000)


# ---------------------------------------------------------------------------
# Name / date helpers
# ---------------------------------------------------------------------------

def _normalize(name: str) -> str:
    return _TEAM_NAME_MAP.get(name.strip(), name.strip())


def _season_from_code(code: str) -> int:
    """'2324' → 2023  (start year of the season)."""
    return 2000 + int(code[:2])


def _parse_float(val: str | None) -> float | None:
    if not val:
        return None
    try:
        v = float(val.strip())
        return v if v > 0 else None
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# CSV download
# ---------------------------------------------------------------------------

def download_missing_csvs(odds_dir: Path) -> list[str]:
    downloaded: list[str] = []
    for season_code, div in _CSV_DOWNLOADS:
        filename = f"{season_code}_{div}.csv"
        path = odds_dir / filename
        if path.exists():
            print(f"  [skip]  {filename} ya existe")
            continue
        url = f"{_FD_BASE}/{season_code}/{div}.csv"
        print(f"  Descargando {filename} ...", end=" ", flush=True)
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            # Sanity-check: real CSV starts with column headers, not HTML
            if b"<html" in resp.content[:200].lower():
                print("ERROR — la URL devolvió HTML (¿temporada no disponible?)")
                continue
            path.write_bytes(resp.content)
            print(f"OK  ({len(resp.content):,} bytes)")
            downloaded.append(filename)
        except Exception as exc:
            print(f"ERROR: {exc}")
    return downloaded


# ---------------------------------------------------------------------------
# Match index + fuzzy lookup
# ---------------------------------------------------------------------------

def _build_match_index(session) -> dict[tuple[int, object], list[Match]]:
    idx: dict = defaultdict(list)
    for m in session.execute(select(Match)).scalars().all():
        idx[(m.league_id, m.match_date.date())].append(m)
    return idx


def _find_match(
    idx: dict,
    league_id: int,
    match_date,
    home: str,
    away: str,
    min_score: float = 1.2,
) -> "Match | None":
    candidates: list[Match] = []
    for delta in (-1, 0, 1):
        candidates.extend(idx.get((league_id, match_date + timedelta(days=delta)), []))

    seen: set[int] = set()
    unique = [m for m in candidates if not (m.id in seen or seen.add(m.id))]
    if not unique:
        return None

    hl, al = home.lower(), away.lower()
    best_score, best = 0.0, None
    for m in unique:
        h = SequenceMatcher(None, hl, m.home_team_name.lower()).ratio()
        a = SequenceMatcher(None, al, m.away_team_name.lower()).ratio()
        if h + a > best_score:
            best_score, best = h + a, m

    return best if best_score >= min_score else None


# ---------------------------------------------------------------------------
# Main ingest
# ---------------------------------------------------------------------------

def run(
    odds_dir: Path = config.DATA_DIR / "odds_historical",
    download: bool = True,
) -> None:
    init_db()

    if download:
        print("Verificando / descargando CSVs de 2020/21 y 2021/22...")
        newly = download_missing_csvs(odds_dir)
        print(f"  {len(newly)} archivo(s) nuevo(s) descargado(s)\n")

    csv_files = sorted(odds_dir.glob("*.csv"))
    if not csv_files:
        print(f"[ERROR] No hay CSVs en {odds_dir}")
        return

    print(f"Procesando {len(csv_files)} CSV(s)...\n")

    matches_created = matches_found = matches_skipped = 0
    odds_inserted: dict[str, int] = {"Pinnacle": 0, "Bet365": 0}
    odds_updated:  dict[str, int] = {"Pinnacle": 0, "Bet365": 0}
    odds_missing = 0

    with get_session() as session:
        print("Cargando índice de partidos...", end=" ", flush=True)
        match_idx = _build_match_index(session)
        print(f"{sum(len(v) for v in match_idx.values())} registros\n")

        # Pre-load all existing odds into a dict for O(1) lookup + in-place update
        existing_odds: dict[tuple[int, str], Odds] = {
            (o.match_id, o.bookmaker): o
            for o in session.execute(select(Odds)).scalars().all()
        }

        for csv_path in csv_files:
            stem = csv_path.stem
            parts = stem.split("_", 1)
            if len(parts) != 2 or parts[1] not in _DIV_MAP:
                print(f"  [SKIP] {csv_path.name}")
                continue

            season_code, div = parts
            league_id, league_name = _DIV_MAP[div]
            season = _season_from_code(season_code)

            f_created = f_found = f_skipped = 0

            with open(csv_path, encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    home_raw = row.get("HomeTeam", "").strip()
                    if not home_raw:
                        continue  # trailing blank line

                    away_raw = row.get("AwayTeam", "").strip()
                    home = _normalize(home_raw)
                    away = _normalize(away_raw)

                    # Parse date + time
                    try:
                        time_str = row.get("Time", "").strip() or "00:00"
                        match_dt = datetime.strptime(
                            f"{row['Date'].strip()} {time_str}", "%d/%m/%Y %H:%M"
                        ).replace(tzinfo=timezone.utc)
                        home_goals = int(row["FTHG"])
                        away_goals = int(row["FTAG"])
                    except (ValueError, KeyError):
                        matches_skipped += 1
                        f_skipped += 1
                        continue

                    # --- Find or create Match ---
                    db_match = _find_match(match_idx, league_id, match_dt.date(), home, away)

                    if db_match is None:
                        mid = _synthetic_match_id(league_id, season, home, away)
                        db_match = session.get(Match, mid)
                        if db_match is None:
                            db_match = Match(
                                id=mid,
                                league_id=league_id,
                                league_name=league_name,
                                season=season,
                                home_team_id=_synthetic_team_id(home),
                                home_team_name=home,
                                away_team_id=_synthetic_team_id(away),
                                away_team_name=away,
                                match_date=match_dt,
                                status="finished",
                                home_goals=home_goals,
                                away_goals=away_goals,
                            )
                            session.add(db_match)
                            match_idx[(league_id, match_dt.date())].append(db_match)
                            matches_created += 1
                            f_created += 1
                        else:
                            matches_found += 1
                            f_found += 1
                    else:
                        matches_found += 1
                        f_found += 1

                    collected_at = datetime(
                        match_dt.year, match_dt.month, match_dt.day,
                        tzinfo=timezone.utc,
                    )

                    # --- Upsert Pinnacle + Bet365 ---
                    for bk_name, h_col, d_col, a_col in (
                        ("Pinnacle", "PSH",   "PSD",   "PSA"),
                        ("Bet365",   "B365H", "B365D", "B365A"),
                    ):
                        h_val = _parse_float(row.get(h_col))
                        d_val = _parse_float(row.get(d_col))
                        a_val = _parse_float(row.get(a_col))

                        if not (h_val and d_val and a_val):
                            odds_missing += 1
                            continue

                        key = (db_match.id, bk_name)
                        if key in existing_odds:
                            rec = existing_odds[key]
                            rec.home_win    = h_val
                            rec.draw        = d_val
                            rec.away_win    = a_val
                            rec.collected_at = collected_at
                            odds_updated[bk_name] += 1
                        else:
                            new_odds = Odds(
                                match_id=db_match.id,
                                bookmaker=bk_name,
                                home_win=h_val,
                                draw=d_val,
                                away_win=a_val,
                                collected_at=collected_at,
                            )
                            session.add(new_odds)
                            existing_odds[key] = new_odds
                            odds_inserted[bk_name] += 1

            total = f_created + f_found
            pct = f"{total / max(total + f_skipped, 1):.0%}"
            print(
                f"  {csv_path.name:<20}  "
                f"creados: {f_created:>4}  "
                f"enlazados: {f_found:>4}  "
                f"skip: {f_skipped:>3}  ({pct})"
            )

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    _LEAGUE_LABEL = {v[0]: v[1] for v in _DIV_MAP.values()}

    SEP = "=" * 68
    print(f"\n{SEP}")
    print("  RESUMEN INGEST CSV")
    print(SEP)
    print(f"  Partidos creados      : {matches_created}")
    print(f"  Partidos enlazados    : {matches_found}  (ya existían en BD)")
    print(f"  Filas con error       : {matches_skipped}")
    print(f"  Odds insertadas       : Pinnacle {odds_inserted['Pinnacle']:>5}  |  Bet365 {odds_inserted['Bet365']:>5}")
    print(f"  Odds actualizadas     : Pinnacle {odds_updated['Pinnacle']:>5}  |  Bet365 {odds_updated['Bet365']:>5}")
    print(f"  Odds sin columna      : {odds_missing}")

    print(f"\n  {'Season':>6}  {'Liga':>7}  {'Nombre':<20}  {'Partidos':>9}  {'Pinnacle':>9}  {'Bet365':>9}")
    print("  " + "-" * 66)

    with get_session() as session:
        rows = session.execute(
            select(
                Match.season,
                Match.league_id,
                func.count(distinct(Match.id)).label("matches"),
                func.sum(case((Odds.bookmaker == "Pinnacle", 1), else_=0)).label("pinnacle"),
                func.sum(case((Odds.bookmaker == "Bet365",   1), else_=0)).label("bet365"),
            )
            .outerjoin(Odds, Odds.match_id == Match.id)
            .group_by(Match.season, Match.league_id)
            .order_by(Match.season, Match.league_id)
        ).all()

        for r in rows:
            liga = _LEAGUE_LABEL.get(r.league_id, str(r.league_id))
            print(
                f"  {r.season:>6}  {r.league_id:>7}  {liga:<20}  "
                f"{r.matches:>9}  {r.pinnacle or 0:>9}  {r.bet365 or 0:>9}"
            )

    print(SEP)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest football-data.co.uk CSVs into DB")
    parser.add_argument("--no-download", action="store_true", help="Skip CSV download")
    args = parser.parse_args()

    print("=== Ingest CSV: football-data.co.uk -> BD ===\n")
    run(download=not args.no_download)
