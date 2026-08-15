"""
collectors/apifootball_collector.py

Single-source collector for API-Football (v3). Replaces the multi-source
stack (football-data CSVs + The Odds API + Understat) now that the user has a
paid PRO plan (all seasons, all competitions).

Rate limit: PRO = 300 req/min. We throttle to ~3 req/s (~180/min) — safely
under the limit so the firewall never flags abnormal spikes.

This module handles fixtures/results ingestion. Odds/injuries/lineups/xG are
added in later phases.
"""

import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).parent.parent))
import config
from database.db import get_engine, get_session, init_db
from database.models import Base, Match, MarketOdds, Odds


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

_BASE = config.API_FOOTBALL_BASE_URL          # https://v3.football.api-sports.io
_HEADERS = {"x-apisports-key": config.API_FOOTBALL_KEY}

# The user's leagues: league_id → (country, display name)
LEAGUES: dict[int, tuple[str, str]] = {
    113: ("Sweden",     "Allsvenskan"),
    114: ("Sweden",     "Superettan"),
    244: ("Finland",    "Veikkausliiga"),
    245: ("Finland",    "Ykkönen"),
    164: ("Iceland",    "Úrvalsdeild"),
    165: ("Iceland",    "1. Deild"),
    103: ("Norway",     "Eliteserien"),
    104: ("Norway",     "1. Division"),
    94:  ("Portugal",   "Primeira Liga"),
    95:  ("Portugal",   "Segunda Liga"),
    39:  ("England",    "Premier League"),
    140: ("Spain",      "La Liga"),
    135: ("Italy",      "Serie A"),
    78:  ("Germany",    "Bundesliga"),
    61:  ("France",     "Ligue 1"),
    262: ("Mexico",       "Liga MX"),
    253: ("USA",          "MLS"),
    71:  ("Brazil",       "Serie A"),
    72:  ("Brazil",       "Serie B"),
    128: ("Argentina",    "Liga Profesional"),
    162: ("Costa Rica",   "Primera División"),
    307: ("Saudi Arabia", "Pro League"),
    88:  ("Netherlands",  "Eredivisie"),
    89:  ("Netherlands",  "Eerste Divisie"),
}

DEFAULT_SEASONS = [2023, 2024, 2025, 2026]

# Leagues that provide per-match statistics (corners/shots/cards/xG). The rest
# (Iceland, Costa Rica, Finland/Norway 2nd) return empty stats — skip them.
STAT_LEAGUES = {113, 114, 244, 103, 94, 95, 39, 140, 135, 78, 61, 262, 253, 71, 72, 128,
                307, 88, 89}

_STATUS_FINISHED = {"FT", "AET", "PEN"}
_STATUS_SCHEDULED = {"NS", "TBD", "PST"}


# ---------------------------------------------------------------------------
# Throttled HTTP
# ---------------------------------------------------------------------------

_MIN_INTERVAL = 0.30           # seconds between requests → ~200/min (< 300 cap)
_last_call = [0.0]


def _throttle() -> None:
    dt = time.monotonic() - _last_call[0]
    if dt < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - dt)
    _last_call[0] = time.monotonic()


def _get(endpoint: str, **params) -> dict:
    """Throttled GET with basic error surfacing."""
    _throttle()
    r = requests.get(f"{_BASE}/{endpoint}", headers=_HEADERS, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    errs = j.get("errors")
    if errs:
        # errors can be {} (ok) or a dict/list with messages
        if isinstance(errs, dict) and errs:
            raise RuntimeError(f"API error {endpoint} {params}: {errs}")
        if isinstance(errs, list) and errs:
            raise RuntimeError(f"API error {endpoint} {params}: {errs}")
    return j


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_fixtures(league_id: int, season: int) -> list[dict]:
    """All fixtures (with results) for a league-season (single response)."""
    j = _get("fixtures", league=league_id, season=season)
    return j.get("response", [])


def _map_status(short: str) -> str:
    if short in _STATUS_FINISHED:
        return "finished"
    if short in _STATUS_SCHEDULED:
        return "scheduled"
    return "other"


# ---------------------------------------------------------------------------
# Upcoming fixtures + odds
# ---------------------------------------------------------------------------

def fetch_fixtures_by_date(day: str) -> list[dict]:
    """All fixtures worldwide for a given YYYY-MM-DD (filter to our leagues)."""
    return _get("fixtures", date=day).get("response", [])


def fetch_odds(fixture_id: int) -> list[dict]:
    return _get("odds", fixture=fixture_id).get("response", [])


def _pick_bookmaker(bookmakers: list, prefer=("Bet365", "Pinnacle", "Unibet")) -> dict | None:
    for p in prefer:
        b = next((b for b in bookmakers if b.get("name") == p), None)
        if b:
            return b
    return bookmakers[0] if bookmakers else None


def _parse_markets(book: dict) -> dict:
    """Extract 1X2, O/U 2.5 and BTTS from one bookmaker's bets."""
    bets = {b["name"]: b["values"] for b in book.get("bets", [])}
    out = {"x2": None, "ou25": None, "btts": None}

    mw = bets.get("Match Winner")
    if mw:
        d = {v["value"]: float(v["odd"]) for v in mw}
        if {"Home", "Draw", "Away"} <= d.keys():
            out["x2"] = (d["Home"], d["Draw"], d["Away"])

    ou = bets.get("Goals Over/Under")
    if ou:
        d = {v["value"]: float(v["odd"]) for v in ou}
        if "Over 2.5" in d and "Under 2.5" in d:
            out["ou25"] = (d["Over 2.5"], d["Under 2.5"])

    bt = bets.get("Both Teams Score")
    if bt:
        d = {v["value"]: float(v["odd"]) for v in bt}
        if "Yes" in d and "No" in d:
            out["btts"] = (d["Yes"], d["No"])

    return out


def collect_upcoming(days_ahead: int = 3, leagues: dict = LEAGUES) -> None:
    """
    Fetch upcoming fixtures (today..+days_ahead) for our leagues, upsert them as
    matches, and store 1X2 / O-U 2.5 / BTTS odds (preferred bookmaker) for each.
    """
    init_db()   # ensure market_odds table exists
    our = set(leagues)

    fixtures: list[dict] = []
    for d in range(days_ahead + 1):
        day = (date.today() + timedelta(days=d)).isoformat()
        for x in fetch_fixtures_by_date(day):
            if x["league"]["id"] in our and x["fixture"]["status"]["short"] in _STATUS_SCHEDULED:
                fixtures.append(x)

    print(f"  {len(fixtures)} partidos próximos en tus ligas (hoy..+{days_ahead}d)")
    now = _utcnow()
    n_1x2 = n_mkt = 0

    with get_session() as session:
        for fx in fixtures:
            m = _fixture_to_match(fx)
            if m:
                session.merge(m)
        session.flush()

        for i, fx in enumerate(fixtures, 1):
            fid = fx["fixture"]["id"]
            try:
                resp = fetch_odds(fid)
            except Exception:
                continue
            if not resp:
                continue
            book = _pick_bookmaker(resp[0].get("bookmakers", []))
            if not book:
                continue
            bname = book["name"]
            mk = _parse_markets(book)

            # idempotent: clear this fixture+book first
            session.query(Odds).filter(Odds.match_id == fid, Odds.bookmaker == bname).delete()
            session.query(MarketOdds).filter(MarketOdds.match_id == fid, MarketOdds.bookmaker == bname).delete()

            if mk["x2"]:
                h, dr, a = mk["x2"]
                session.add(Odds(match_id=fid, bookmaker=bname, market="1x2",
                                 home_win=h, draw=dr, away_win=a, collected_at=now))
                n_1x2 += 1
            if mk["ou25"]:
                ov, un = mk["ou25"]
                session.add(MarketOdds(match_id=fid, bookmaker=bname, market="ou_goals",
                                       selection="over", line=2.5, odd=ov, collected_at=now))
                session.add(MarketOdds(match_id=fid, bookmaker=bname, market="ou_goals",
                                       selection="under", line=2.5, odd=un, collected_at=now))
                n_mkt += 1
            if mk["btts"]:
                y, n = mk["btts"]
                session.add(MarketOdds(match_id=fid, bookmaker=bname, market="btts",
                                       selection="yes", line=None, odd=y, collected_at=now))
                session.add(MarketOdds(match_id=fid, bookmaker=bname, market="btts",
                                       selection="no", line=None, odd=n, collected_at=now))
                n_mkt += 1

            if i % 25 == 0:
                print(f"    odds {i}/{len(fixtures)}...")

    print(f"  Guardado: 1X2 para {n_1x2} partidos · mercados O-U/BTTS para {n_mkt}")


# ---------------------------------------------------------------------------
# Recent results (refresh finished matches of the last few days)
# ---------------------------------------------------------------------------

def collect_recent_results(days_back: int = 2, leagues: dict = LEAGUES,
                           with_stats: bool = True) -> None:
    """Fetch the last `days_back` days of fixtures for our leagues and upsert
    them (final status + goals). Also ingests per-match stats for the finished
    matches in stats-enabled leagues (corners/shots/cards)."""
    from sqlalchemy import distinct, select
    from database.models import TeamStatistic

    init_db()
    our = set(leagues)
    fixtures: list[dict] = []
    for d in range(days_back + 1):
        day = (date.today() - timedelta(days=d)).isoformat()
        for x in fetch_fixtures_by_date(day):
            if x["league"]["id"] in our:
                fixtures.append(x)

    now = _utcnow()
    finished = 0
    stat_fids: list[int] = []
    with get_session() as session:
        for fx in fixtures:
            m = _fixture_to_match(fx)
            if m is None:
                continue
            session.merge(m)
            if m.status == "finished" and m.home_goals is not None:
                finished += 1
                if m.league_id in STAT_LEAGUES:
                    stat_fids.append(m.id)
        session.flush()

        n_stats = 0
        if with_stats and stat_fids:
            have = set(session.execute(select(distinct(TeamStatistic.match_id))).scalars().all())
            todo = [f for f in stat_fids if f not in have]
            for fid in todo:
                try:
                    resp = fetch_stats(fid)
                except Exception:
                    continue
                if not resp:
                    continue
                for block in resp:
                    tid = block.get("team", {}).get("id")
                    if tid is None:
                        continue
                    for st in block.get("statistics", []):
                        num, sstr = _stat_value(st.get("value"))
                        session.add(TeamStatistic(
                            match_id=fid, team_id=tid, type=st["type"],
                            value_num=num, value_str=sstr, collected_at=now,
                        ))
                n_stats += 1

    print(f"  {len(fixtures)} fixtures recientes actualizados (últimos {days_back}d) · "
          f"{finished} finalizados · stats ingestadas de {n_stats}")


# ---------------------------------------------------------------------------
# Per-match statistics (corners / shots / cards / saves / xG)
# ---------------------------------------------------------------------------

def fetch_stats(fixture_id: int) -> list[dict]:
    return _get("fixtures/statistics", fixture=fixture_id).get("response", [])


def _stat_value(v) -> tuple[float | None, str | None]:
    """Parse an API stat value into (numeric, raw_string). '52%' → (52.0, '52%')."""
    if v is None:
        return None, None
    if isinstance(v, (int, float)):
        return float(v), None
    sv = str(v).strip()
    if sv.endswith("%"):
        try:
            return float(sv[:-1]), sv
        except ValueError:
            return None, sv
    try:
        return float(sv), None
    except ValueError:
        return None, sv


def collect_stats_for_upcoming(days_ahead: int = 3, n_recent: int = 10,
                               leagues: dict = LEAGUES) -> None:
    """
    For every team playing in an upcoming (odds-collected) fixture in a
    stats-enabled league, fetch + store per-match statistics for its last
    `n_recent` finished fixtures. Idempotent: skips fixtures already stored.
    """
    from sqlalchemy import distinct, select
    from database.models import Odds, TeamStatistic

    init_db()
    now = _utcnow()

    with get_session() as session:
        up_ids = [o.match_id for o in
                  session.execute(select(Odds).where(Odds.market == "1x2")).scalars().all()]
        upcoming = session.execute(select(Match).where(Match.id.in_(up_ids))).scalars().all()

        teams: set[int] = set()
        for m in upcoming:
            if m.league_id in STAT_LEAGUES:
                teams.add(m.home_team_id)
                teams.add(m.away_team_id)
        print(f"  equipos en partidos próximos (ligas con stats): {len(teams)}")

        fixture_ids: set[int] = set()
        for tid in teams:
            rows = session.execute(
                select(Match.id).where(
                    (Match.home_team_id == tid) | (Match.away_team_id == tid),
                    Match.status == "finished",
                    Match.home_goals.is_not(None),
                ).order_by(Match.match_date.desc()).limit(n_recent)
            ).scalars().all()
            fixture_ids.update(rows)

        have = set(session.execute(select(distinct(TeamStatistic.match_id))).scalars().all())
        todo = [f for f in fixture_ids if f not in have]
        print(f"  fixtures a consultar: {len(todo)} "
              f"(de {len(fixture_ids)} recientes; {len(fixture_ids) - len(todo)} ya tenían stats)")

        n_ok = 0
        for i, fid in enumerate(todo, 1):
            try:
                resp = fetch_stats(fid)
            except Exception:
                continue
            if not resp:
                continue
            for block in resp:
                tid = block.get("team", {}).get("id")
                if tid is None:
                    continue
                for st in block.get("statistics", []):
                    num, sstr = _stat_value(st.get("value"))
                    session.add(TeamStatistic(
                        match_id=fid, team_id=tid, type=st["type"],
                        value_num=num, value_str=sstr, collected_at=now,
                    ))
            n_ok += 1
            if i % 50 == 0:
                session.flush()
                print(f"    stats {i}/{len(todo)}...")

    print(f"  Stats guardadas para {n_ok} fixtures")


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def _fixture_to_match(fx: dict) -> Match | None:
    try:
        f = fx["fixture"]
        lg = fx["league"]
        tm = fx["teams"]
        goals = fx["goals"]
        ht = (fx.get("score") or {}).get("halftime") or {}
        venue = (f.get("venue") or {}).get("name")
        return Match(
            id=f["id"],
            league_id=lg["id"],
            league_name=lg["name"],
            country=lg.get("country"),
            season=lg["season"],
            round=lg.get("round"),
            home_team_id=tm["home"]["id"],
            home_team_name=tm["home"]["name"],
            away_team_id=tm["away"]["id"],
            away_team_name=tm["away"]["name"],
            match_date=datetime.fromisoformat(f["date"]),
            status=_map_status(f["status"]["short"]),
            home_goals=goals.get("home"),
            away_goals=goals.get("away"),
            home_goals_ht=ht.get("home"),
            away_goals_ht=ht.get("away"),
            venue=venue,
            referee=f.get("referee"),
        )
    except (KeyError, TypeError, ValueError):
        return None


def rebuild(
    seasons: list[int] = DEFAULT_SEASONS,
    leagues: dict[int, tuple[str, str]] = LEAGUES,
    wipe: bool = True,
) -> None:
    """
    Rebuild the matches table from API-Football for all leagues × seasons.

    wipe=True recreates the whole schema (drop_all + create_all) — clean
    single-source rebuild that also applies any schema changes and avoids
    duplicates with the old synthetic-ID CSV rows. Back up the DB first.
    """
    if wipe:
        print("  Recreando esquema (drop_all + create_all)...")
        Base.metadata.drop_all(get_engine())
        Base.metadata.create_all(get_engine())
    else:
        init_db()

    total_calls = len(leagues) * len(seasons)
    print(f"Reconstruyendo BD desde API-Football — {len(leagues)} ligas × "
          f"{len(seasons)} temporadas = {total_calls} llamadas "
          f"(throttle ~{1/_MIN_INTERVAL:.0f}/s)\n")

    with get_session() as session:
        grand_total = 0
        finished_total = 0
        call_i = 0
        for league_id, (country, name) in leagues.items():
            league_count = 0
            for season in seasons:
                call_i += 1
                try:
                    fixtures = fetch_fixtures(league_id, season)
                except Exception as exc:
                    print(f"  [{call_i}/{total_calls}] {country} {name} {season}: ERROR {exc}")
                    continue

                inserted = 0
                for fx in fixtures:
                    m = _fixture_to_match(fx)
                    if m is None:
                        continue
                    session.merge(m)      # upsert by primary key (fixture id)
                    inserted += 1
                    if m.status == "finished" and m.home_goals is not None:
                        finished_total += 1
                league_count += inserted
                grand_total += inserted
            print(f"  {country:<11} {name:<18} → {league_count} partidos")

    print(f"\n  TOTAL: {grand_total} partidos ({finished_total} finalizados con resultado)")


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    try:                                    # Windows console is cp1252 by default
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    p = argparse.ArgumentParser(description="API-Football collector")
    p.add_argument("--seasons", nargs="+", type=int, default=DEFAULT_SEASONS)
    p.add_argument("--no-wipe", action="store_true")
    p.add_argument("--upcoming", action="store_true",
                   help="Recolectar partidos próximos + odds (no reconstruye)")
    p.add_argument("--stats", action="store_true",
                   help="Ingestar stats por-partido de equipos con partido próximo")
    p.add_argument("--results", action="store_true",
                   help="Refrescar resultados de partidos finalizados recientes")
    p.add_argument("--days", type=int, default=3, help="Días hacia adelante")
    p.add_argument("--back", type=int, default=2, help="Días hacia atrás (--results)")
    p.add_argument("--recent", type=int, default=10, help="Partidos recientes por equipo (--stats)")
    args = p.parse_args()

    if args.upcoming:
        print("=== API-Football → partidos próximos + odds ===\n")
        collect_upcoming(days_ahead=args.days)
    elif args.stats:
        print("=== API-Football → stats por-partido (equipos próximos) ===\n")
        collect_stats_for_upcoming(days_ahead=args.days, n_recent=args.recent)
    elif args.results:
        print("=== API-Football → resultados recientes ===\n")
        collect_recent_results(days_back=args.back)
    else:
        print("=== API-Football → BD (rebuild) ===\n")
        rebuild(seasons=args.seasons, wipe=not args.no_wipe)
