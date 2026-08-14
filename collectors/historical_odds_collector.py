"""
collectors/historical_odds_collector.py

Reads CSVs from data/odds_historical/, matches rows against the DB matches
table, and upserts Pinnacle + Bet365 opening odds into the odds table.
"""

import csv
import difflib
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import config
from database.db import get_session, init_db
from database.models import Match, Odds
from sqlalchemy import select

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


def _season_from_code(code: str) -> int:
    """'2324' → 2023  (start year of the season)"""
    return 2000 + int(code[:2])


# ---------------------------------------------------------------------------
# Match index
# ---------------------------------------------------------------------------

def _build_match_index(session) -> dict[tuple[int, object], list[Match]]:
    """Returns {(league_id, date) → [Match, ...]} for all DB matches."""
    idx: dict = defaultdict(list)
    for m in session.execute(select(Match)).scalars().all():
        idx[(m.league_id, m.match_date.date())].append(m)
    return idx


def _find_db_match(
    idx: dict,
    league_id: int,
    match_date,
    home_name: str,
    away_name: str,
    min_score: float = 1.2,
) -> Match | None:
    """
    Fuzzy-matches a CSV row to a DB Match.

    Gathers candidates within ±1 day, then scores each candidate by the sum
    of SequenceMatcher ratios for home + away names.  Returns the best match
    only if its combined score ≥ min_score (≈0.6 each).
    """
    candidates: list[Match] = []
    for delta in (-1, 0, 1):
        candidates.extend(idx.get((league_id, match_date + timedelta(days=delta)), []))

    # Deduplicate while preserving order
    seen: set[int] = set()
    unique = [m for m in candidates if not (m.id in seen or seen.add(m.id))]

    if not unique:
        return None

    best_score, best = 0.0, None
    hn_lower = home_name.lower()
    an_lower = away_name.lower()

    for m in unique:
        h = difflib.SequenceMatcher(None, hn_lower, m.home_team_name.lower()).ratio()
        a = difflib.SequenceMatcher(None, an_lower, m.away_team_name.lower()).ratio()
        if h + a > best_score:
            best_score = h + a
            best = m

    return best if best_score >= min_score else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_float(val: str | None) -> float | None:
    """Returns positive float or None for empty / non-positive values."""
    if not val:
        return None
    try:
        v = float(val.strip())
        return v if v > 0 else None
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    init_db()

    odds_dir = config.DATA_DIR / "odds_historical"
    csv_files = sorted(odds_dir.glob("*.csv"))

    if not csv_files:
        print(f"[ERROR] No se encontraron CSVs en {odds_dir}")
        return

    print(f"Encontrados {len(csv_files)} CSVs en {odds_dir}\n")

    with get_session() as session:
        # --- Build match index ---
        print("Cargando partidos de la BD...", end=" ", flush=True)
        match_idx = _build_match_index(session)
        n_db = sum(len(v) for v in match_idx.values())
        print(f"{n_db} entradas de índice ({len(set(m.id for ms in match_idx.values() for m in ms))} partidos únicos)\n")

        # --- Pre-load existing (match_id, bookmaker) to detect duplicates ---
        existing: dict[tuple[int, str], Odds] = {
            (o.match_id, o.bookmaker): o
            for o in session.execute(select(Odds)).scalars().all()
        }

        # Counters
        total_rows    = 0
        matched       = 0
        not_matched   = 0
        inserted      = {"Pinnacle": 0, "Bet365": 0}
        updated       = {"Pinnacle": 0, "Bet365": 0}
        skipped_odds  = 0

        for csv_path in csv_files:
            stem = csv_path.stem          # e.g. "2324_E0"
            parts = stem.split("_", 1)
            if len(parts) != 2 or parts[1] not in _DIV_MAP:
                print(f"  [SKIP] {csv_path.name} — nombre inesperado")
                continue

            season_code, div = parts
            league_id, _ = _DIV_MAP[div]

            file_rows = file_matched = 0

            with open(csv_path, encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    home_raw = row.get("HomeTeam", "").strip()
                    if not home_raw:
                        continue    # trailing empty line

                    total_rows += 1
                    file_rows  += 1

                    # Parse date
                    try:
                        match_date = datetime.strptime(row["Date"].strip(), "%d/%m/%Y").date()
                    except (ValueError, KeyError):
                        not_matched += 1
                        continue

                    away_raw = row.get("AwayTeam", "").strip()

                    db_match = _find_db_match(
                        match_idx, league_id, match_date, home_raw, away_raw
                    )

                    if db_match is None:
                        not_matched += 1
                        continue

                    matched    += 1
                    file_matched += 1

                    collected_at = datetime(
                        match_date.year, match_date.month, match_date.day,
                        tzinfo=timezone.utc,
                    )

                    # --- Pinnacle (PSH / PSD / PSA) ---
                    ps_h = _parse_float(row.get("PSH"))
                    ps_d = _parse_float(row.get("PSD"))
                    ps_a = _parse_float(row.get("PSA"))

                    if ps_h and ps_d and ps_a:
                        key = (db_match.id, "Pinnacle")
                        if key in existing:
                            rec = existing[key]
                            rec.home_win = ps_h
                            rec.draw     = ps_d
                            rec.away_win = ps_a
                            rec.collected_at = collected_at
                            updated["Pinnacle"] += 1
                        else:
                            new = Odds(
                                match_id=db_match.id, bookmaker="Pinnacle",
                                home_win=ps_h, draw=ps_d, away_win=ps_a,
                                collected_at=collected_at,
                            )
                            session.add(new)
                            existing[key] = new
                            inserted["Pinnacle"] += 1
                    else:
                        skipped_odds += 1

                    # --- Bet365 (B365H / B365D / B365A) ---
                    b3_h = _parse_float(row.get("B365H"))
                    b3_d = _parse_float(row.get("B365D"))
                    b3_a = _parse_float(row.get("B365A"))

                    if b3_h and b3_d and b3_a:
                        key = (db_match.id, "Bet365")
                        if key in existing:
                            rec = existing[key]
                            rec.home_win = b3_h
                            rec.draw     = b3_d
                            rec.away_win = b3_a
                            rec.collected_at = collected_at
                            updated["Bet365"] += 1
                        else:
                            new = Odds(
                                match_id=db_match.id, bookmaker="Bet365",
                                home_win=b3_h, draw=b3_d, away_win=b3_a,
                                collected_at=collected_at,
                            )
                            session.add(new)
                            existing[key] = new
                            inserted["Bet365"] += 1
                    else:
                        skipped_odds += 1

            pct = f"{file_matched / file_rows:.0%}" if file_rows else "—"
            print(f"  {csv_path.name:<20}  {file_rows:>4} filas  |  matched: {file_matched:>4} ({pct})")

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    SEP = "=" * 56
    print(f"\n{SEP}")
    print("  RESUMEN")
    print(SEP)
    print(f"  Filas procesadas       : {total_rows}")
    print(f"  Partidos encontrados   : {matched:>5}  ({matched / max(total_rows, 1):.1%})")
    print(f"  Partidos no encontrados: {not_matched:>5}  ({not_matched / max(total_rows, 1):.1%})")
    print(f"  Odds sin datos         : {skipped_odds}")
    print(f"\n  Odds insertadas / actualizadas por bookmaker:")
    for bk in ("Pinnacle", "Bet365"):
        ins = inserted[bk]
        upd = updated[bk]
        print(f"    {bk:<10}  +{ins:<5} insertadas   ~{upd:<5} actualizadas   = {ins + upd} total")
    print(SEP)


if __name__ == "__main__":
    run()
