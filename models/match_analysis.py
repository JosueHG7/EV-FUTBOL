"""
models/match_analysis.py

Pre-match analysis dossier — automates the manual research (form, history,
H2H, xG, rest) the user does by hand, and adds the model's probability
estimates (1X2 + goal markets).

This is a DECISION-SUPPORT tool, not a +EV tipster: it surfaces the
analysis and where the model diverges from the market, but makes no claim
of beating closing odds (see backtester audit — no edge vs Pinnacle/Bet365).

Public API
----------
build_dossier(model, matches_df, xg_df, home, away, commence_time, market_odds)
    -> dict with sections: model_probs, goal_markets, form, xg, h2h, rest, reading
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from models.ev_calculator import remove_vig


# Single-source (API-Football): fixtures, odds and the DB all use the same team
# names, so no cross-source mapping is needed. Kept as a hook in case a future
# odds source needs remapping.
ODDS_API_TEAM_MAP: dict[str, str] = {}


def normalize_team(name: str) -> str:
    return ODDS_API_TEAM_MAP.get(name, name)


# ---------------------------------------------------------------------------
# Recent-form / venue helpers  (compute the team's CURRENT state from the DB)
# ---------------------------------------------------------------------------

def _team_matches(
    matches_df: pd.DataFrame, team: str, before: pd.Timestamp,
    venue: str | None = None,
) -> pd.DataFrame:
    """Finished matches of `team` strictly before `before`, newest first.

    venue: None = all, "home" = only where team plays home, "away" = only away.
    """
    if venue == "home":
        mask = matches_df["home_team"] == team
    elif venue == "away":
        mask = matches_df["away_team"] == team
    else:
        mask = (matches_df["home_team"] == team) | (matches_df["away_team"] == team)
    sub = matches_df[mask & (matches_df["match_date"] < before)]
    return sub.sort_values("match_date", ascending=False)


def _form(matches_df: pd.DataFrame, team: str, before: pd.Timestamp,
          n: int = 5, venue: str | None = None) -> dict:
    """Recent form from the team's perspective (last n matches)."""
    sub = _team_matches(matches_df, team, before, venue).head(n)
    if sub.empty:
        return {"results": [], "gf": None, "ga": None, "points": 0, "n": 0}

    results, gf, ga, pts = [], [], [], 0
    for _, m in sub.iterrows():
        is_home = m["home_team"] == team
        scored   = m["home_goals"] if is_home else m["away_goals"]
        conceded = m["away_goals"] if is_home else m["home_goals"]
        gf.append(scored); ga.append(conceded)
        if scored > conceded:
            results.append("V"); pts += 3
        elif scored == conceded:
            results.append("E"); pts += 1
        else:
            results.append("D")
    return {
        "results": results,                       # newest first
        "gf": round(float(np.mean(gf)), 2),
        "ga": round(float(np.mean(ga)), 2),
        "points": pts,
        "n": len(sub),
    }


def _points_last(matches_df: pd.DataFrame, team: str, before: pd.Timestamp,
                 n: int) -> int:
    sub = _team_matches(matches_df, team, before, None).head(n)
    pts = 0
    for _, m in sub.iterrows():
        is_home = m["home_team"] == team
        s = m["home_goals"] if is_home else m["away_goals"]
        c = m["away_goals"] if is_home else m["home_goals"]
        pts += 3 if s > c else (1 if s == c else 0)
    return pts


def _rest(matches_df: pd.DataFrame, team: str, before: pd.Timestamp) -> dict:
    """Days since last match and fixture congestion (games in last 14 days)."""
    sub = _team_matches(matches_df, team, before, None)
    if sub.empty:
        return {"days_rest": None, "games_14d": 0}
    last_date = sub.iloc[0]["match_date"]
    days = (before - last_date).days
    cutoff = before - pd.Timedelta(days=14)
    games_14 = int((sub["match_date"] >= cutoff).sum())
    return {"days_rest": days, "games_14d": games_14}


def _h2h(matches_df: pd.DataFrame, home: str, away: str,
         before: pd.Timestamp, n: int = 5) -> dict:
    """Last n meetings between the two teams (either venue)."""
    mask = (
        ((matches_df["home_team"] == home) & (matches_df["away_team"] == away)) |
        ((matches_df["home_team"] == away) & (matches_df["away_team"] == home))
    )
    sub = matches_df[mask & (matches_df["match_date"] < before)] \
        .sort_values("match_date", ascending=False).head(n)
    if sub.empty:
        return {"n": 0, "home_wins": 0, "draws": 0, "away_wins": 0,
                "avg_goals": None, "last": []}

    hw = dw = aw = 0
    total_goals, last = 0, []
    for _, m in sub.iterrows():
        hg, ag = m["home_goals"], m["away_goals"]
        total_goals += hg + ag
        # Normalise to the CURRENT fixture's home team perspective
        if m["home_team"] == home:
            cur_home_goals, cur_away_goals = hg, ag
        else:
            cur_home_goals, cur_away_goals = ag, hg
        if cur_home_goals > cur_away_goals:
            hw += 1
        elif cur_home_goals == cur_away_goals:
            dw += 1
        else:
            aw += 1
        last.append({
            "date": str(m["match_date"].date()),
            "home": m["home_team"], "away": m["away_team"],
            "score": f"{int(hg)}-{int(ag)}",
        })
    return {
        "n": len(sub), "home_wins": hw, "draws": dw, "away_wins": aw,
        "avg_goals": round(total_goals / len(sub), 2), "last": last,
    }


def _xg(xg_df: pd.DataFrame | None, team: str, before: pd.Timestamp,
        n: int = 5) -> dict:
    """Rolling xG for/against over the team's last n matches with xG data."""
    if xg_df is None or xg_df.empty:
        return {"xgf": None, "xga": None, "n": 0}
    mask = (
        ((xg_df["home_team"] == team) | (xg_df["away_team"] == team)) &
        (xg_df["match_date"] < before)
    )
    sub = xg_df[mask].sort_values("match_date", ascending=False).head(n)
    if sub.empty:
        return {"xgf": None, "xga": None, "n": 0}
    xgf, xga, goals_for = [], [], []
    for _, m in sub.iterrows():
        is_home = m["home_team"] == team
        xgf.append(m["xg_h"] if is_home else m["xg_a"])
        xga.append(m["xg_a"] if is_home else m["xg_h"])
        goals_for.append(m["goals_h"] if is_home else m["goals_a"])
    xgf_m = float(np.mean(xgf))
    return {
        "xgf": round(xgf_m, 2),
        "xga": round(float(np.mean(xga)), 2),
        "goals_for": round(float(np.mean(goals_for)), 2),
        "n": len(sub),
    }


# ---------------------------------------------------------------------------
# Rolling stat averages — the user's core method (last N matches)
# ---------------------------------------------------------------------------

# API-Football stat type → friendly key
_STAT_KEYS = {
    "Corner Kicks":     "corners",
    "Shots on Goal":    "sot",       # shots on target
    "Total Shots":      "shots",
    "Yellow Cards":     "cards",
    "Goalkeeper Saves": "saves",
    "expected_goals":   "xg",
}


def _stat_averages(stats_df, matches_df: pd.DataFrame, team: str,
                   before: pd.Timestamp, n: int = 5) -> dict:
    """
    Rolling averages of per-match stats (corners, shots on target, cards,
    saves, xG) — for and against — over the team's last n matches with stats.
    Returns {"n": k, "corners_for":.., "corners_against":.., ...} or {"n": 0}.
    """
    sub = matches_df[
        ((matches_df["home_team"] == team) | (matches_df["away_team"] == team)) &
        (matches_df["match_date"] < before)
    ].sort_values("match_date", ascending=False).head(n)
    if sub.empty:
        return {"n": 0, "has_stats": False}

    recs, gf, ga = [], [], []   # recs: (match_id, team_id, opp_id)
    for _, m in sub.iterrows():
        if m["home_team"] == team:
            recs.append((m["match_id"], m["home_team_id"], m["away_team_id"]))
            gf.append(m["home_goals"]); ga.append(m["away_goals"])
        else:
            recs.append((m["match_id"], m["away_team_id"], m["home_team_id"]))
            gf.append(m["away_goals"]); ga.append(m["home_goals"])

    out = {
        "n": len(sub), "has_stats": False,
        "goals_for": round(float(np.mean(gf)), 2),
        "goals_against": round(float(np.mean(ga)), 2),
    }

    # Corner/shot/card/save/xG stats — only where the league provides them
    if stats_df is None or getattr(stats_df, "empty", True):
        return out
    mids = [r[0] for r in recs]
    s = stats_df[stats_df["match_id"].isin(mids)]
    if s.empty:
        return out

    lut = {(r.match_id, r.team_id, r.type): r.value_num for r in s.itertuples(index=False)}
    accum = {v: {"for": [], "against": []} for v in _STAT_KEYS.values()}
    for mid, tid, oid in recs:
        for stype, key in _STAT_KEYS.items():
            vf = lut.get((mid, tid, stype))
            va = lut.get((mid, oid, stype))
            if vf is not None and pd.notna(vf):
                accum[key]["for"].append(vf)
            if va is not None and pd.notna(va):
                accum[key]["against"].append(va)

    for key, d in accum.items():
        if d["for"] or d["against"]:
            out["has_stats"] = True
        out[f"{key}_for"] = round(float(np.mean(d["for"])), 2) if d["for"] else None
        out[f"{key}_against"] = round(float(np.mean(d["against"])), 2) if d["against"] else None
    return out


def _team_corner_samples(stats_df, matches_df: pd.DataFrame, team: str,
                         before: pd.Timestamp, n: int) -> tuple[list, list, list]:
    """Per-match corners (for, against, total) over the team's last n matches."""
    sub = matches_df[
        ((matches_df["home_team"] == team) | (matches_df["away_team"] == team)) &
        (matches_df["match_date"] < before)
    ].sort_values("match_date", ascending=False).head(n)
    if sub.empty or stats_df is None or getattr(stats_df, "empty", True):
        return [], [], []
    recs = [
        (m["match_id"], m["home_team_id"], m["away_team_id"]) if m["home_team"] == team
        else (m["match_id"], m["away_team_id"], m["home_team_id"])
        for _, m in sub.iterrows()
    ]
    lut = _stats_lut(stats_df, [r[0] for r in recs])
    fors, againsts, totals = [], [], []
    for mid, tid, oid in recs:
        cf = lut.get((mid, tid, "Corner Kicks"))
        ca = lut.get((mid, oid, "Corner Kicks"))
        cf = cf if (cf is not None and pd.notna(cf)) else None
        ca = ca if (ca is not None and pd.notna(ca)) else None
        if cf is not None:
            fors.append(cf)
        if ca is not None:
            againsts.append(ca)
        if cf is not None and ca is not None:
            totals.append(cf + ca)
    return fors, againsts, totals


def corner_analysis(stats_df, matches_df: pd.DataFrame, home: str, away: str,
                    before: pd.Timestamp, n: int = 10) -> dict | None:
    """
    Expected total corners (attack+defense combined, Dixon-Coles style) plus the
    sample's dispersion and size — so the UI can show a GAP vs the line, not a
    fabricated probability. No historical corner lines exist yet → informative
    only, not a validated/backtested edge.

    Returns {proj, std, n_home, n_away, confidence} or None if no corner data.
    """
    hf, ha, ht = _team_corner_samples(stats_df, matches_df, home, before, n)
    af, aa, at = _team_corner_samples(stats_df, matches_df, away, before, n)
    if not (hf and ha and af and aa):
        return None

    proj = (np.mean(hf) + np.mean(aa)) / 2 + (np.mean(af) + np.mean(ha)) / 2
    samples = ht + at
    std = float(np.std(samples)) if len(samples) >= 2 else None
    n_home = min(len(hf), len(ha))
    n_away = min(len(af), len(aa))
    mn = min(n_home, n_away)
    confidence = "sólida" if mn >= 10 else ("chica" if mn < 5 else "media")

    return {
        "proj": round(float(proj), 1),
        "std": round(std, 1) if std is not None else None,
        "n_home": n_home, "n_away": n_away,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Per-match detail — the last N matches row by row (score24-style)
# ---------------------------------------------------------------------------

def _stats_lut(stats_df, match_ids: list) -> dict:
    lut: dict = {}
    if stats_df is not None and not getattr(stats_df, "empty", True):
        s = stats_df[stats_df["match_id"].isin(match_ids)]
        for r in s.itertuples(index=False):
            lut[(r.match_id, r.team_id, r.type)] = r.value_num
    return lut


def team_recent_detail(stats_df, matches_df: pd.DataFrame, team: str,
                       before: pd.Timestamp, n: int = 6) -> list[dict]:
    """The team's last n matches, each as a row with its own stats (not averaged)."""
    sub = matches_df[
        ((matches_df["home_team"] == team) | (matches_df["away_team"] == team)) &
        (matches_df["match_date"] < before)
    ].sort_values("match_date", ascending=False).head(n)
    if sub.empty:
        return []

    lut = _stats_lut(stats_df, sub["match_id"].tolist())

    def _num(v):
        return v if (v is not None and pd.notna(v)) else None

    rows = []
    for _, m in sub.iterrows():
        is_home = m["home_team"] == team
        opp = m["away_team"] if is_home else m["home_team"]
        gf = int(m["home_goals"]) if is_home else int(m["away_goals"])
        ga = int(m["away_goals"]) if is_home else int(m["home_goals"])
        res = "V" if gf > ga else ("E" if gf == ga else "D")
        tid = m["home_team_id"] if is_home else m["away_team_id"]
        oid = m["away_team_id"] if is_home else m["home_team_id"]
        mid = m["match_id"]

        def g(stype, opp_side=False):
            return _num(lut.get((mid, oid if opp_side else tid, stype)))

        rows.append({
            "Fecha": str(m["match_date"].date()),
            "Rival": ("vs " if is_home else "@ ") + str(opp),
            "Marcador": f"{gf}-{ga}",
            "Res": res,
            "Córners": g("Corner Kicks"), "Córners_c": g("Corner Kicks", True),
            "TirosP": g("Shots on Goal"), "TirosP_c": g("Shots on Goal", True),
            "Tarj": g("Yellow Cards"),
            "xG": g("expected_goals"), "xG_c": g("expected_goals", True),
        })
    return rows


def h2h_detail(stats_df, matches_df: pd.DataFrame, home: str, away: str,
               before: pd.Timestamp, n: int = 6) -> list[dict]:
    """Last n meetings between the two teams, each with score + stats."""
    mask = (
        ((matches_df["home_team"] == home) & (matches_df["away_team"] == away)) |
        ((matches_df["home_team"] == away) & (matches_df["away_team"] == home))
    )
    sub = matches_df[mask & (matches_df["match_date"] < before)] \
        .sort_values("match_date", ascending=False).head(n)
    if sub.empty:
        return []

    lut = _stats_lut(stats_df, sub["match_id"].tolist())

    def _num(v):
        return v if (v is not None and pd.notna(v)) else None

    rows = []
    for _, m in sub.iterrows():
        mid = m["match_id"]
        hid, aid = m["home_team_id"], m["away_team_id"]

        def pair(stype):
            h = _num(lut.get((mid, hid, stype)))
            a = _num(lut.get((mid, aid, stype)))
            if h is None and a is None:
                return "—"
            return f"{h if h is not None else '·'}-{a if a is not None else '·'}"

        rows.append({
            "Fecha": str(m["match_date"].date()),
            "Local": m["home_team"], "Visitante": m["away_team"],
            "Marcador": f"{int(m['home_goals'])}-{int(m['away_goals'])}",
            "Córners (L-V)": pair("Corner Kicks"),
            "Tiros P (L-V)": pair("Shots on Goal"),
            "Tarj (L-V)": pair("Yellow Cards"),
        })
    return rows


# ---------------------------------------------------------------------------
# Reading — a short natural-language summary of the model's lean
# ---------------------------------------------------------------------------

_OUTCOME_ES = {"home_win": "victoria local", "draw": "empate", "away_win": "victoria visitante"}


def _reading(probs: dict, market: dict | None, home: str, away: str) -> str:
    lean = max(probs, key=probs.get)
    lean_txt = {"home_win": home, "draw": "empate", "away_win": away}[lean]
    conf = probs[lean]
    strength = "clara" if conf >= 0.55 else ("ligera" if conf >= 0.40 else "muy repartida")
    txt = f"El modelo se inclina por {lean_txt} ({conf:.0%}) — ventaja {strength}."
    if market is not None:
        diff = probs[lean] - market.get(lean, 0.0)
        if diff >= 0.06:
            txt += (f" El modelo ve más probable {_OUTCOME_ES[lean]} que el mercado "
                    f"(+{diff*100:.0f} pts) — punto de interés para revisar.")
        elif diff <= -0.06:
            txt += " El mercado es más optimista que el modelo aquí."
        else:
            txt += " Coincide con el mercado (sin discrepancia notable)."
    return txt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_dossier(
    model,
    matches_df: pd.DataFrame,
    xg_df: pd.DataFrame | None,
    home: str,
    away: str,
    commence_time: pd.Timestamp,
    market_odds: dict | None = None,
    stats_df=None,
) -> dict:
    """
    Build a full pre-match analysis dossier for `home` vs `away`.

    Returns a dict with keys: home, away, date, known (bool), model_probs,
    market_probs, goal_markets, correct_scores, form, xg, h2h, rest, reading.
    If either team is unknown to the model, `known` is False and model
    sections are None (form/h2h/xg from the DB are still provided).
    """
    home = normalize_team(home)
    away = normalize_team(away)

    before = pd.Timestamp(commence_time)
    if before.tzinfo is not None:
        before = before.tz_localize(None)
    matches_df = matches_df.copy()
    matches_df["match_date"] = pd.to_datetime(matches_df["match_date"])
    if matches_df["match_date"].dt.tz is not None:
        matches_df["match_date"] = matches_df["match_date"].dt.tz_localize(None)

    dossier: dict = {
        "home": home, "away": away, "date": str(before.date()),
        "known": False, "model_probs": None, "market_probs": None,
        "goal_markets": None, "correct_scores": None,
        "form": {
            "home": _form(matches_df, home, before, 5, venue="home"),
            "away": _form(matches_df, away, before, 5, venue="away"),
            "home_pts10": _points_last(matches_df, home, before, 10),
            "away_pts10": _points_last(matches_df, away, before, 10),
        },
        "xg": {
            "home": _xg(xg_df, home, before, 5),
            "away": _xg(xg_df, away, before, 5),
        },
        "h2h": _h2h(matches_df, home, away, before, 5),
        "rest": {
            "home": _rest(matches_df, home, before),
            "away": _rest(matches_df, away, before),
        },
        "stats": {
            "home5":  _stat_averages(stats_df, matches_df, home, before, 5),
            "away5":  _stat_averages(stats_df, matches_df, away, before, 5),
            "home10": _stat_averages(stats_df, matches_df, home, before, 10),
            "away10": _stat_averages(stats_df, matches_df, away, before, 10),
        },
        "reading": None,
    }
    dossier["stats"]["corners"] = corner_analysis(
        stats_df, matches_df, home, away, before, n=10
    )

    # Market implied (no-vig) probabilities
    market_probs = None
    if market_odds is not None:
        try:
            market_probs = remove_vig(
                market_odds["home_win"], market_odds["draw"], market_odds["away_win"]
            )
            dossier["market_probs"] = {k: round(v, 4) for k, v in market_probs.items()}
        except (KeyError, ZeroDivisionError, ValueError):
            market_probs = None

    # Model sections (require both teams known)
    teams = getattr(model, "teams", set())
    if home in teams and away in teams:
        dossier["known"] = True
        probs = model.predict_1x2(home, away)
        dossier["model_probs"] = {k: round(v, 4) for k, v in probs.items()}

        poi = getattr(model, "poisson", model)   # EnsembleModel.poisson or a PoissonModel
        ou   = poi.predict_over_under(home, away, 2.5)
        btts = poi.predict_btts(home, away)
        dossier["goal_markets"] = {
            "over25":  round(ou["over"], 4),
            "under25": round(ou["under"], 4),
            "btts_yes": round(btts["yes"], 4),
            "btts_no":  round(btts["no"], 4),
        }
        matrix = poi.predict_score_matrix(home, away)
        flat = [(i, j, float(matrix[i, j]))
                for i in range(matrix.shape[0]) for j in range(matrix.shape[1])]
        flat.sort(key=lambda t: t[2], reverse=True)
        dossier["correct_scores"] = [
            {"score": f"{i}-{j}", "prob": round(p, 4)} for i, j, p in flat[:3]
        ]
        dossier["reading"] = _reading(probs, market_probs, home, away)

    return dossier


# ---------------------------------------------------------------------------
# __main__  — smoke test on one match
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import main as pipeline
    from models.poisson_model import load_matches
    from collectors.understat_collector import load_all_xg

    print("Entrenando modelo + cargando datos...")
    model, teams, label, _ = pipeline.build_live_model()
    matches = load_matches()
    try:
        xg = load_all_xg()
    except Exception:
        xg = None

    HOME, AWAY = "Real Madrid", "Getafe"
    d = build_dossier(model, matches, xg, HOME, AWAY, pd.Timestamp.now(),
                      market_odds={"home_win": 1.42, "draw": 4.80, "away_win": 7.50})
    import json
    print(json.dumps(d, ensure_ascii=False, indent=2, default=str))
