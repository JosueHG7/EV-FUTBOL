"""
models/feature_engineering.py

Builds a rich feature matrix for every match using ONLY data available
strictly before each match date — zero look-ahead / data leakage.

Matches are processed in chronological order.  Running histories are
updated AFTER each row's features are computed so that a match never
sees its own result.

Feature groups
--------------
  1. Recent form     — rolling windows of 5 and 10 matches
  2. Venue stats     — home / away averages (all-time)
  3. Head-to-Head    — last 5 encounters between the two teams
  4. Fatigue         — days rest and fixture congestion
  5. Poisson model   — running (incremental) attack/defense estimates
  6. xG              — expected goals rolling windows and overperformance
                       (optional: pass xg_df from understat_collector)
"""

import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson as sp_poisson

sys.path.append(str(Path(__file__).parent.parent))
from models.poisson_model import load_matches

_MAX_GOALS = 10


# ---------------------------------------------------------------------------
# Running Poisson model — O(1) per match, strictly no look-ahead
# ---------------------------------------------------------------------------

class _RunningPoisson:
    """
    Maintains incremental running sums so Poisson parameters can be
    derived in O(1) after each new match is added.

    Does NOT apply Dixon-Coles or temporal decay (those require the full
    optimisation loop).  The simple formulation is sufficient for features.
    """

    def __init__(self) -> None:
        self._total_home: float = 0.0
        self._total_away: float = 0.0
        self._n: int = 0
        self._scored:   defaultdict[str, float] = defaultdict(float)
        self._conceded: defaultdict[str, float] = defaultdict(float)
        self._games:    defaultdict[str, int]   = defaultdict(int)

    def add(self, home: str, away: str, hg: int, ag: int) -> None:
        self._total_home += hg
        self._total_away += ag
        self._n += 1
        self._scored[home]   += hg;  self._conceded[home] += ag
        self._scored[away]   += ag;  self._conceded[away] += hg
        self._games[home]    += 1;   self._games[away]    += 1

    def predict(self, home: str, away: str) -> tuple | None:
        """
        Returns (p_home_win, p_draw, p_away_win, lambda_home, lambda_away)
        or None if not enough data.  Minimum 20 matches observed.
        """
        if (self._n < 20
                or home not in self._games
                or away not in self._games):
            return None

        avg = (self._total_home + self._total_away) / (2 * self._n)
        if avg == 0:
            return None
        home_adv = (self._total_home / self._n) / avg

        def _atk_def(team: str) -> tuple[float, float]:
            g = self._games[team]
            return (self._scored[team]   / g) / avg, \
                   (self._conceded[team] / g) / avg

        h_atk, h_def = _atk_def(home)
        a_atk, a_def = _atk_def(away)

        lh = h_atk * a_def * home_adv * avg
        la = a_atk * h_def * avg

        h_pmf = sp_poisson.pmf(np.arange(_MAX_GOALS), lh)
        a_pmf = sp_poisson.pmf(np.arange(_MAX_GOALS), la)
        mat   = np.outer(h_pmf, a_pmf)

        p_h = float(np.sum(np.tril(mat, -1)))
        p_d = float(np.sum(np.diag(mat)))
        p_a = float(np.sum(np.triu(mat,  1)))
        return round(p_h, 4), round(p_d, 4), round(p_a, 4), \
               round(lh, 4), round(la, 4)


# ---------------------------------------------------------------------------
# Feature helpers — each takes pre-filtered history lists
# ---------------------------------------------------------------------------

def _form(history: list, n: int, prefix: str) -> dict:
    """
    Recent-form features for the last n matches.
    History entries: (date, goals_scored, goals_conceded, points_earned).
    """
    last = history[-n:] if len(history) >= n else history
    if not last:
        return {
            f"{prefix}_form_goals_scored_{n}":   np.nan,
            f"{prefix}_form_goals_conceded_{n}": np.nan,
            f"{prefix}_form_wins_{n}":           np.nan,
            f"{prefix}_form_points_{n}":         np.nan,
        }
    gs  = float(np.mean([e[1] for e in last]))
    gc  = float(np.mean([e[2] for e in last]))
    win = float(np.mean([1.0 if e[3] == 3 else 0.0 for e in last]))
    pts = float(sum(e[3] for e in last))
    return {
        f"{prefix}_form_goals_scored_{n}":   round(gs,  3),
        f"{prefix}_form_goals_conceded_{n}": round(gc,  3),
        f"{prefix}_form_wins_{n}":           round(win, 3),
        f"{prefix}_form_points_{n}":         pts,
    }


def _venue_stats(history: list, prefix: str) -> dict:
    """
    All-time venue-specific averages (home-only or away-only history).
    History entries: (date, goals_scored, goals_conceded).
    """
    if not history:
        return {
            f"{prefix}_goals_scored_avg":   np.nan,
            f"{prefix}_goals_conceded_avg": np.nan,
        }
    gs = float(np.mean([e[1] for e in history]))
    gc = float(np.mean([e[2] for e in history]))
    return {
        f"{prefix}_goals_scored_avg":   round(gs, 3),
        f"{prefix}_goals_conceded_avg": round(gc, 3),
    }


def _h2h(encounters: list, current_home: str) -> dict:
    """
    H2H features from last 5 encounters.
    Wins/losses are relative to who is the home team in the CURRENT match.

    Encounter entries: (date, enc_home_team, enc_away_team, hg, ag).
    """
    last = encounters[-5:] if len(encounters) >= 5 else encounters
    if not last:
        return {
            "h2h_home_wins":  np.nan,
            "h2h_draws":      np.nan,
            "h2h_away_wins":  np.nan,
            "h2h_avg_goals":  np.nan,
        }

    hw = dd = aw = 0
    total_g = 0
    for _, enc_home, _, hg, ag in last:
        total_g += hg + ag
        if hg > ag:               # enc_home team won
            if enc_home == current_home: hw += 1
            else:                        aw += 1
        elif hg < ag:             # enc_away team won
            if enc_home == current_home: aw += 1
            else:                        hw += 1
        else:
            dd += 1

    n = len(last)
    return {
        "h2h_home_wins":  round(hw     / n, 3),
        "h2h_draws":      round(dd     / n, 3),
        "h2h_away_wins":  round(aw     / n, 3),
        "h2h_avg_goals":  round(total_g / n, 3),
    }


def _xg_roll(history: list, prefix: str) -> dict:
    """
    xG features for the last 5 matches with available xG data.

    History entries: (date, xg_scored, xg_conceded, goals_scored).
    prefix: 'h' for home team, 'a' for away team.
    """
    nan_result = {
        f"xg_{prefix}_roll5":          np.nan,
        f"xg_conceded_{prefix}_roll5": np.nan,
        f"xg_diff_{prefix}":           np.nan,
        f"xg_overperf_{prefix}":       np.nan,
    }
    if not history:
        return nan_result

    last5       = history[-5:]
    xg_scored   = float(np.mean([e[1] for e in last5]))
    xg_conceded = float(np.mean([e[2] for e in last5]))
    xg_diff     = xg_scored - xg_conceded

    total_xg    = sum(e[1] for e in history)
    total_goals = sum(e[3] for e in history)
    if total_xg > 0:
        overperf = round(float(np.clip(total_goals / total_xg, 0.3, 3.0)), 4)
    else:
        overperf = np.nan

    return {
        f"xg_{prefix}_roll5":          round(xg_scored,   4),
        f"xg_conceded_{prefix}_roll5": round(xg_conceded, 4),
        f"xg_diff_{prefix}":           round(xg_diff,     4),
        f"xg_overperf_{prefix}":       overperf,
    }


def _fatigue(all_history: list, match_date: pd.Timestamp, prefix: str) -> dict:
    """
    Fatigue / congestion features.
    History entries: (date, goals_scored, goals_conceded, points).
    """
    if not all_history:
        return {
            f"{prefix}_days_since_last":    np.nan,
            f"{prefix}_games_last_14days":  0,
        }
    last_date  = all_history[-1][0]
    days_since = (match_date - last_date).days
    cutoff     = match_date - timedelta(days=14)
    games_14   = sum(1 for e in all_history if e[0] >= cutoff)
    return {
        f"{prefix}_days_since_last":    days_since,
        f"{prefix}_games_last_14days":  games_14,
    }


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def build_features(
    matches_df: pd.DataFrame,
    xg_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build feature matrix for every match in matches_df.

    Parameters
    ----------
    matches_df : DataFrame with columns home_team, away_team, home_goals,
                 away_goals, match_date, league_id, season.
                 (As returned by load_matches().)
    xg_df      : optional DataFrame from understat_collector.load_all_xg()
                 with columns match_date, home_team, away_team, xg_h, xg_a,
                 goals_h, goals_a.  When provided, adds 8 xG feature columns.
                 When None (default), those columns are omitted entirely —
                 total features stays at 33, backward-compatible.

    Returns
    -------
    DataFrame with original match columns, a 'result' target column
    (H/D/A), and all feature columns — one row per match, chronologically
    sorted.  Early rows have NaN for form/H2H features (not enough history).
    """
    df = matches_df.copy()
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values("match_date").reset_index(drop=True)

    # Build xG lookup: (date_str, home_team, away_team) → (xg_h, xg_a)
    # Using date-only key to handle minor timezone differences between sources.
    _xg_lookup: dict[tuple, tuple] = {}
    if xg_df is not None and not xg_df.empty:
        for _, r in xg_df.iterrows():
            key = (
                str(pd.to_datetime(r["match_date"]).date()),
                r["home_team"],
                r["away_team"],
            )
            _xg_lookup[key] = (float(r["xg_h"]), float(r["xg_a"]))

    # Running state — updated AFTER each row's features are computed
    # team_all  : team → [(date, scored, conceded, points)]
    # team_home : team → [(date, scored, conceded)]   — home matches only
    # team_away : team → [(date, scored, conceded)]   — away matches only
    # h2h_store : (sorted_pair) → [(date, enc_home, enc_away, hg, ag)]
    # team_xg   : team → [(date, xg_scored, xg_conceded, goals_scored)]
    team_all   : defaultdict[str, list] = defaultdict(list)
    team_home  : defaultdict[str, list] = defaultdict(list)
    team_away  : defaultdict[str, list] = defaultdict(list)
    h2h_store  : defaultdict[tuple, list] = defaultdict(list)
    team_xg    : defaultdict[str, list] = defaultdict(list)
    running_poi = _RunningPoisson()

    feature_rows: list[dict] = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        date = row["match_date"]
        hg   = int(row["home_goals"])
        ag   = int(row["away_goals"])
        h2h_key = (min(home, away), max(home, away))

        feat: dict = {}

        # ---- Group 1: Recent form ----------------------------------------
        feat.update(_form(team_all[home], 5,  "home"))
        feat.update(_form(team_all[home], 10, "home"))
        feat.update(_form(team_all[away], 5,  "away"))
        feat.update(_form(team_all[away], 10, "away"))

        # ---- Group 2: Venue stats ----------------------------------------
        feat.update(_venue_stats(team_home[home], "home"))
        feat.update(_venue_stats(team_away[away], "away"))

        # ---- Group 3: Head-to-head ---------------------------------------
        feat.update(_h2h(h2h_store[h2h_key], home))

        # ---- Group 4: Fatigue --------------------------------------------
        feat.update(_fatigue(team_all[home], date, "home"))
        feat.update(_fatigue(team_all[away], date, "away"))

        # ---- Group 5: Poisson strengths ----------------------------------
        poi = running_poi.predict(home, away)
        if poi is not None:
            feat["poisson_home_win"], feat["poisson_draw"], \
            feat["poisson_away_win"], feat["lambda_home"], \
            feat["lambda_away"] = poi
        else:
            feat["poisson_home_win"] = np.nan
            feat["poisson_draw"]     = np.nan
            feat["poisson_away_win"] = np.nan
            feat["lambda_home"]      = np.nan
            feat["lambda_away"]      = np.nan

        # ---- Group 6: xG features (only when xg_df was provided) --------
        if _xg_lookup:
            feat.update(_xg_roll(team_xg[home], "h"))
            feat.update(_xg_roll(team_xg[away], "a"))

        feature_rows.append(feat)

        # ---- Update running state AFTER features are computed ------------
        home_pts = 3 if hg > ag else (1 if hg == ag else 0)
        away_pts = 3 if ag > hg else (1 if hg == ag else 0)

        team_all[home].append((date, hg, ag, home_pts))
        team_all[away].append((date, ag, hg, away_pts))
        team_home[home].append((date, hg, ag))
        team_away[away].append((date, ag, hg))
        h2h_store[h2h_key].append((date, home, away, hg, ag))
        running_poi.add(home, away, hg, ag)

        # xG history: update only when Understat has data for this match
        xg_pair = _xg_lookup.get((str(date.date()), home, away))
        if xg_pair is not None:
            xg_h_val, xg_a_val = xg_pair
            team_xg[home].append((date, xg_h_val, xg_a_val, hg))
            team_xg[away].append((date, xg_a_val, xg_h_val, ag))

    # Combine original columns + target + features
    result_col = df.apply(
        lambda r: "H" if r["home_goals"] > r["away_goals"]
                  else ("D" if r["home_goals"] == r["away_goals"] else "A"),
        axis=1,
    )
    base = df[["home_team", "away_team", "home_goals", "away_goals",
               "match_date", "league_id", "season"]].copy()
    base["result"] = result_col

    return pd.concat([base.reset_index(drop=True),
                      pd.DataFrame(feature_rows)], axis=1)


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from collectors.understat_collector import load_all_xg

    print("Cargando partidos de la BD...")
    df = load_matches()
    print(f"  {len(df)} partidos | {df['league_id'].nunique()} ligas\n")

    print("Cargando xG de Understat (con caché 48h)...")
    try:
        xg_df = load_all_xg()
        print(f"  {len(xg_df)} partidos con xG\n")
    except Exception as exc:
        print(f"  Fallo al cargar xG ({exc}) — continuando sin xG\n")
        xg_df = None

    print("Construyendo features (procesando en orden cronologico)...")
    feat_df = build_features(df, xg_df=xg_df)

    # Exclude identifier / target columns from feature count
    id_cols  = {"home_team", "away_team", "home_goals", "away_goals",
                "match_date", "league_id", "season", "result"}
    feat_cols = [c for c in feat_df.columns if c not in id_cols]

    print(f"\n  Partidos             : {len(feat_df)}")
    print(f"  Features generadas   : {len(feat_cols)}\n")

    # Group summary
    groups = {
        "Forma reciente (Grupo 1)": [c for c in feat_cols if "form_"   in c],
        "Local/Visit (Grupo 2)":    [c for c in feat_cols if "_goals_scored_avg" in c or "_goals_conceded_avg" in c],
        "Head-to-Head (Grupo 3)":   [c for c in feat_cols if c.startswith("h2h_")],
        "Fatiga (Grupo 4)":         [c for c in feat_cols if "days_since" in c or "games_last" in c],
        "Poisson (Grupo 5)":        [c for c in feat_cols if "poisson_" in c or "lambda_" in c],
        "xG Understat (Grupo 6)":   [c for c in feat_cols if c.startswith("xg_")],
    }
    for grp, cols in groups.items():
        print(f"  {grp:<30} {len(cols):>2} features")

    # First 3 rows (transposed for readability)
    print("\nPrimeras 3 filas (features — transpuesta):")
    sample = feat_df[feat_cols].head(3).T
    sample.columns = ["partido_0", "partido_1", "partido_2"]
    print(sample.to_string())

    # Null percentages
    null_pct = (feat_df[feat_cols].isnull().mean() * 100).sort_values(ascending=False)
    with_nulls  = null_pct[null_pct > 0]
    zero_nulls  = (null_pct == 0).sum()

    print(f"\n% valores nulos por feature (solo las que tienen > 0%):")
    for col, pct in with_nulls.items():
        bar = "#" * int(pct / 2)
        print(f"  {col:<40} {pct:>5.1f}%  {bar}")
    print(f"\n  Features sin nulos : {zero_nulls}")
    print(f"  Features con nulos : {len(with_nulls)}")
    print(f"  Max nulos          : {null_pct.max():.1f}% ({null_pct.idxmax()})")
