"""
models/validate_cross_league.py — PROTOTYPE VALIDATION, not wired into
refresh.py, main.py, or the dashboard.

Walk-forward comparison of the production PoissonModel (independent
per-team averaging) against JointPoissonModel (joint opponent-aware MLE —
see models/poisson_joint.py) on REAL outcomes, split into:
  - "cross"    matches in the 5 newly-added competitions (Champions/Europa/
               Conference League, Libertadores, Sudamericana) — exactly the
               matches the production model can't compare across leagues.
  - "domestic" everything else — a control group: the joint model should be
               roughly as good here, not worse.

No look-ahead: each monthly fold trains both models only on matches strictly
before that month, then scores that month's matches. Same discipline as
models/real_backtester.py (the walk-forward that replaced the buggy
positional join — see PROJECT_STATUS.md "El bug que desmontó el edge").

Scoring: log-loss and Brier score of the 1X2 distribution against the real
result (proper scoring rules — lower is better), plus plain directional
accuracy. Not an EV/ROI backtest: we don't have historical odds for these
competitions' past matches, and accuracy against the outcome is what this
question is actually about (see PROJECT_STATUS.md pendiente #2).

Usage: python -m models.validate_cross_league
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from models.poisson_model import PoissonModel, load_matches
from models.poisson_joint import JointPoissonModel

NEW_COMPS = {2, 3, 848, 13, 11}
WARMUP_DAYS = 180
MIN_TRAIN_MATCHES = 500


def _outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    if home_goals == away_goals:
        return "draw"
    return "away_win"


def _brier(probs: dict, actual: str) -> float:
    return sum((probs[k] - (1.0 if k == actual else 0.0)) ** 2
               for k in ("home_win", "draw", "away_win"))


def _log_loss(probs: dict, actual: str) -> float:
    return -float(np.log(max(probs[actual], 1e-10)))


def run(reg: float = 0.01) -> pd.DataFrame:
    print("Cargando partidos...")
    df = load_matches()
    df = df.sort_values("match_date", kind="stable").reset_index(drop=True)
    print(f"  {len(df)} partidos · {df['match_date'].min().date()} -> "
          f"{df['match_date'].max().date()}\n")

    warmup_end = df["match_date"].min() + pd.Timedelta(days=WARMUP_DAYS)
    eval_df = df[df["match_date"] > warmup_end].copy()
    eval_df["month"] = eval_df["match_date"].dt.to_period("M")
    months = sorted(eval_df["month"].unique())
    print(f"{len(months)} meses a evaluar (desde {months[0]} hasta {months[-1]})\n")

    rows: list[dict] = []
    for i, month in enumerate(months, 1):
        month_start = month.start_time
        train_df = df[df["match_date"] < month_start]
        if len(train_df) < MIN_TRAIN_MATCHES:
            continue

        t0 = time.time()
        base = PoissonModel()
        base.fit(train_df, decay_factor=0.98)
        joint = JointPoissonModel(reg=reg)
        joint.fit(train_df, decay_factor=0.98)
        fit_s = time.time() - t0

        test = eval_df[eval_df["month"] == month]
        n_scored = 0
        for _, m in test.iterrows():
            home, away = m["home_team"], m["away_team"]
            if home not in base.teams or away not in base.teams:
                continue
            actual = _outcome(int(m["home_goals"]), int(m["away_goals"]))
            bucket = "cross" if m["league_id"] in NEW_COMPS else "domestic"

            pb = base.predict_1x2(home, away)
            pj = joint.predict_1x2(home, away)
            n_scored += 1
            rows.append({
                "month": str(month), "bucket": bucket,
                "model": "base", "log_loss": _log_loss(pb, actual),
                "brier": _brier(pb, actual),
                "correct": max(pb, key=pb.get) == actual,
            })
            rows.append({
                "month": str(month), "bucket": bucket,
                "model": "joint", "log_loss": _log_loss(pj, actual),
                "brier": _brier(pj, actual),
                "correct": max(pj, key=pj.get) == actual,
            })
        print(f"  [{i}/{len(months)}] {month}  train={len(train_df):>6}  "
              f"scored={n_scored:>4}  fit={fit_s:.1f}s")

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("RESULTADOS (menor log_loss/brier = mejor; correct = acierto direccional)")
    print("=" * 72)
    agg = (results.groupby(["bucket", "model"])
           .agg(n=("correct", "size"),
                log_loss=("log_loss", "mean"),
                brier=("brier", "mean"),
                accuracy=("correct", "mean"))
           .reset_index())
    print(agg.to_string(index=False))

    print("\nPor mes, solo bucket 'cross' (competiciones nuevas):")
    cross = results[results["bucket"] == "cross"]
    by_month = (cross.groupby(["month", "model"])
               .agg(n=("correct", "size"), log_loss=("log_loss", "mean"))
               .reset_index()
               .pivot(index="month", columns="model", values=["n", "log_loss"]))
    print(by_month.to_string())


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    results = run()
    out_path = Path(__file__).parent.parent / "data" / "validate_cross_league_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nGuardado: {out_path}")
    summarize(results)
