"""
models/real_backtester.py

Walk-forward backtest using REAL Pinnacle odds stored in the odds table.

Architecture:
  _compute_candidates()  — expensive: fits model each week, returns all +EV
                           bet candidates.  Called ONCE per sweep.
  _settle()              — cheap: filters by min_ev, runs Kelly / bankroll
                           simulation.  Called once per threshold in a sweep.
  run()                  — single threshold: calls both.
  sweep(thresholds)      — multi-threshold: calls _compute_candidates once,
                           then _settle for each threshold.
"""

import json
import sys
from datetime import timedelta
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select

sys.path.append(str(Path(__file__).parent.parent))
import config
from database.db import get_session
from database.models import Match, Odds
from models.poisson_model import PoissonModel
from models.ev_calculator import find_value_bets
from models.kelly import recommended_stake
from models.feature_engineering import build_features
from models.xgboost_model import XGBoostModel, prepare_dataset
from models.ensemble_model import EnsembleModel
from collectors.understat_collector import load_all_xg

_OUTCOME_LABELS = {"home_win": "Local", "draw": "Empate", "away_win": "Visitante"}


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    if home_goals == away_goals:
        return "draw"
    return "away_win"


def _brier(probs: dict[str, float], actual: str) -> float:
    return sum(
        (probs[k] - (1.0 if k == actual else 0.0)) ** 2
        for k in ("home_win", "draw", "away_win")
    )


def _max_drawdown(history: list[dict]) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction (0.15 = 15%)."""
    if not history:
        return 0.0
    peak = history[0]["bankroll"]
    max_dd = 0.0
    for entry in history:
        v = entry["bankroll"]
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return round(max_dd, 4)


# ---------------------------------------------------------------------------
# Real backtester
# ---------------------------------------------------------------------------

class RealBacktester:
    """
    Walk-forward backtest del modelo Poisson contra cuotas REALES de Pinnacle.

    Sólo evalúa partidos para los que existe un registro en la tabla odds
    con bookmaker='Pinnacle'.  El modelo se entrena con todos los datos
    históricos anteriores a la semana evaluada (sin look-ahead).
    """
    WARMUP_DAYS = 180

    def __init__(
        self,
        bankroll: float = config.DEFAULT_BANKROLL,
        min_ev: float = config.MIN_EV_THRESHOLD,
        kelly_fraction: float = 0.25,
        max_weekly_exposure: float = 0.20,
    ) -> None:
        self.initial_bankroll    = bankroll
        self.min_ev              = min_ev
        self.kelly_fraction      = kelly_fraction
        self.max_weekly_exposure = max_weekly_exposure

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Single run with self.min_ev."""
        candidates, brier_all, skipped_no_model = self._compute_candidates()
        return self._settle(candidates, brier_all, skipped_no_model, self.min_ev)

    def sweep(self, thresholds: list[float]) -> list[dict]:
        """
        Run staking simulation for multiple EV thresholds.

        Model fitting (the expensive part) is done only ONCE.
        The staking simulation is re-run cheaply for each threshold.
        Returns a list of summary dicts, one per threshold.
        """
        print("  Paso 1/2 — ajustando modelo semana a semana...")
        candidates, brier_all, skipped_no_model = self._compute_candidates()
        print(f"  {len(candidates)} candidatos +EV generados\n")

        print("  Paso 2/2 — simulando bankroll por umbral...")
        rows = []
        for t in thresholds:
            result = self._settle(candidates, brier_all, skipped_no_model, t)
            s = result["summary"]
            rows.append({
                "min_ev":          t,
                "picks":           s["total_picks"],
                "win_rate":        s.get("win_rate", 0.0),
                "roi":             s.get("roi", 0.0),
                "max_drawdown":    s["max_drawdown"],
                "final_bankroll":  s.get("final_bankroll", self.initial_bankroll),
                "total_pnl":       s.get("total_pnl", 0.0),
            })
            print(f"    EV>{t:.0%}  picks={s['total_picks']:<5}  "
                  f"ROI={s.get('roi',0):+.1%}  "
                  f"DD={s['max_drawdown']:.1%}  "
                  f"final={s.get('final_bankroll', self.initial_bankroll):.2f}")
        return rows

    # ------------------------------------------------------------------
    # Core phases
    # ------------------------------------------------------------------

    def _compute_candidates(
        self,
    ) -> tuple[list[dict], list[float], int]:
        """
        Walk-forward model fitting pass.

        For each week in the evaluation window:
          - trains PoissonModel on all data up to week_start (no look-ahead)
          - computes model probabilities for every match with Pinnacle odds
          - records ALL +EV candidates (ev > 0) — EV threshold filtering
            happens later in _settle() so a single pass serves all thresholds

        Returns:
            candidates      — list of bet dicts with pre-computed EV/probs
            brier_all       — list of per-match Brier scores
            skipped_no_model — count of matches skipped (team not in model)
        """
        df = self._load()

        with_odds = df["pinnacle_home"].notna().sum()
        print(f"  {len(df)} partidos en BD  |  {with_odds} con odds Pinnacle "
              f"({with_odds/len(df):.1%})\n")

        warmup_end = df["match_date"].min() + timedelta(days=self.WARMUP_DAYS)
        predict_df = df[
            (df["match_date"] > warmup_end) &
            df["pinnacle_home"].notna()
        ].copy()

        if predict_df.empty:
            raise ValueError(
                "No hay partidos con odds Pinnacle tras el warmup de 6 meses. "
                "Ejecuta collectors/historical_odds_collector.py primero."
            )

        predict_df["week"] = predict_df["match_date"].dt.to_period("W")
        weeks = sorted(predict_df["week"].unique())

        candidates: list[dict] = []
        brier_all: list[float] = []
        skipped_no_model = 0

        for week in weeks:
            week_start = week.start_time
            train_df = df[df["match_date"] < week_start]
            if len(train_df) < 100:
                continue

            model = PoissonModel()
            try:
                model.fit(train_df)
            except ValueError:
                continue

            for _, match in predict_df[predict_df["week"] == week].iterrows():
                home = match["home_team"]
                away = match["away_team"]

                if home not in model.teams or away not in model.teams:
                    skipped_no_model += 1
                    continue

                market_odds = {
                    "home_win": float(match["pinnacle_home"]),
                    "draw":     float(match["pinnacle_draw"]),
                    "away_win": float(match["pinnacle_away"]),
                }

                if not all(
                    config.MIN_ODDS <= v <= config.MAX_ODDS
                    for v in market_odds.values()
                ):
                    continue

                actual = _outcome(int(match["home_goals"]), int(match["away_goals"]))

                try:
                    model_probs = model.predict_1x2(home, away)
                except KeyError:
                    skipped_no_model += 1
                    continue

                brier_all.append(_brier(model_probs, actual))

                # Capture ALL positive-EV candidates (min_ev=0.0)
                for vb in find_value_bets(model_probs, market_odds, min_ev=0.0):
                    candidates.append({
                        "week":       str(week),
                        "date":       str(match["match_date"].date()),
                        "league_id":  int(match["league_id"]) if pd.notna(match.get("league_id")) else None,
                        "home":       home,
                        "away":       away,
                        "actual":     actual,
                        "bet_type":   vb["bet_type"],
                        "odds":       vb["odds"],
                        "model_prob": vb["model_prob"],
                        "ev":         vb["ev"],
                        "edge":       vb["edge"],
                        "overround":  vb["overround"],
                    })

        return candidates, brier_all, skipped_no_model

    # ------------------------------------------------------------------
    # 3-model comparison
    # ------------------------------------------------------------------

    def run_comparison(
        self,
        min_ev: float = 0.03,
        warmup_days: int = 365,
        check_leakage: bool = False,
    ) -> list[dict]:
        """
        Walk-forward backtest for Poisson, XGBoost, and Ensemble at min_ev.

        Uses 4-weekly training windows so XGBoost retraining is feasible.
        Returns one summary dict per model.
        """
        print("  Paso 1/2 — walk-forward Poisson + XGBoost + Ensemble...")
        cands, briers, skips = self._compute_all_candidates(
            warmup_days=warmup_days, check_leakage=check_leakage
        )

        print(f"\n  Candidatos +EV (min_ev=0):")
        for m in ("poisson", "xgboost", "ensemble"):
            print(f"    {m:<10} : {len(cands[m])}")

        print(f"\n  Paso 2/2 — simulando bankroll (min_ev={min_ev:.0%})...")
        rows = []
        for model_name in ("poisson", "xgboost", "ensemble"):
            result = self._settle(
                cands[model_name],
                briers[model_name],
                skips[model_name],
                min_ev,
            )
            s = result["summary"]
            rows.append({
                "model":          model_name.capitalize(),
                "picks":          s["total_picks"],
                "win_rate":       s.get("win_rate", 0.0),
                "roi":            s.get("roi", 0.0),
                "max_drawdown":   s["max_drawdown"],
                "final_bankroll": s.get("final_bankroll", self.initial_bankroll),
                "brier":          s.get("brier_score"),
            })
            print(f"    {model_name:<10} picks={s['total_picks']:<5}  "
                  f"ROI={s.get('roi', 0):+.1%}  "
                  f"DD={s['max_drawdown']:.1%}  "
                  f"final={s.get('final_bankroll', self.initial_bankroll):.2f}")
        return rows

    def _compute_all_candidates(
        self,
        warmup_days: int = 365,
        check_leakage: bool = False,
    ) -> tuple[dict, dict, dict]:
        """
        Unified walk-forward: trains all 3 models per 4-week window.

        Parameters
        ----------
        warmup_days   : minimum history before first eval window starts
        check_leakage : if True, assert max(train_date) < min(eval_date)
                        for every window and print the first 3 date ranges

        Ensemble uses DEFAULT_WEIGHTS (0.6/0.4) — avoids leakage from
        post-hoc weight optimisation.

        Returns
        -------
        cands  : {'poisson': [...], 'xgboost': [...], 'ensemble': [...]}
        briers : {'poisson': [...], ...}    — per-match Brier floats
        skips  : {'poisson': int, ...}      — count skipped (no model)
        """
        df = self._load()   # all finished matches + Pinnacle odds

        # Build feature matrix from the same matches (no separate DB query)
        matches_for_feat = df[[
            "home_team", "away_team", "home_goals", "away_goals",
            "league_id", "season", "match_date",
        ]].copy()

        print("  Cargando xG de Understat (con caché)...", flush=True)
        try:
            xg_df = load_all_xg()
            print(f"    {len(xg_df)} partidos con xG", flush=True)
        except Exception as exc:
            print(f"    Fallo al cargar xG ({exc}) — sin features xG", flush=True)
            xg_df = None

        print("  Construyendo features (orden cronologico)...", flush=True)
        feat_df = build_features(matches_for_feat, xg_df=xg_df)
        X, y    = prepare_dataset(matches_for_feat, feat_df)

        # Attach Pinnacle odds to feat_df — same row order as df
        feat_df = feat_df.copy()
        feat_df["match_id"]      = df["match_id"].values
        feat_df["pinnacle_home"] = df["pinnacle_home"].values
        feat_df["pinnacle_draw"] = df["pinnacle_draw"].values
        feat_df["pinnacle_away"] = df["pinnacle_away"].values

        dates      = pd.to_datetime(feat_df["match_date"])
        with_odds  = feat_df["pinnacle_home"].notna()
        warmup_end = dates.min() + timedelta(days=warmup_days)

        print(f"  {len(df)} partidos  |  {with_odds.sum()} con odds Pinnacle  "
              f"|  warmup={warmup_days}d (hasta {warmup_end.date()})\n")

        # Identify 4-weekly eval windows (only matches after warmup with odds)
        predict_mask = (dates > warmup_end) & with_odds
        predict_df   = feat_df[predict_mask].copy()

        if predict_df.empty:
            raise ValueError(
                f"No hay partidos con odds tras el warmup de {warmup_days} dias."
            )

        # Assign each eval match to a 4-weekly window index
        predict_df["_win"] = predict_df["match_date"].apply(
            lambda d: int((d - warmup_end).days // 28)
        )
        windows   = sorted(predict_df["_win"].unique())
        n_windows = len(windows)

        if check_leakage:
            print("  --- VERIFICACION DE DATA LEAKAGE ---")

        cands  = {"poisson": [], "xgboost": [], "ensemble": []}
        briers = {"poisson": [], "xgboost": [], "ensemble": []}
        skips  = {"poisson": 0,  "xgboost": 0,  "ensemble": 0}

        for w_idx, win in enumerate(windows):
            win_matches = predict_df[predict_df["_win"] == win]
            win_start   = win_matches["match_date"].min()
            train_bool  = (dates < win_start).values

            if train_bool.sum() < 100:
                continue

            # --- Data leakage assertion ---
            # Use strict timestamp comparison (not calendar days) because
            # match_date includes time-of-day — two matches on the same
            # calendar day at different kick-off times are NOT a leak.
            max_train_date = dates[train_bool].max()
            min_eval_date  = pd.Timestamp(win_matches["match_date"].min())
            assert max_train_date < min_eval_date, (
                f"DATA LEAKAGE en ventana {w_idx+1}: "
                f"max_train={max_train_date}  min_eval={min_eval_date}"
            )
            if check_leakage and w_idx < 3:
                gap_sec = (min_eval_date - max_train_date).total_seconds()
                gap_str = (f"{gap_sec/3600:.1f}h"
                           if gap_sec < 86_400 else f"{gap_sec/86_400:.1f}d")
                print(f"  Ventana {w_idx+1}: "
                      f"train hasta {max_train_date}  |  "
                      f"eval desde  {min_eval_date}  |  "
                      f"gap={gap_str}  OK")

            # Progress: first, last, and every 10th window
            if not check_leakage and (
                w_idx == 0 or (w_idx + 1) % 10 == 0 or w_idx == n_windows - 1
            ):
                print(f"  Ventana {w_idx+1:>3}/{n_windows}  "
                      f"train={train_bool.sum()}  eval={len(win_matches)}")

            # --- Poisson ---
            tr_matches = matches_for_feat[train_bool].reset_index(drop=True)
            pm = PoissonModel()
            try:
                pm.fit(tr_matches)
            except Exception:
                continue

            # --- XGBoost ---
            xm = XGBoostModel()
            xm.fit(X[train_bool], y[train_bool])

            # --- Ensemble (Poisson + XGBoost, default weights) ---
            ens = EnsembleModel()
            ens.poisson  = pm
            ens.xgboost  = xm
            ens._fitted  = True
            ens.weights  = dict(EnsembleModel.DEFAULT_WEIGHTS)

            # --- Generate candidates for every eval match ---
            for _, match in win_matches.iterrows():
                if pd.isna(match["pinnacle_home"]):
                    continue

                home = match["home_team"]
                away = match["away_team"]
                feat_row = X.loc[[match.name]]

                market_odds = {
                    "home_win": float(match["pinnacle_home"]),
                    "draw":     float(match["pinnacle_draw"]),
                    "away_win": float(match["pinnacle_away"]),
                }
                if not all(
                    config.MIN_ODDS <= v <= config.MAX_ODDS
                    for v in market_odds.values()
                ):
                    continue

                actual   = _outcome(int(match["home_goals"]), int(match["away_goals"]))
                week_str = str(pd.Period(match["match_date"], freq="W"))
                base = {
                    "week":      week_str,
                    "date":      str(match["match_date"].date()),
                    "league_id": int(match["league_id"]) if pd.notna(match.get("league_id")) else None,
                    "home":      home,
                    "away":      away,
                    "actual":    actual,
                }

                # Poisson
                try:
                    poi_p = pm.predict_1x2(home, away)
                    briers["poisson"].append(_brier(poi_p, actual))
                    for vb in find_value_bets(poi_p, market_odds, min_ev=0.0):
                        cands["poisson"].append({**base,
                            "bet_type":   vb["bet_type"],
                            "odds":       vb["odds"],
                            "model_prob": vb["model_prob"],
                            "ev":         vb["ev"],
                            "edge":       vb["edge"],
                            "overround":  vb["overround"],
                        })
                except (KeyError, RuntimeError):
                    skips["poisson"] += 1

                # XGBoost
                xgb_arr = xm.predict_proba(feat_row)[0]
                xgb_p = {
                    "home_win": float(xgb_arr[0]),
                    "draw":     float(xgb_arr[1]),
                    "away_win": float(xgb_arr[2]),
                }
                briers["xgboost"].append(_brier(xgb_p, actual))
                for vb in find_value_bets(xgb_p, market_odds, min_ev=0.0):
                    cands["xgboost"].append({**base,
                        "bet_type":   vb["bet_type"],
                        "odds":       vb["odds"],
                        "model_prob": vb["model_prob"],
                        "ev":         vb["ev"],
                        "edge":       vb["edge"],
                        "overround":  vb["overround"],
                    })

                # Ensemble
                try:
                    ens_p = ens.predict_proba(home, away, feat_row)
                    briers["ensemble"].append(_brier(ens_p, actual))
                    for vb in find_value_bets(ens_p, market_odds, min_ev=0.0):
                        cands["ensemble"].append({**base,
                            "bet_type":   vb["bet_type"],
                            "odds":       vb["odds"],
                            "model_prob": vb["model_prob"],
                            "ev":         vb["ev"],
                            "edge":       vb["edge"],
                            "overround":  vb["overround"],
                        })
                except (KeyError, RuntimeError):
                    skips["ensemble"] += 1

        return cands, briers, skips

    def _settle(
        self,
        candidates: list[dict],
        brier_all: list[float],
        skipped_no_model: int,
        min_ev: float,
    ) -> dict:
        """
        Staking simulation on pre-computed candidates.

        Filters candidates by ev >= min_ev, applies fractional Kelly per week
        (with weekly exposure cap), and accumulates P&L.
        """
        bankroll     = self.initial_bankroll
        picks: list[dict]            = []
        bankroll_history: list[dict] = []
        skipped_no_ev = 0

        # Group by week (candidates are already in chronological order)
        sorted_cands = sorted(candidates, key=lambda c: c["week"])
        for week_str, week_iter in groupby(sorted_cands, key=lambda c: c["week"]):
            week_list = list(week_iter)
            week_bankroll = bankroll
            week_buffer: list[dict] = []

            for c in week_list:
                if c["ev"] < min_ev:
                    skipped_no_ev += 1
                    continue

                stake_info = recommended_stake(
                    c["model_prob"],
                    c["odds"],
                    week_bankroll,
                    fraction=self.kelly_fraction,
                )
                if stake_info["stake"] == 0:
                    continue

                stake = stake_info["stake"]
                won   = c["bet_type"] == c["actual"]
                pnl   = round(stake * (c["odds"] - 1) if won else -stake, 2)

                week_buffer.append({
                    **{k: c[k] for k in ("week", "date", "league_id", "home", "away",
                                         "bet_type", "odds", "model_prob", "ev")},
                    "stake": stake,
                    "won":   bool(won),
                    "pnl":   pnl,
                })

            # Scale stakes if total exposure exceeds the weekly cap
            total_desired = sum(p["stake"] for p in week_buffer)
            max_stake     = week_bankroll * self.max_weekly_exposure
            if total_desired > max_stake and total_desired > 0:
                scale = max_stake / total_desired
                for p in week_buffer:
                    p["stake"] = round(p["stake"] * scale, 2)
                    p["pnl"]   = round(
                        p["stake"] * (p["odds"] - 1) if p["won"] else -p["stake"], 2
                    )

            week_pnl = sum(p["pnl"] for p in week_buffer)
            bankroll = round(bankroll + week_pnl, 2)

            for p in week_buffer:
                p["bankroll"] = bankroll
                picks.append(p)

            bankroll_history.append({"week": week_str, "bankroll": bankroll})

        return self._compile(
            picks, brier_all, bankroll_history, skipped_no_model, skipped_no_ev
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> pd.DataFrame:
        with get_session() as session:
            pinnacle: dict[int, dict] = {
                o.match_id: {
                    "home_win": o.home_win,
                    "draw":     o.draw,
                    "away_win": o.away_win,
                }
                for o in session.execute(
                    select(Odds).where(Odds.bookmaker == "Pinnacle")
                ).scalars().all()
            }

            records = []
            for m in session.execute(
                select(Match).where(
                    Match.status == "finished",
                    Match.home_goals.is_not(None),
                    Match.away_goals.is_not(None),
                )
            ).scalars().all():
                o = pinnacle.get(m.id)
                records.append({
                    "match_id":      m.id,
                    "home_team":     m.home_team_name,
                    "away_team":     m.away_team_name,
                    "home_goals":    m.home_goals,
                    "away_goals":    m.away_goals,
                    "league_id":     m.league_id,
                    "season":        m.season,
                    "match_date":    m.match_date,
                    "pinnacle_home": o["home_win"] if o else None,
                    "pinnacle_draw": o["draw"]     if o else None,
                    "pinnacle_away": o["away_win"] if o else None,
                })

        df = pd.DataFrame(records)
        df["match_date"] = pd.to_datetime(df["match_date"])
        return df.sort_values("match_date").reset_index(drop=True)

    def _compile(
        self,
        picks: list[dict],
        brier_all: list[float],
        bankroll_history: list[dict],
        skipped_no_model: int,
        skipped_no_ev: int,
    ) -> dict:
        brier_mean = round(float(np.mean(brier_all)), 4) if brier_all else None
        max_dd     = _max_drawdown(bankroll_history)
        final_br   = bankroll_history[-1]["bankroll"] if bankroll_history else self.initial_bankroll

        if not picks:
            return {
                "summary": {
                    "initial_bankroll": self.initial_bankroll,
                    "final_bankroll":   final_br,
                    "total_picks":      0,
                    "win_rate":         0.0,
                    "total_staked":     0.0,
                    "total_pnl":        0.0,
                    "roi":              0.0,
                    "brier_score":      brier_mean,
                    "max_drawdown":     max_dd,
                    "skipped_no_model": skipped_no_model,
                    "skipped_no_ev":    skipped_no_ev,
                },
                "pnl_by_league":    {},
                "pnl_by_bet_type":  {},
                "bankroll_history": bankroll_history,
                "picks":            [],
            }

        pdf = pd.DataFrame(picks)
        total_staked = float(pdf["stake"].sum())
        total_pnl    = float(pdf["pnl"].sum())
        n_won        = int(pdf["won"].sum())
        n_weeks      = pdf["week"].nunique()

        return {
            "summary": {
                "initial_bankroll": self.initial_bankroll,
                "final_bankroll":   final_br,
                "total_picks":      len(picks),
                "picks_per_week":   round(len(picks) / max(n_weeks, 1), 2),
                "win_rate":         round(n_won / len(picks), 4),
                "total_staked":     round(total_staked, 2),
                "total_pnl":        round(total_pnl, 2),
                "roi":              round(total_pnl / total_staked, 4) if total_staked > 0 else 0.0,
                "brier_score":      brier_mean,
                "max_drawdown":     max_dd,
                "skipped_no_model": skipped_no_model,
                "skipped_no_ev":    skipped_no_ev,
            },
            "pnl_by_league":    pdf.groupby("league_id")["pnl"].sum().round(2).to_dict(),
            "pnl_by_bet_type":  pdf.groupby("bet_type")["pnl"].sum().round(2).to_dict(),
            "bankroll_history": bankroll_history,
            "picks":            picks,
        }


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    MIN_EV     = 0.03
    THRESHOLDS = [0.03, 0.05, 0.07, 0.10, 0.15]
    SEP        = "=" * 74

    bt = RealBacktester()

    # ----------------------------------------------------------------
    # Part 1 — Poisson EV threshold sweep (existing)
    # ----------------------------------------------------------------
    print(f"{SEP}")
    print("  PARTE 1 — POISSON: sweep de umbral MIN_EV (ventanas semanales)")
    print(f"{SEP}\n")

    rows_sweep = bt.sweep(THRESHOLDS)

    print(f"\n{SEP}")
    print("  TABLA COMPARATIVA — Poisson, umbral MIN_EV variable")
    print(SEP)
    print(f"  {'Min EV':>6}  {'Picks':>6}  {'Win%':>6}  {'ROI':>7}  "
          f"{'Max DD':>7}  {'P&L':>8}  {'Bankroll Final':>14}")
    print("  " + "-" * 66)
    for r in rows_sweep:
        wr  = f"{r['win_rate']:.1%}" if r["picks"] else "—"
        roi = f"{r['roi']:+.1%}"     if r["picks"] else "—"
        dd  = f"{r['max_drawdown']:.1%}"
        pnl = f"{r['total_pnl']:+.2f}"
        br  = f"{r['final_bankroll']:.2f}"
        print(f"  {r['min_ev']:>5.0%}   {r['picks']:>6}  {wr:>6}  {roi:>7}  "
              f"{dd:>7}  {pnl:>8}  {br:>14}")
    print(SEP)

    # Save best Poisson threshold result
    eligible = [r for r in rows_sweep if r["picks"] >= 50]
    if eligible:
        best_t = max(eligible, key=lambda r: r["roi"])["min_ev"]
        print(f"\n  Mejor ROI Poisson con >=50 picks: EV>{best_t:.0%}")
        bt2 = RealBacktester(min_ev=best_t)
        candidates, brier_all, skipped = bt2._compute_candidates()
        best_result = bt2._settle(candidates, brier_all, skipped, best_t)
        out_path = config.DATA_DIR / "real_backtest_results.json"
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(best_result, f, ensure_ascii=False, indent=2, default=str)
        print(f"  Guardado en: {out_path}")

    # ----------------------------------------------------------------
    # Part 2 — 3-model comparison at MIN_EV = 3% (4-weekly windows)
    # ----------------------------------------------------------------
    print(f"\n{SEP}")
    print(f"  PARTE 2 — COMPARATIVA 3 MODELOS: Poisson | XGBoost | Ensemble")
    print(f"  MIN_EV = {MIN_EV:.0%}  |  bankroll = {bt.initial_bankroll:.0f}  "
          f"|  Kelly fraccion = {bt.kelly_fraction:.0%}  "
          f"|  ventanas 4-semanales")
    print(f"{SEP}\n")

    rows_cmp = bt.run_comparison(min_ev=MIN_EV)

    print(f"\n{SEP}")
    print("  TABLA COMPARATIVA — 3 MODELOS")
    print(SEP)
    print(f"  {'Modelo':<12}  {'Picks':>6}  {'Win%':>6}  {'ROI':>7}  "
          f"{'Max DD':>7}  {'Brier':>7}  {'Bankroll Final':>14}")
    print("  " + "-" * 66)
    for r in rows_cmp:
        wr    = f"{r['win_rate']:.1%}" if r["picks"] else "—"
        roi   = f"{r['roi']:+.1%}"     if r["picks"] else "—"
        dd    = f"{r['max_drawdown']:.1%}"
        br    = f"{r['brier']:.4f}"    if r["brier"] else "—"
        final = f"{r['final_bankroll']:.2f}"
        print(f"  {r['model']:<12}  {r['picks']:>6}  {wr:>6}  {roi:>7}  "
              f"{dd:>7}  {br:>7}  {final:>14}")
    print(SEP)

    # Highlight winner by ROI (models with at least 10 picks)
    eligible_cmp = [r for r in rows_cmp if r["picks"] >= 10]
    if eligible_cmp:
        best_roi   = max(eligible_cmp, key=lambda r: r["roi"])
        best_brier = min(eligible_cmp, key=lambda r: r["brier"] or 99)
        print(f"\n  Mejor ROI   : {best_roi['model']}  ({best_roi['roi']:+.1%})")
        print(f"  Mejor Brier : {best_brier['model']}  ({best_brier['brier']:.4f})")
    print(SEP)
