import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
import config
from models.poisson_model import PoissonModel, load_matches
from models.ev_calculator import find_value_bets
from models.kelly import recommended_stake

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
    """Multi-class Brier score for one match (lower = better, perfect = 0)."""
    return sum(
        (probs[k] - (1.0 if k == actual else 0.0)) ** 2
        for k in ("home_win", "draw", "away_win")
    )


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class WalkForwardBacktester:
    """
    Walk-forward backtest del modelo Poisson contra odds de mercado simuladas.

    El "bookmaker" simulado usa las tasas históricas globales (home/draw/away)
    con un overround fijo, mientras que nuestro modelo usa parámetros por equipo.
    La diferencia genera EV cuando el modelo identifica equipos fuera de la media.
    """
    WARMUP_DAYS = 180

    def __init__(
        self,
        bankroll: float = config.DEFAULT_BANKROLL,
        overround: float = 1.05,
        min_ev: float = config.MIN_EV_THRESHOLD,
        kelly_fraction: float = 0.25,
        max_weekly_exposure: float = 0.20,
    ) -> None:
        self.initial_bankroll    = bankroll
        self.bankroll            = bankroll
        self.overround           = overround
        self.min_ev              = min_ev
        self.kelly_fraction      = kelly_fraction
        self.max_weekly_exposure = max_weekly_exposure

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Ejecuta el backtest completo y devuelve resultados."""
        df = self._load()
        warmup_end  = df["match_date"].min() + timedelta(days=self.WARMUP_DAYS)
        predict_df  = df[df["match_date"] > warmup_end].copy()

        if predict_df.empty:
            raise ValueError("Datos insuficientes tras el warmup de 6 meses.")

        predict_df["week"] = predict_df["match_date"].dt.to_period("W")
        weeks = sorted(predict_df["week"].unique())

        picks: list[dict]   = []
        brier_all: list[float] = []
        bankroll_history: list[dict] = []

        for week in weeks:
            week_start = week.start_time  # tz-naive Timestamp

            train_df = df[df["match_date"] < week_start]
            if len(train_df) < 100:
                continue

            baseline = self._baseline_odds(train_df)

            model = PoissonModel()
            try:
                model.fit(train_df)
            except ValueError:
                continue

            # Snapshot bankroll at week start: all stakes this week use this value.
            # P&L is accumulated and applied once at week end (batch settlement).
            week_bankroll = self.bankroll
            week_buffer: list[dict] = []

            for _, match in predict_df[predict_df["week"] == week].iterrows():
                home = match["home_team"]
                away = match["away_team"]

                if home not in model.teams or away not in model.teams:
                    continue

                actual = _outcome(int(match["home_goals"]), int(match["away_goals"]))

                try:
                    model_probs = model.predict_1x2(home, away)
                except KeyError:
                    continue

                brier_all.append(_brier(model_probs, actual))

                for vb in find_value_bets(model_probs, baseline, min_ev=self.min_ev):
                    stake_info = recommended_stake(
                        vb["model_prob"],
                        vb["odds"],
                        week_bankroll,       # fixed for the whole week
                        fraction=self.kelly_fraction,
                    )
                    if stake_info["stake"] == 0:
                        continue

                    stake = stake_info["stake"]
                    won   = vb["bet_type"] == actual
                    pnl   = round(stake * (vb["odds"] - 1) if won else -stake, 2)

                    week_buffer.append({
                        "week":       str(week),
                        "date":       str(match["match_date"].date()),
                        "league_id":  int(match["league_id"]) if pd.notna(match.get("league_id")) else None,
                        "home":       home,
                        "away":       away,
                        "bet_type":   vb["bet_type"],
                        "odds":       vb["odds"],
                        "model_prob": vb["model_prob"],
                        "ev":         vb["ev"],
                        "stake":      stake,
                        "won":        bool(won),
                        "pnl":        pnl,
                    })

            # Scale stakes if total exposure exceeds the weekly cap
            total_desired = sum(p["stake"] for p in week_buffer)
            max_stake     = week_bankroll * self.max_weekly_exposure
            if total_desired > max_stake and total_desired > 0:
                scale = max_stake / total_desired
                for p in week_buffer:
                    p["stake"] = round(p["stake"] * scale, 2)
                    p["pnl"]   = round(p["stake"] * (p["odds"] - 1) if p["won"] else -p["stake"], 2)

            # Settle the week: apply total P&L once
            week_pnl = sum(p["pnl"] for p in week_buffer)
            self.bankroll = round(self.bankroll + week_pnl, 2)

            for p in week_buffer:
                p["bankroll"] = self.bankroll
                picks.append(p)

            bankroll_history.append({"week": str(week), "bankroll": self.bankroll})

        return self._compile(picks, brier_all, bankroll_history)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load(self) -> pd.DataFrame:
        df = load_matches()
        if df.empty:
            raise ValueError("No hay partidos en la BD. Ejecuta ingest.py primero.")
        df = df.dropna(subset=["match_date", "home_goals", "away_goals"]).copy()
        df["match_date"] = pd.to_datetime(df["match_date"])
        return df.sort_values("match_date").reset_index(drop=True)

    def _baseline_odds(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Odds del bookmaker simulado: tasas históricas globales + overround.
        Genera valor cuando el modelo Poisson detecta equipos fuera de la media.
        """
        n = len(df)
        p_home = float((df["home_goals"] > df["away_goals"]).sum()) / n
        p_draw = float((df["home_goals"] == df["away_goals"]).sum()) / n
        p_away = float((df["home_goals"] < df["away_goals"]).sum()) / n
        return {
            "home_win": round(1.0 / (p_home * self.overround), 4),
            "draw":     round(1.0 / (p_draw * self.overround), 4),
            "away_win": round(1.0 / (p_away * self.overround), 4),
        }

    def _compile(
        self,
        picks: list[dict],
        brier_all: list[float],
        bankroll_history: list[dict],
    ) -> dict:
        brier_mean = round(float(np.mean(brier_all)), 4) if brier_all else None

        if not picks:
            return {
                "summary":          {"total_picks": 0, "brier_score": brier_mean},
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
                "final_bankroll":   self.bankroll,
                "total_picks":      len(picks),
                "picks_per_week":   round(len(picks) / max(n_weeks, 1), 2),
                "win_rate":         round(n_won / len(picks), 4),
                "total_staked":     round(total_staked, 2),
                "total_pnl":        round(total_pnl, 2),
                "roi":              round(total_pnl / total_staked, 4) if total_staked > 0 else 0.0,
                "brier_score":      brier_mean,
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
    print("Ejecutando backtest walk-forward...\n")
    bt = WalkForwardBacktester()
    results = bt.run()

    s = results["summary"]
    SEP = "=" * 56

    print(SEP)
    print("  RESUMEN DEL BACKTEST")
    print(SEP)
    print(f"  Picks totales     : {s['total_picks']}")
    print(f"  Picks / semana    : {s['picks_per_week']}")
    print(f"  Win rate          : {s['win_rate']:.1%}")
    print(f"  Total apostado    : {s['total_staked']:.2f}")
    print(f"  P&L total         : {s['total_pnl']:+.2f}")
    print(f"  ROI               : {s['roi']:+.1%}")
    print(f"  Brier Score       : {s['brier_score']}")
    print(f"  Bankroll inicial  : {s['initial_bankroll']:.2f}")
    print(f"  Bankroll final    : {s['final_bankroll']:.2f}")

    print(f"\n  P&L por tipo de apuesta:")
    for bet_type, pnl in sorted(results["pnl_by_bet_type"].items()):
        print(f"    {_OUTCOME_LABELS.get(bet_type, bet_type):<12}: {pnl:+.2f}")

    print(f"\n  P&L por liga (league_id):")
    for lid, pnl in sorted(results["pnl_by_league"].items(), key=lambda x: x[1], reverse=True):
        print(f"    Liga {lid:<6}: {pnl:+.2f}")

    history = results["bankroll_history"]
    if history:
        print(f"\n  Evolución bankroll (últimas 10 semanas):")
        last10 = history[-10:]
        lo = min(e["bankroll"] for e in last10)
        hi = max(e["bankroll"] for e in last10)
        span = hi - lo or 1.0
        for entry in last10:
            bar = "#" * int((entry["bankroll"] - lo) / span * 30)
            print(f"    {entry['week']}  {entry['bankroll']:>8.2f}  {bar}")

    out_path = config.DATA_DIR / "backtest_results.json"
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Guardado en: {out_path}")
    print(SEP)
