"""
models/soft_book_experiment.py

¿Podemos batir a un book BLANDO (Bet365) aunque no batamos a Pinnacle?

Genera los candidatos +EV walk-forward UNA sola vez contra las odds de Bet365
(la parte cara: ajusta Poisson+XGBoost por ventana), y luego barre filtros de
apuesta de forma barata sobre esos candidatos:
  - MAX_ODDS  : descartar underdogs de cuota alta (donde el modelo se equivoca)
  - MAX_EV    : descartar "EV demasiado bueno para ser verdad" (>15-20%)

Uso:
    python models/soft_book_experiment.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from models.real_backtester import RealBacktester


def _filter(cands: list[dict], max_odds: float | None, max_ev: float | None) -> list[dict]:
    """Post-filtra candidatos por cuota máxima y EV máximo (baratísimo)."""
    out = cands
    if max_odds is not None:
        out = [c for c in out if c["odds"] <= max_odds]
    if max_ev is not None:
        out = [c for c in out if c["ev"] <= max_ev]
    return out


def _row(bt, cands, briers, skips, model, min_ev, max_odds, max_ev):
    filtered = _filter(cands[model], max_odds, max_ev)
    res = bt._settle(filtered, briers[model], skips[model], min_ev)
    s = res["summary"]
    mo = "—" if max_odds is None else f"{max_odds:.0f}"
    me = "—" if max_ev is None else f"{max_ev:.0%}"
    print(f"  {model:<9} {min_ev:>4.0%} {mo:>7} {me:>6}  "
          f"{s['total_picks']:>6}  {s.get('win_rate',0):>6.1%}  "
          f"{s.get('roi',0):>+7.1%}  {s['max_drawdown']:>6.1%}  "
          f"{s.get('final_bankroll', bt.initial_bankroll):>10.2f}")
    return s.get("roi", 0.0), s["total_picks"]


if __name__ == "__main__":
    SEP = "=" * 78
    print(SEP)
    print("  EXPERIMENTO: ¿batimos a Bet365 (book blando)? + barrido de filtros")
    print(SEP + "\n")

    bt = RealBacktester(bookmaker="Bet365")

    print("  Generando candidatos walk-forward contra Bet365 (una vez)...")
    cands, briers, skips = bt._compute_all_candidates(warmup_days=365)
    print(f"\n  Candidatos +EV (min_ev=0):")
    for m in ("poisson", "xgboost", "ensemble"):
        print(f"    {m:<10}: {len(cands[m])}")

    print(f"\n{SEP}")
    print("  BARRIDO DE FILTROS (Bet365)")
    print(SEP)
    print(f"  {'modelo':<9} {'minEV':>4} {'maxOdd':>7} {'maxEV':>6}  "
          f"{'picks':>6}  {'win%':>6}  {'ROI':>7}  {'DD':>6}  {'final':>10}")
    print("  " + "-" * 74)

    # Baseline sin filtros (equivalente a lo que hace el sistema hoy)
    for model in ("poisson", "xgboost", "ensemble"):
        _row(bt, cands, briers, skips, model, 0.03, None, None)
    print("  " + "-" * 74)

    # Barrido de filtros sobre el Ensemble (el modelo de producción)
    configs = [
        (0.03, 6, None),
        (0.03, 5, None),
        (0.03, 4, None),
        (0.03, 5, 0.20),
        (0.03, 5, 0.15),
        (0.05, 5, 0.15),
        (0.03, 4, 0.15),
    ]
    for min_ev, max_odds, max_ev in configs:
        _row(bt, cands, briers, skips, "ensemble", min_ev, max_odds, max_ev)

    print(SEP)
    print("  Nota: Pinnacle (referencia, corrido antes) daba Ensemble ROI -5.1%.")
    print(SEP)
