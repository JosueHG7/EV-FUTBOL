import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config


# ---------------------------------------------------------------------------
# Core Kelly math
# ---------------------------------------------------------------------------

def kelly_fraction(model_prob: float, odds_decimal: float) -> float:
    """
    Kelly completo: f = (model_prob × odds - 1) / (odds - 1)

    Devuelve 0.0 si no hay ventaja (f <= 0).
    """
    if odds_decimal <= 1.0:
        return 0.0
    f = (model_prob * odds_decimal - 1.0) / (odds_decimal - 1.0)
    return max(f, 0.0)


def fractional_kelly(
    model_prob: float, odds_decimal: float, fraction: float = 0.25
) -> float:
    """
    Kelly fraccionado: multiplica el Kelly completo por `fraction`.

    fraction=0.25 (25%) reduce la varianza de forma significativa.
    Devuelve 0.0 si Kelly completo <= 0.
    """
    kelly_full = kelly_fraction(model_prob, odds_decimal)
    if kelly_full <= 0.0:
        return 0.0
    return kelly_full * fraction


def recommended_stake(
    model_prob: float,
    odds_decimal: float,
    bankroll: float,
    fraction: float = 0.25,
    min_kelly: float = config.MIN_KELLY_FRACTION,
    max_kelly: float = config.MAX_KELLY_FRACTION,
) -> dict:
    """
    Calcula la apuesta recomendada con Kelly fraccionado y límites de riesgo.

    Parámetros:
        model_prob    — probabilidad del modelo para este resultado
        odds_decimal  — cuota decimal del mercado
        bankroll      — bankroll actual en unidades monetarias
        fraction      — fracción de Kelly a usar (default 0.25)
        min_kelly     — porcentaje mínimo del bankroll (default 1%)
        max_kelly     — porcentaje máximo del bankroll (default 5%)

    Devuelve dict con:
        kelly_full        — Kelly completo (sin fraccionar)
        kelly_fractional  — Kelly fraccionado aplicado
        stake             — importe a apostar (0 si no hay ventaja suficiente)
        pct_bankroll      — porcentaje del bankroll que representa la apuesta
    """
    kelly_full = kelly_fraction(model_prob, odds_decimal)
    kelly_frac = fractional_kelly(model_prob, odds_decimal, fraction)

    if kelly_frac <= min_kelly:
        return {
            "kelly_full":       round(kelly_full, 4),
            "kelly_fractional": round(kelly_frac, 4),
            "stake":            0.0,
            "pct_bankroll":     0.0,
        }

    clamped = min(kelly_frac, max_kelly)
    stake = round(clamped * bankroll, 2)

    return {
        "kelly_full":       round(kelly_full, 4),
        "kelly_fractional": round(kelly_frac, 4),
        "stake":            stake,
        "pct_bankroll":     round(clamped, 4),
    }


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    BANKROLL = 100.0
    ODDS     = 3.2
    PROBS    = [0.30, 0.35, 0.40, 0.45]

    print(f"Bankroll: {BANKROLL:.0f} | Odds: {ODDS}\n")
    header = f"{'prob':>6}  {'kelly_full':>10}  {'kelly_25%':>10}  {'stake':>8}  {'pct_bankroll':>12}"
    print(header)
    print("-" * len(header))

    for prob in PROBS:
        kf   = kelly_fraction(prob, ODDS)
        k25  = fractional_kelly(prob, ODDS, fraction=0.25)
        rec  = recommended_stake(prob, ODDS, BANKROLL)
        print(
            f"{prob:>6.0%}  "
            f"{kf:>10.2%}  "
            f"{k25:>10.2%}  "
            f"{rec['stake']:>8.2f}  "
            f"{rec['pct_bankroll']:>12.2%}"
        )
