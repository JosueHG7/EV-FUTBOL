import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def implied_probability(odds_decimal: float) -> float:
    """Convierte cuota decimal a probabilidad implícita (sin ajuste de vig)."""
    if odds_decimal <= 1.0:
        raise ValueError(f"Cuota inválida: {odds_decimal} (debe ser > 1.0)")
    return 1.0 / odds_decimal


def remove_vig(
    home_odds: float, draw_odds: float, away_odds: float
) -> dict[str, float]:
    """
    Elimina el margen de la casa y devuelve probabilidades normalizadas a 1.0.

    overround > 1.0 indica el margen del bookmaker
    (ej: 1.05 = 5% de margen).
    """
    raw = {
        "home_win": 1.0 / home_odds,
        "draw":     1.0 / draw_odds,
        "away_win": 1.0 / away_odds,
    }
    overround = sum(raw.values())
    return {k: v / overround for k, v in raw.items()}


def calculate_ev(model_prob: float, odds_decimal: float) -> float:
    """
    Calcula el valor esperado de una apuesta.

    EV = (model_prob × (odds - 1)) - (1 - model_prob)
       = model_prob × odds - 1

    EV > 0 → apuesta con valor positivo.
    Ejemplo: EV=0.05 significa 5% de ventaja por unidad apostada.
    """
    return (model_prob * (odds_decimal - 1)) - (1.0 - model_prob)


# ---------------------------------------------------------------------------
# Value bet scanner
# ---------------------------------------------------------------------------

def find_value_bets(
    model_probs: dict[str, float],
    market_odds: dict[str, float],
    min_ev: float = 0.03,
) -> list[dict]:
    """
    Compara las probabilidades del modelo con las cuotas de mercado.

    Parámetros:
        model_probs : {home_win, draw, away_win} del modelo Poisson
        market_odds : {home_win, draw, away_win} cuotas decimales del bookmaker
        min_ev      : umbral mínimo de EV para incluir (default 3%)

    Devuelve lista de value bets ordenada por EV descendente.
    Cada item:
        bet_type        — home_win | draw | away_win
        model_prob      — probabilidad del modelo
        implied_prob    — probabilidad implícita en la cuota (sin vig)
        odds            — cuota decimal del mercado
        ev              — valor esperado (0.05 = 5% edge)
        edge            — model_prob - implied_prob
        overround       — margen total del mercado
    """
    no_vig_probs = remove_vig(
        market_odds["home_win"],
        market_odds["draw"],
        market_odds["away_win"],
    )
    overround = sum(1.0 / o for o in market_odds.values())

    value_bets: list[dict] = []

    for bet_type in ("home_win", "draw", "away_win"):
        m_prob  = model_probs[bet_type]
        odds    = market_odds[bet_type]
        imp     = no_vig_probs[bet_type]
        ev      = calculate_ev(m_prob, odds)
        edge    = m_prob - imp

        if ev >= min_ev:
            value_bets.append({
                "bet_type":     bet_type,
                "model_prob":   round(m_prob, 4),
                "implied_prob": round(imp,    4),
                "odds":         odds,
                "ev":           round(ev,   4),
                "edge":         round(edge, 4),
                "overround":    round(overround, 4),
            })

    return sorted(value_bets, key=lambda x: x["ev"], reverse=True)


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from models.poisson_model import PoissonModel, load_matches

    # --- Cargar y entrenar modelo ---
    print("Cargando datos y entrenando modelo Poisson...")
    df = load_matches()
    model = PoissonModel()
    model.fit(df)
    print(f"  {len(df)} partidos | {len(model.teams)} equipos\n")

    # --- Predicciones del modelo ---
    HOME, AWAY = "Real Madrid", "Barcelona"
    model_probs = model.predict_1x2(HOME, AWAY)
    print(f"Predicciones del modelo — {HOME} vs {AWAY}:")
    print(f"  Local    : {model_probs['home_win']:.1%}")
    print(f"  Empate   : {model_probs['draw']:.1%}")
    print(f"  Visitante: {model_probs['away_win']:.1%}")

    # --- Cuotas de mercado simuladas ---
    market_odds = {"home_win": 2.10, "draw": 3.40, "away_win": 3.20}
    overround = sum(1 / o for o in market_odds.values())
    print(f"\nCuotas de mercado: Local {market_odds['home_win']} | "
          f"Empate {market_odds['draw']} | Visitante {market_odds['away_win']}")
    print(f"  Overround (margen casa): {overround:.4f} ({(overround - 1) * 100:.2f}%)\n")

    # --- Probabilidades sin vig ---
    no_vig = remove_vig(market_odds["home_win"], market_odds["draw"], market_odds["away_win"])
    print("Probabilidades implícitas (sin vig):")
    for k, v in no_vig.items():
        print(f"  {k:<10}: {v:.1%}")

    # --- EV por resultado ---
    print("\nEV por resultado:")
    labels = {"home_win": "Local", "draw": "Empate", "away_win": "Visitante"}
    for bet_type, label in labels.items():
        ev = calculate_ev(model_probs[bet_type], market_odds[bet_type])
        marker = " <<< VALUE BET" if ev >= 0.03 else ""
        print(f"  {label:<10}: EV={ev:+.2%}{marker}")

    # --- Value bets ---
    print("\nValue bets (EV >= 3%):")
    value_bets = find_value_bets(model_probs, market_odds, min_ev=0.03)
    if not value_bets:
        print("  Sin value bets en este mercado.")
    else:
        for vb in value_bets:
            print(
                f"  [{vb['bet_type']}] "
                f"odds={vb['odds']} | "
                f"modelo={vb['model_prob']:.1%} | "
                f"implícita={vb['implied_prob']:.1%} | "
                f"EV={vb['ev']:+.2%} | "
                f"edge={vb['edge']:+.1%}"
            )
