"""
grading.py — califica los snapshots de ModelPrediction contra el resultado real.

Puro (sin BD, sin pandas): recibe las probabilidades selladas antes del partido
más el desenlace real y devuelve, por mercado, la "apuesta direccional" del
modelo y del mercado y si acertó.

Es diagnóstico honesto — mide **acierto direccional** (¿el lado más probable del
modelo salió?), NO un backtest de ROI. Un modelo sin calibrar puede acertar
dirección y aun así no ser rentable frente a las cuotas.
"""

from __future__ import annotations

MARKETS = ("1x2", "ou25", "btts", "corners", "cards")


# ---------------------------------------------------------------------------
# Etiquetas
# ---------------------------------------------------------------------------

def _ou_lbl(over: bool | None) -> str | None:
    return None if over is None else ("Over" if over else "Under")


def _yn_lbl(yes: bool | None) -> str | None:
    return None if yes is None else ("Sí" if yes else "No")


def outcome_1x2(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home"
    if home_goals < away_goals:
        return "away"
    return "draw"


def _argmax3(a, b, c, labels=("home", "draw", "away")) -> str | None:
    """Etiqueta de la probabilidad más alta, o None si alguna falta."""
    if a is None or b is None or c is None:
        return None
    vals = (a, b, c)
    return labels[vals.index(max(vals))]


# ---------------------------------------------------------------------------
# Llamadas direccionales (sin resultado — para snapshots pendientes)
# ---------------------------------------------------------------------------

def model_calls(pred) -> dict:
    """Lado más probable del MODELO por mercado (None si falta el dato)."""
    over = None if pred.m_over25 is None else pred.m_over25 >= 0.5
    yes = None if pred.m_btts_yes is None else pred.m_btts_yes >= 0.5
    corner = None
    if pred.corner_proj is not None and pred.corner_line is not None:
        corner = _ou_lbl(pred.corner_proj > pred.corner_line)
    card = None
    if pred.card_proj is not None and pred.card_line is not None:
        card = _ou_lbl(pred.card_proj > pred.card_line)
    return {
        "1x2": _argmax3(pred.m_home, pred.m_draw, pred.m_away),
        "ou25": _ou_lbl(over),
        "btts": _yn_lbl(yes),
        "corners": corner,
        "cards": card,
    }


def market_calls(pred) -> dict:
    """Lado más probable del MERCADO (de-vigged) por mercado."""
    over = None if pred.mk_over25 is None else pred.mk_over25 >= 0.5
    yes = None if pred.mk_btts_yes is None else pred.mk_btts_yes >= 0.5
    return {
        "1x2": _argmax3(pred.mk_home, pred.mk_draw, pred.mk_away),
        "ou25": _ou_lbl(over),
        "btts": _yn_lbl(yes),
        "corners": None,   # la línea ES el mercado; no hay lado favorito
        "cards": None,      # ídem
    }


# ---------------------------------------------------------------------------
# Calificación (con resultado real)
# ---------------------------------------------------------------------------

def grade_snapshot(pred, home_goals: int, away_goals: int,
                   corners_total: float | None = None,
                   cards_total: float | None = None) -> dict:
    """
    Devuelve {mercado: {"model", "market", "actual", "model_hit", "market_hit"}}.
    *_hit es None cuando faltan datos para calificar ese mercado (p. ej. una liga
    sin línea de córners/tarjetas, o un mercado sin cuota registrada).
    """
    mc, kc = model_calls(pred), market_calls(pred)

    # 1X2
    act_1x2 = outcome_1x2(home_goals, away_goals)
    # Over/Under 2.5 goles
    act_over = _ou_lbl((home_goals + away_goals) > 2.5)
    # BTTS
    act_btts = _yn_lbl(home_goals > 0 and away_goals > 0)
    # Córners / tarjetas totales — un empate exacto con la línea es push (no calificable).
    act_corners = None
    if (pred.corner_line is not None and corners_total is not None
            and corners_total != pred.corner_line):
        act_corners = _ou_lbl(corners_total > pred.corner_line)
    act_cards = None
    if (pred.card_line is not None and cards_total is not None
            and cards_total != pred.card_line):
        act_cards = _ou_lbl(cards_total > pred.card_line)

    actual = {"1x2": act_1x2, "ou25": act_over, "btts": act_btts,
              "corners": act_corners, "cards": act_cards}

    g = {}
    for mk in MARKETS:
        a = actual[mk]
        m, k = mc[mk], kc[mk]
        g[mk] = {
            "model": m, "market": k, "actual": a,
            "model_hit":  (m == a) if (m is not None and a is not None) else None,
            "market_hit": (k == a) if (k is not None and a is not None) else None,
        }
    return g


def summarize(grades: list[dict]) -> dict:
    """Agrega aciertos por mercado: {mk: {model_hits, model_n, market_hits, market_n}}."""
    out = {}
    for mk in MARKETS:
        mh = mn = kh = kn = 0
        for g in grades:
            cell = g[mk]
            if cell["model_hit"] is not None:
                mn += 1
                mh += int(cell["model_hit"])
            if cell["market_hit"] is not None:
                kn += 1
                kh += int(cell["market_hit"])
        out[mk] = {"model_hits": mh, "model_n": mn, "market_hits": kh, "market_n": kn}
    return out


# ---------------------------------------------------------------------------
# P&L / ROI — ¿le habría ganado a la cuota? (necesita las cuotas del snapshot)
# ---------------------------------------------------------------------------

# Atributo de ModelPrediction con la cuota decimal de cada selección.
_SEL_ODDS = {
    "1x2":     {"home": "o_home", "draw": "o_draw", "away": "o_away"},
    "ou25":    {"Over": "o_over25", "Under": "o_under25"},
    "btts":    {"Sí": "o_btts_yes", "No": "o_btts_no"},
    "corners": {"Over": "o_corner_over", "Under": "o_corner_under"},
    "cards": {"Over": "o_card_over", "Under": "o_card_under"},
}


def model_odds(pred, grade: dict) -> dict:
    """Cuota decimal del lado que llamó el MODELO por mercado (None si falta)."""
    out = {}
    for mk in MARKETS:
        attr = _SEL_ODDS[mk].get(grade[mk]["model"]) if grade[mk]["model"] else None
        out[mk] = getattr(pred, attr, None) if attr else None
    return out


def _bet_pnl(call, hit, odds) -> float | None:
    """Ganancia de apostar 1 unidad a `call` a cuota `odds`. None si no aplica.
    Gana → odds-1 ; pierde → -1. Excluye push/sin cuota/sin resultado."""
    if call is None or hit is None or odds is None:
        return None
    return (odds - 1.0) if hit else -1.0


def pnl_snapshot(pred, grade: dict) -> dict:
    """
    Devuelve {mercado: {"model_pnl": float|None, "market_pnl": float|None}} —
    apostando 1 unidad plana al lado que llamó cada uno, a la cuota sellada.
    """
    out = {}
    for mk in MARKETS:
        cell = grade[mk]
        m_attr = _SEL_ODDS[mk].get(cell["model"]) if cell["model"] else None
        k_attr = _SEL_ODDS[mk].get(cell["market"]) if cell["market"] else None
        m_odds = getattr(pred, m_attr, None) if m_attr else None
        k_odds = getattr(pred, k_attr, None) if k_attr else None
        out[mk] = {
            "model_pnl":  _bet_pnl(cell["model"], cell["model_hit"], m_odds),
            "market_pnl": _bet_pnl(cell["market"], cell["market_hit"], k_odds),
        }
    return out


def summarize_pnl(pnls: list[dict]) -> dict:
    """Agrega P&L por mercado: {mk: {model_pnl, model_bets, market_pnl, market_bets}}.
    ROI = model_pnl / model_bets (unidades ganadas por apuesta, staking plano)."""
    out = {}
    for mk in MARKETS:
        mp = kp = 0.0
        mb = kb = 0
        for p in pnls:
            cell = p[mk]
            if cell["model_pnl"] is not None:
                mp += cell["model_pnl"]
                mb += 1
            if cell["market_pnl"] is not None:
                kp += cell["market_pnl"]
                kb += 1
        out[mk] = {"model_pnl": mp, "model_bets": mb,
                   "market_pnl": kp, "market_bets": kb}
    return out
