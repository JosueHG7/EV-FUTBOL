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

MARKETS = ("1x2", "ou25", "btts", "corners")


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
    return {
        "1x2": _argmax3(pred.m_home, pred.m_draw, pred.m_away),
        "ou25": _ou_lbl(over),
        "btts": _yn_lbl(yes),
        "corners": corner,
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
    }


# ---------------------------------------------------------------------------
# Calificación (con resultado real)
# ---------------------------------------------------------------------------

def grade_snapshot(pred, home_goals: int, away_goals: int,
                   corners_total: float | None = None) -> dict:
    """
    Devuelve {mercado: {"model", "market", "actual", "model_hit", "market_hit"}}.
    *_hit es None cuando faltan datos para calificar ese mercado (p. ej. una liga
    sin línea de córners, o un mercado sin cuota registrada).
    """
    mc, kc = model_calls(pred), market_calls(pred)

    # 1X2
    act_1x2 = outcome_1x2(home_goals, away_goals)
    # Over/Under 2.5 goles
    act_over = _ou_lbl((home_goals + away_goals) > 2.5)
    # BTTS
    act_btts = _yn_lbl(home_goals > 0 and away_goals > 0)
    # Córners totales — un empate exacto con la línea es push (no calificable).
    act_corners = None
    if (pred.corner_line is not None and corners_total is not None
            and corners_total != pred.corner_line):
        act_corners = _ou_lbl(corners_total > pred.corner_line)

    actual = {"1x2": act_1x2, "ou25": act_over, "btts": act_btts, "corners": act_corners}

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
