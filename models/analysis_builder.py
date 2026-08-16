"""
analysis_builder.py — armado del análisis pre-partido, PURO (sin Streamlit).

Compartido por:
  - refresh.py  → precalcula todos los dossiers una vez al día y los guarda.
  - dashboard   → carga el artefacto (instantáneo) o, si falta, arma en vivo.

Mantener libre de imports de streamlit para que refresh.py pueda usarlo.
"""

from datetime import datetime

import pandas as pd
from sqlalchemy import select

from database.db import get_session
from database.models import Match, Odds, MarketOdds, TeamStatistic
from models.match_analysis import build_dossier

# Las fechas en la BD son naive pero en UTC (API-Football). El dashboard corre
# en la máquina del usuario, así que mostramos/agrupamos en su hora local.
_LOCAL_TZ = datetime.now().astimezone().tzinfo


def to_local(dt) -> "pd.Timestamp":
    """Interpreta un datetime naive como UTC y lo devuelve en hora local (naive)."""
    ts = pd.Timestamp(dt)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.tz_convert(_LOCAL_TZ).tz_localize(None)


def load_stats_df() -> "pd.DataFrame":
    """Estadísticas por-partido (team_statistics) en formato largo — una sola
    definición usada por el dashboard y por refresh.py (evita duplicar la query)."""
    with get_session() as s:
        rows = s.execute(select(
            TeamStatistic.match_id, TeamStatistic.team_id,
            TeamStatistic.type, TeamStatistic.value_num,
        )).all()
    return pd.DataFrame(rows, columns=["match_id", "team_id", "type", "value_num"])


def load_upcoming() -> list[dict]:
    """Partidos próximos con odds (desde la BD): 1X2 + O/U 2.5 + BTTS + córners."""
    out: list[dict] = []
    with get_session() as s:
        for o in s.execute(select(Odds).where(Odds.market == "1x2")).scalars().all():
            m = s.get(Match, o.match_id)
            if m is None:
                continue
            md = to_local(m.match_date)   # UTC → hora local del usuario
            mos = s.execute(select(MarketOdds).where(MarketOdds.match_id == m.id)).scalars().all()
            ou = {r.selection: r.odd for r in mos if r.market == "ou_goals"}
            bt = {r.selection: r.odd for r in mos if r.market == "btts"}
            co_o = next((r for r in mos if r.market == "corners_ou" and r.selection == "over"), None)
            co_u = next((r for r in mos if r.market == "corners_ou" and r.selection == "under"), None)
            corners = ({"line": co_o.line, "over": co_o.odd, "under": co_u.odd}
                       if co_o and co_u else None)
            out.append({
                "match_id": m.id, "league_id": m.league_id, "league": m.league_name,
                "country": m.country, "home": m.home_team_name, "away": m.away_team_name,
                "date": str(md.date()), "dt": md, "book": o.bookmaker,
                "odds_1x2": {"home_win": o.home_win, "draw": o.draw, "away_win": o.away_win},
                "ou25": ou if ("over" in ou and "under" in ou) else None,
                "btts": bt if ("yes" in bt and "no" in bt) else None,
                "corners": corners,
            })
    out.sort(key=lambda x: x["dt"])
    return out


def devig2(a: float, b: float) -> tuple[float, float]:
    ia, ib = 1.0 / a, 1.0 / b
    tot = ia + ib
    return ia / tot, ib / tot


def signals(d: dict, ou25: dict | None, btts: dict | None) -> list[dict]:
    """Diagnóstico: dónde el modelo (SIN calibrar) se aleja del mercado.

    Una entrada por mercado real (1X2, Goles O/U 2.5, BTTS), mostrando ambos
    lados. `delta` = máxima divergencia |modelo − mercado| dentro del mercado.
    NO es una sugerencia de apuesta: el modelo no está validado contra resultados.
    """
    sig: list[dict] = []
    mp, kp = d.get("model_probs"), d.get("market_probs")
    if mp and kp:
        model = {"Local": mp["home_win"], "Empate": mp["draw"], "Visit.": mp["away_win"]}
        market = {"Local": kp["home_win"], "Empate": kp["draw"], "Visit.": kp["away_win"]}
        delta = max(abs(model[o] - market[o]) for o in model)
        sig.append({"mkt": "1X2", "model": model, "market": market, "delta": delta})
    gm = d.get("goal_markets")
    if gm and ou25:
        ov, un = devig2(ou25["over"], ou25["under"])
        model = {"Over": gm["over25"], "Under": gm["under25"]}
        market = {"Over": ov, "Under": un}
        sig.append({"mkt": "Goles O/U 2.5", "model": model, "market": market,
                    "delta": abs(model["Over"] - market["Over"])})
    if gm and btts:
        y, n = devig2(btts["yes"], btts["no"])
        model = {"Sí": gm["btts_yes"], "No": gm["btts_no"]}
        market = {"Sí": y, "No": n}
        sig.append({"mkt": "BTTS", "model": model, "market": market,
                    "delta": abs(model["Sí"] - market["Sí"])})
    return sig


def build_analysis(model, matches_df, xg_df, stats_df, upcoming: list[dict]) -> list[dict]:
    """Arma dossier + señales para cada partido próximo (muta e devuelve `upcoming`).
    Cada item recibe `_d` (dossier), `_sig` (señales), `_top` (máxima divergencia)."""
    for u in upcoming:
        d = build_dossier(model, matches_df, xg_df, u["home"], u["away"], u["dt"],
                          u["odds_1x2"], stats_df=stats_df)
        u["_d"] = d
        u["_sig"] = signals(d, u["ou25"], u["btts"]) if d["known"] else []
        u["_top"] = max((s["delta"] for s in u["_sig"]), default=None)
    return upcoming
