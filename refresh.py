"""
refresh.py — refresco diario: un comando para actualizar todo.

  1. Resultados recientes (partidos finalizados + sus stats).
  2. Partidos próximos + odds (7 días, incluye córners).
  3. Stats por-partido de los equipos con partido próximo.
  4. Entrena el modelo y lo guarda en disco (el dashboard lo carga al instante).
  5. Snapshot de las predicciones del modelo (bitácora de validación, solo lectura).

Uso:  python refresh.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")   # consola Windows = cp1252
except Exception:
    pass

import main as pipeline
from collectors import apifootball_collector as af


# ---------------------------------------------------------------------------
# Snapshot de predicciones — se registra ANTES del partido, se califica después.
# ---------------------------------------------------------------------------

def _devig2(a: float, b: float) -> float:
    ia, ib = 1.0 / a, 1.0 / b
    return ia / (ia + ib)


def snapshot_predictions(model, teams) -> None:
    import pandas as pd
    from sqlalchemy import select
    from database.db import get_session
    from database.models import Match, Odds, MarketOdds, TeamStatistic, ModelPrediction
    from models.poisson_model import load_matches
    from models.match_analysis import build_dossier

    matches = load_matches()
    with get_session() as s:
        srows = s.execute(select(
            TeamStatistic.match_id, TeamStatistic.team_id,
            TeamStatistic.type, TeamStatistic.value_num,
        )).all()
    stats_df = pd.DataFrame(srows, columns=["match_id", "team_id", "type", "value_num"])

    now = datetime.now(timezone.utc)
    n = 0
    with get_session() as s:
        up_ids = [o.match_id for o in
                  s.execute(select(Odds).where(Odds.market == "1x2")).scalars().all()]
        for mid in up_ids:
            m = s.get(Match, mid)
            if m is None or m.home_team_name not in teams or m.away_team_name not in teams:
                continue
            o = s.execute(select(Odds).where(
                Odds.match_id == mid, Odds.market == "1x2")).scalars().first()
            mos = s.execute(select(MarketOdds).where(MarketOdds.match_id == mid)).scalars().all()
            # Nota: el colector guarda un solo bookmaker por mercado. Si eso
            # cambiara, este dict colapsaría por selección y la cuota congelada
            # (que ahora paga el ROI) dependería del orden de iteración.
            ou = {r.selection: r.odd for r in mos if r.market == "ou_goals"}
            bt = {r.selection: r.odd for r in mos if r.market == "btts"}
            co_o = next((r for r in mos if r.market == "corners_ou" and r.selection == "over"), None)
            co_u = next((r for r in mos if r.market == "corners_ou" and r.selection == "under"), None)

            before = pd.Timestamp(m.match_date)
            before = before.tz_localize(None) if before.tzinfo else before
            market_1x2 = {"home_win": o.home_win, "draw": o.draw, "away_win": o.away_win}
            d = build_dossier(model, matches, None, m.home_team_name, m.away_team_name,
                              before, market_1x2, stats_df=stats_df)
            if not d["known"]:
                continue

            mp, kp, gm = d["model_probs"], d.get("market_probs"), d["goal_markets"]
            ca = d["stats"].get("corners")
            fields = {
                "m_home": mp["home_win"], "m_draw": mp["draw"], "m_away": mp["away_win"],
                "mk_home": kp["home_win"] if kp else None,
                "mk_draw": kp["draw"] if kp else None,
                "mk_away": kp["away_win"] if kp else None,
                "m_over25": gm["over25"],
                "mk_over25": _devig2(ou["over"], ou["under"]) if ("over" in ou and "under" in ou) else None,
                "m_btts_yes": gm["btts_yes"],
                "mk_btts_yes": _devig2(bt["yes"], bt["no"]) if ("yes" in bt and "no" in bt) else None,
                "corner_proj": ca["proj"] if ca else None,
                "corner_line": co_o.line if co_o else None,
                "corner_std": ca["std"] if ca else None,
                # Cuotas decimales reales al momento del snapshot (para ROI)
                "o_home": o.home_win, "o_draw": o.draw, "o_away": o.away_win,
                "o_over25": ou.get("over"), "o_under25": ou.get("under"),
                "o_btts_yes": bt.get("yes"), "o_btts_no": bt.get("no"),
                "o_corner_over": co_o.odd if co_o else None,
                "o_corner_under": co_u.odd if co_u else None,
            }
            existing = s.execute(select(ModelPrediction).where(
                ModelPrediction.match_id == mid)).scalars().first()
            if existing:
                for k, v in fields.items():
                    setattr(existing, k, v)
                existing.snapshot_at = now
            else:
                s.add(ModelPrediction(match_id=mid, snapshot_at=now, **fields))
            n += 1
    print(f"  Snapshot registrado para {n} partidos próximos")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = datetime.now()
    print("=" * 64)
    print("  REFRESCO DIARIO — Asistente de Análisis Pre-Partido")
    print("=" * 64)

    print("\n[1/5] Resultados recientes (+ stats)...")
    af.collect_recent_results(days_back=2)

    print("\n[2/5] Partidos próximos + odds (7 días)...")
    af.collect_upcoming(days_ahead=7)

    print("\n[3/5] Stats por-partido de equipos próximos...")
    af.collect_stats_for_upcoming(days_ahead=7, n_recent=10)

    print("\n[4/5] Entrenando modelo y guardando en disco...")
    model, teams, label, df = pipeline.build_live_model()
    pipeline.save_model_artifact(model, label, len(df))
    print(f"  Modelo guardado: {label} · {len(df)} partidos · {len(teams)} equipos")

    print("\n[5/5] Snapshot de predicciones (bitácora de validación)...")
    snapshot_predictions(model, teams)

    print("\n" + "=" * 64)
    print(f"  REFRESCO COMPLETO en {(datetime.now() - t0).seconds}s.")
    print("  Abre el dashboard:  streamlit run dashboard/app.py  (carga al instante)")
    print("=" * 64)


if __name__ == "__main__":
    main()
