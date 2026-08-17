"""
refresh.py — refresco diario: un comando para actualizar todo.

  1. Resultados recientes (partidos finalizados + sus stats).
  2. Partidos próximos + odds (7 días, incluye córners).
  3. Stats por-partido de los equipos con partido próximo.
  4. Entrena el modelo y lo guarda en disco (el dashboard lo carga al instante).
  5. Precalcula el análisis (dossiers + señales) y lo guarda (dashboard instantáneo).
  6. Snapshot de las predicciones del modelo (bitácora de validación, solo lectura).

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

# ---------------------------------------------------------------------------
# Precálculo del análisis — dossiers + señales. Se construye UNA vez y alimenta
# tanto el artefacto del dashboard como el snapshot (evita rearmar dossiers).
# ---------------------------------------------------------------------------

def precompute_analysis(model, matches_df, label: str) -> list:
    from collectors.understat_collector import load_all_xg
    from models.analysis_builder import load_upcoming, load_stats_df, build_analysis

    try:
        xg_df = load_all_xg()
    except Exception:
        xg_df = None
    stats_df = load_stats_df()

    analyzed = build_analysis(model, matches_df, xg_df, stats_df, load_upcoming())
    known = sum(1 for u in analyzed if u["_d"]["known"])
    try:
        pipeline.save_analysis_artifact(analyzed, label, len(matches_df))
        print(f"  Análisis precalculado y guardado: {len(analyzed)} partidos ({known} con modelo)")
    except Exception as exc:  # un dossier no-serializable no debe voltear el refresh
        print(f"  [aviso] análisis armado pero no se pudo guardar: "
              f"{type(exc).__name__}: {exc} — el dashboard lo armará en vivo.",
              file=sys.stderr)
    return analyzed


# ---------------------------------------------------------------------------
# Snapshot de predicciones — se registra ANTES del partido, se califica después.
# Consume los dossiers ya construidos por precompute_analysis (no los rearma).
# ---------------------------------------------------------------------------

def snapshot_predictions(analyzed: list) -> None:
    from sqlalchemy import select
    from database.db import get_session
    from database.models import ModelPrediction
    from models.analysis_builder import devig2, to_local

    now = datetime.now(timezone.utc)
    now_local = to_local(now)   # mismo marco naive-local que u["dt"], para comparar
    n = 0
    with get_session() as s:
        for u in analyzed:
            d = u["_d"]
            if not d["known"]:
                continue
            mp, kp, gm = d["model_probs"], d.get("market_probs"), d["goal_markets"]
            ca = d["stats"].get("corners")
            o1, ou, bt, co = u["odds_1x2"], u.get("ou25"), u.get("btts"), u.get("corners")
            fields = {
                "m_home": mp["home_win"], "m_draw": mp["draw"], "m_away": mp["away_win"],
                "mk_home": kp["home_win"] if kp else None,
                "mk_draw": kp["draw"] if kp else None,
                "mk_away": kp["away_win"] if kp else None,
                "m_over25": gm["over25"],
                "mk_over25": devig2(ou["over"], ou["under"])[0] if ou else None,
                "m_btts_yes": gm["btts_yes"],
                "mk_btts_yes": devig2(bt["yes"], bt["no"])[0] if bt else None,
                "corner_proj": ca["proj"] if ca else None,
                "corner_line": co["line"] if co else None,
                "corner_std": ca["std"] if ca else None,
                # Cuotas decimales reales al momento del snapshot (para ROI)
                "o_home": o1["home_win"], "o_draw": o1["draw"], "o_away": o1["away_win"],
                "o_over25": ou["over"] if ou else None,
                "o_under25": ou["under"] if ou else None,
                "o_btts_yes": bt["yes"] if bt else None,
                "o_btts_no": bt["no"] if bt else None,
                "o_corner_over": co["over"] if co else None,
                "o_corner_under": co["under"] if co else None,
                # Defensa en profundidad: si el snapshot se toma con el partido ya
                # iniciado (no debería, load_upcoming ya filtra, pero cubre la
                # ventana entre ambos y una eventual regresión del filtro), se
                # auto-marca como contaminado y no contará para el ROI/acierto.
                "leak_flagged": now_local >= u["dt"],
            }
            mid = u["match_id"]
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

    print("\n[1/6] Resultados recientes (+ stats)...")
    af.collect_recent_results(days_back=2)

    print("\n[2/6] Partidos próximos + odds (7 días)...")
    af.collect_upcoming(days_ahead=7)

    print("\n[3/6] Stats por-partido de equipos próximos...")
    af.collect_stats_for_upcoming(days_ahead=7, n_recent=10)

    print("\n[4/6] Entrenando modelo y guardando en disco...")
    model, teams, label, df = pipeline.build_live_model()
    pipeline.save_model_artifact(model, label, len(df))
    print(f"  Modelo guardado: {label} · {len(df)} partidos · {len(teams)} equipos")

    print("\n[5/6] Precalculando análisis (dossiers + señales)...")
    analyzed = precompute_analysis(model, df, label)

    print("\n[6/6] Snapshot de predicciones (bitácora de validación)...")
    snapshot_predictions(analyzed)

    print("\n" + "=" * 64)
    print(f"  REFRESCO COMPLETO en {(datetime.now() - t0).seconds}s.")
    print("  Abre el dashboard:  streamlit run dashboard/app.py  (carga al instante)")
    print("=" * 64)


if __name__ == "__main__":
    main()
