"""
dashboard/app.py

Streamlit dashboard for the Football EV Betting System.

Run:
    streamlit run dashboard/app.py

Two sections:
  1. Picks en vivo  — entrena el EnsembleModel (cacheado) y escanea las odds
                      actuales (data/odds_raw.json) en busca de value bets +EV.
  2. Rendimiento    — KPIs, bankroll histórico, PnL por liga / tipo de apuesta
                      y tabla de picks del backtest walk-forward validado
                      (data/real_backtest_results.json).
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))
import config
import main as pipeline

# ---------------------------------------------------------------------------
# Labels / constants
# ---------------------------------------------------------------------------

_LEAGUE_BY_ID = {
    2:   "Champions League",
    39:  "Premier League",
    61:  "Ligue 1",
    78:  "Bundesliga",
    135: "Serie A",
    140: "La Liga",
}

_BET_LABELS = {
    "home_win": "Local",
    "draw":     "Empate",
    "away_win": "Visitante",
}

_BACKTEST_PATH = config.DATA_DIR / "real_backtest_results.json"


# ---------------------------------------------------------------------------
# Cached data / model loaders
# ---------------------------------------------------------------------------

def _db_mtime() -> float:
    """mtime de la BD SQLite. Se pasa como clave de caché a los loaders de datos:
    cuando refresh.py escribe en la BD el mtime cambia y las cachés se invalidan
    solas → el dashboard refleja el refresco sin reiniciar ni esperar el TTL."""
    try:
        return config.DB_PATH.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_resource(max_entries=2, show_spinner="Cargando modelo…")
def _load_model_cached(pkl_mtime: float):
    """Carga el modelo entrenado por refresh.py (instantáneo). Si no existe,
    lo entrena al vuelo (lento) como respaldo. Cacheado por mtime del .pkl:
    cuando refresh.py reescribe el artefacto, el mtime cambia y la caché se
    invalida sola (un dashboard abierto recoge el modelo nuevo sin reiniciar)."""
    loaded = pipeline.load_model_artifact()
    if loaded is not None:
        return loaded
    model, teams, label, df = pipeline.build_live_model()
    return model, teams, label, len(df)


def _get_live_model():
    mtime = pipeline.MODEL_PATH.stat().st_mtime if pipeline.MODEL_PATH.exists() else 0.0
    return _load_model_cached(mtime)


@st.cache_data(ttl=300, show_spinner=False)
def _get_raw_odds():
    """Carga data/odds_raw.json (refresca cada 5 min)."""
    return pipeline.load_raw_odds()


@st.cache_data(show_spinner=False)
def _load_backtest():
    """Carga el JSON de resultados del backtest walk-forward."""
    if not _BACKTEST_PATH.exists():
        return None
    return json.loads(_BACKTEST_PATH.read_text(encoding="utf-8"))


@st.cache_data(ttl=900, max_entries=3, show_spinner=False)
def _get_matches_cached(db_mtime: float):
    """Partidos finalizados de la BD (para forma/H2H/descanso)."""
    from models.poisson_model import load_matches
    return load_matches()


def _get_matches():
    return _get_matches_cached(_db_mtime())


@st.cache_data(ttl=3600, show_spinner=False)
def _get_xg():
    """xG de Understat (caché 1h en la app; el colector cachea 48h)."""
    from collectors.understat_collector import load_all_xg
    try:
        return load_all_xg()
    except Exception:
        return None


@st.cache_data(ttl=900, max_entries=3, show_spinner=False)
def _get_stats_df_cached(db_mtime: float):
    """Estadísticas por-partido (team_statistics) para promedios rodantes."""
    from models.analysis_builder import load_stats_df
    return load_stats_df()


def _get_stats_df():
    return _get_stats_df_cached(_db_mtime())


@st.cache_data(ttl=900, max_entries=4, show_spinner=False)
def _get_recent_finished_cached(db_mtime: float, days_back: int = 3):
    """Partidos finalizados de los últimos días, con desenlace de mercados + stats."""
    from datetime import datetime, timezone, timedelta
    from sqlalchemy import select
    from database.db import get_session
    from database.models import Match, TeamStatistic

    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    out = []
    with get_session() as s:
        ms = s.execute(select(Match).where(
            Match.status == "finished", Match.home_goals.is_not(None),
            Match.match_date >= cutoff,
        )).scalars().all()
        mids = [m.id for m in ms]
        totals: dict = {}
        if mids:
            for mid, typ, val in s.execute(select(
                TeamStatistic.match_id, TeamStatistic.type, TeamStatistic.value_num
            ).where(TeamStatistic.match_id.in_(mids))).all():
                if val is None or pd.isna(val):
                    continue
                totals.setdefault(mid, {})
                totals[mid][typ] = totals[mid].get(typ, 0.0) + val
        from models.analysis_builder import to_local
        for m in ms:
            md = to_local(m.match_date)   # UTC → hora local
            tg = m.home_goals + m.away_goals
            st_ = totals.get(m.id, {})
            out.append({
                "league": m.league_name, "country": m.country,
                "home": m.home_team_name, "away": m.away_team_name,
                "date": str(md.date()), "dt": md,
                "score": f"{m.home_goals}-{m.away_goals}",
                "over25": tg > 2.5, "btts": m.home_goals > 0 and m.away_goals > 0,
                "corners": st_.get("Corner Kicks"), "cards": st_.get("Yellow Cards"),
            })
    out.sort(key=lambda x: x["dt"], reverse=True)
    return out


def _get_recent_finished(days_back: int = 3):
    return _get_recent_finished_cached(_db_mtime(), days_back)


@st.cache_data(ttl=900, max_entries=3, show_spinner=False)
def _get_predictions_cached(db_mtime: float):
    """Snapshots de ModelPrediction, partidos en dos grupos:
    - graded: partido finalizado → calificado contra el resultado real.
    - pending: aún sin jugar → solo lo que el modelo predijo.
    Devuelve (graded, pending, summary, pnl_summary). Todo se resuelve dentro
    de la sesión (los objetos ORM no sobreviven fuera de ella)."""
    from sqlalchemy import select
    from database.db import get_session
    from database.models import ModelPrediction, Match, TeamStatistic
    from models.grading import (grade_snapshot, summarize, model_calls,
                                 market_calls, pnl_snapshot, summarize_pnl,
                                 model_odds)

    graded, pending, grade_objs, pnl_objs = [], [], [], []
    n_leaked = 0   # calificados excluidos del agregado por snapshot contaminado
    with get_session() as s:
        preds = s.execute(select(ModelPrediction)).scalars().all()
        pids = [p.match_id for p in preds]
        # Un solo SELECT para todos los Match (evita N+1).
        matches = {}
        if pids:
            matches = {m.id: m for m in s.execute(
                select(Match).where(Match.id.in_(pids))).scalars().all()}

        # Un partido está calificable solo con AMBOS goles presentes.
        finished_ids = [
            mid for mid, m in matches.items()
            if m.status == "finished"
            and m.home_goals is not None and m.away_goals is not None
        ]
        # Córners totales solo de los finalizados (los pendientes no los usan).
        corners: dict = {}
        if finished_ids:
            for mid, val in s.execute(select(
                TeamStatistic.match_id, TeamStatistic.value_num
            ).where(TeamStatistic.match_id.in_(finished_ids),
                    TeamStatistic.type == "Corner Kicks")).all():
                if val is None or pd.isna(val):
                    continue
                corners[mid] = corners.get(mid, 0.0) + val
        finished_set = set(finished_ids)

        from models.analysis_builder import to_local
        for p in preds:
            m = matches.get(p.match_id)
            if m is None:
                continue
            md = to_local(m.match_date)   # UTC → hora local
            base = {
                "date": str(md.date()), "dt": md,
                "home": m.home_team_name, "away": m.away_team_name,
                "league": m.league_name,
            }
            if m.id in finished_set:
                g = grade_snapshot(p, m.home_goals, m.away_goals, corners.get(m.id))
                # Los snapshots contaminados (tomados con el partido ya iniciado) se
                # califican y muestran, pero NO cuentan para el ROI/acierto agregado.
                if not p.leak_flagged:
                    grade_objs.append(g)
                    pnl_objs.append(pnl_snapshot(p, g))
                else:
                    n_leaked += 1
                graded.append({
                    **base, "score": f"{m.home_goals}-{m.away_goals}",
                    "corners_total": corners.get(m.id), "g": g,
                    "corner_line": p.corner_line, "leaked": p.leak_flagged,
                    "odds": model_odds(p, g),
                })
            else:
                pending.append({
                    **base,
                    "mc": model_calls(p), "kc": market_calls(p),
                    "m_home": p.m_home, "m_draw": p.m_draw, "m_away": p.m_away,
                    "m_over25": p.m_over25, "m_btts_yes": p.m_btts_yes,
                    "corner_proj": p.corner_proj, "corner_line": p.corner_line,
                    # Cuotas selladas (para mostrar al lado de cada llamada).
                    "o_home": p.o_home, "o_draw": p.o_draw, "o_away": p.o_away,
                    "o_over25": p.o_over25, "o_under25": p.o_under25,
                    "o_btts_yes": p.o_btts_yes, "o_btts_no": p.o_btts_no,
                    "o_corner_over": p.o_corner_over, "o_corner_under": p.o_corner_under,
                })

    graded.sort(key=lambda x: x["dt"], reverse=True)
    pending.sort(key=lambda x: x["dt"])
    return (graded, pending, summarize(grade_objs), summarize_pnl(pnl_objs),
            n_leaked)


def _get_predictions():
    return _get_predictions_cached(_db_mtime())


# ---------------------------------------------------------------------------
# Section: Picks en vivo
# ---------------------------------------------------------------------------

def _render_live_picks() -> None:
    st.header("🎯 Picks en vivo")

    with st.sidebar:
        st.subheader("Parámetros")
        min_ev = st.slider(
            "EV mínimo", 0.0, 0.20,
            float(config.MIN_EV_THRESHOLD), 0.005,
            format="%.1f%%",
            help="Umbral mínimo de valor esperado para mostrar un pick.",
        )
        bankroll = st.number_input(
            "Bankroll", min_value=1.0, value=float(config.DEFAULT_BANKROLL), step=10.0,
            help="Bankroll de referencia para el stake Kelly.",
        )

    try:
        model, teams, label, n_matches = _get_live_model()
    except Exception as exc:
        st.error(f"No se pudo entrenar el modelo: {exc}")
        st.info("Verifica que la BD tiene datos (`python database/ingest.py`).")
        return

    try:
        raw_odds = _get_raw_odds()
    except FileNotFoundError as exc:
        st.warning(f"{exc}")
        st.info("Ejecuta `python collectors/apifootball_collector.py --upcoming` para obtener odds actuales.")
        return

    collected_at = raw_odds.get("collected_at", "desconocido")
    picks, stats = pipeline.scan_picks(
        model, teams, raw_odds, min_ev=min_ev, bankroll=bankroll,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Picks encontrados", len(picks))
    c2.metric("Partidos con odds", stats["total_matches"])
    c3.metric("Sin modelo", stats["skipped_model"], help="Nombres de equipo no coinciden con la BD.")
    c4.metric("Equipos en modelo", len(teams))
    st.caption(
        f"Modelo: **{label}** · {n_matches} partidos entrenados · "
        f"Book: {config.TARGET_BOOKMAKER} · Odds recopiladas: {collected_at}"
    )

    if not picks:
        st.info("Sin value bets con los parámetros actuales. Prueba a bajar el EV mínimo.")
        return

    rows = []
    for p in picks:
        rows.append({
            "Liga":       p["league"],
            "Partido":    f"{p['home']} vs {p['away']}",
            "Fecha":      p["date"][:10] if p["date"] else "?",
            "Apuesta":    _BET_LABELS.get(p["bet_type"], p["bet_type"]),
            "Cuota":      p["odds"],
            "Modelo":     p["model_prob"],
            "Implícita":  p["implied_prob"],
            "Edge":       p["edge"],
            "EV":         p["ev"],
            "Stake":      p["stake"],
            "% Bankroll": p["pct_bankroll"],
        })
    df = pd.DataFrame(rows)

    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Modelo":     st.column_config.NumberColumn(format="percent"),
            "Implícita":  st.column_config.NumberColumn(format="percent"),
            "Edge":       st.column_config.NumberColumn(format="percent"),
            "EV":         st.column_config.NumberColumn(format="percent"),
            "% Bankroll": st.column_config.NumberColumn(format="percent"),
            "Cuota":      st.column_config.NumberColumn(format="%.2f"),
            "Stake":      st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.caption(
        "Modelo / Implícita / Edge / EV / % Bankroll se muestran en porcentaje. "
        "Stake calculado con Kelly fraccionado (¼)."
    )


# ---------------------------------------------------------------------------
# Section: Rendimiento (backtest)
# ---------------------------------------------------------------------------

def _render_performance() -> None:
    st.header("📈 Rendimiento (backtest walk-forward)")

    data = _load_backtest()
    if data is None:
        st.warning(f"No se encontró {_BACKTEST_PATH.name}.")
        st.info("Ejecuta `python models/real_backtester.py` para generarlo.")
        return

    s = data["summary"]

    c1, c2, c3 = st.columns(3)
    c1.metric("ROI", f"{s['roi']:+.1%}")
    c1.metric("Bankroll final", f"{s['final_bankroll']:.0f}",
              delta=f"{s['final_bankroll'] - s['initial_bankroll']:+.0f}")
    c2.metric("Win rate", f"{s['win_rate']:.1%}")
    c2.metric("Total picks", f"{s['total_picks']:,}")
    c3.metric("Brier Score", f"{s['brier_score']:.4f}")
    c3.metric("Max drawdown", f"{s.get('max_drawdown', 0):.1%}")
    st.caption(
        f"Bankroll inicial: {s['initial_bankroll']:.0f} · "
        f"Total apostado: {s['total_staked']:,.0f} · "
        f"PnL: {s['total_pnl']:+,.0f} · "
        f"Picks/semana: {s['picks_per_week']:.1f}"
    )

    # --- Bankroll history ---
    st.subheader("Evolución del bankroll")
    bh = pd.DataFrame(data["bankroll_history"]).set_index("week")
    st.line_chart(bh, y="bankroll", height=320)

    # --- PnL by league / bet type ---
    col_l, col_b = st.columns(2)

    with col_l:
        st.subheader("PnL por liga")
        pnl_league = {
            _LEAGUE_BY_ID.get(int(k), k): v
            for k, v in data["pnl_by_league"].items()
        }
        st.bar_chart(pd.Series(pnl_league, name="PnL").sort_values(), height=300)

    with col_b:
        st.subheader("PnL por tipo de apuesta")
        pnl_bet = {
            _BET_LABELS.get(k, k): v
            for k, v in data["pnl_by_bet_type"].items()
        }
        st.bar_chart(pd.Series(pnl_bet, name="PnL"), height=300)

    # --- Picks table ---
    st.subheader("Historial de picks")
    picks = pd.DataFrame(data["picks"])
    picks["league"] = picks["league_id"].map(_LEAGUE_BY_ID).fillna(picks["league_id"])
    picks["bet"] = picks["bet_type"].map(_BET_LABELS).fillna(picks["bet_type"])
    picks["won"] = picks["won"].astype(bool)

    ligas = ["Todas"] + sorted(picks["league"].unique().tolist())
    fc1, fc2 = st.columns([1, 1])
    liga_sel = fc1.selectbox("Liga", ligas)
    only_won = fc2.checkbox("Solo aciertos", value=False)

    view = picks
    if liga_sel != "Todas":
        view = view[view["league"] == liga_sel]
    if only_won:
        view = view[view["won"] == 1]

    show = view[[
        "week", "date", "league", "home", "away", "bet",
        "odds", "model_prob", "ev", "stake", "won", "pnl", "bankroll",
    ]].rename(columns={
        "week": "Sem", "date": "Fecha", "league": "Liga",
        "home": "Local", "away": "Visitante", "bet": "Apuesta",
        "odds": "Cuota", "model_prob": "Modelo", "ev": "EV",
        "stake": "Stake", "won": "Ganó", "pnl": "PnL", "bankroll": "Bankroll",
    })

    st.caption(f"{len(show):,} picks")
    st.dataframe(
        show,
        hide_index=True,
        use_container_width=True,
        height=420,
        column_config={
            "Modelo":   st.column_config.NumberColumn(format="percent"),
            "EV":       st.column_config.NumberColumn(format="percent"),
            "Cuota":    st.column_config.NumberColumn(format="%.2f"),
            "Stake":    st.column_config.NumberColumn(format="%.2f"),
            "PnL":      st.column_config.NumberColumn(format="%+.2f"),
            "Bankroll": st.column_config.NumberColumn(format="%.1f"),
            "Ganó":     st.column_config.CheckboxColumn(),
        },
    )


# ---------------------------------------------------------------------------
# Section: Análisis (dossier por partido)
# ---------------------------------------------------------------------------

_RESULT_EMOJI = {"V": "🟢", "E": "🟡", "D": "🔴"}


# load_upcoming / devig2 / signals viven ahora en models.analysis_builder
# (puros, sin streamlit) para que refresh.py pueda precalcular el análisis.


def _dist_str(dic: dict) -> str:
    return " / ".join(f"{k} {v:.0%}" for k, v in dic.items())


def _render_form_col(label: str, f: dict, pts10: int) -> None:
    st.markdown(f"**{label}**")
    if f["n"] == 0:
        st.caption("sin datos")
        return
    chips = " ".join(_RESULT_EMOJI.get(r, "") + r for r in f["results"])
    st.markdown(chips)
    st.caption(
        f"Goles: {f['gf']} a favor / {f['ga']} en contra (media)  ·  "
        f"Puntos: {f['points']}/{f['n']*3} (últimos {f['n']}) · {pts10}/30 (últimos 10)"
    )


_STAT_METRICS = [
    ("Goles", "goals"), ("xG", "xg"), ("Córners", "corners"),
    ("Tiros a puerta", "sot"), ("Tiros totales", "shots"),
    ("Tarjetas", "cards"), ("Atajadas GK", "saves"),
]


def _stat_cell(s: dict, key: str) -> str:
    f, ag = s.get(f"{key}_for"), s.get(f"{key}_against")
    if f is None and ag is None:
        return "—"
    fs = f"{f:.1f}" if f is not None else "—"
    ags = f"{ag:.1f}" if ag is not None else "—"
    return f"{fs} / {ags}"


def _render_stats_table(d: dict) -> None:
    """Promedios rodantes últimos 5 (a favor / en contra) — el método del usuario."""
    home, away = d["home"], d["away"]
    h5, a5 = d["stats"]["home5"], d["stats"]["away5"]
    if not (h5["n"] or a5["n"]):
        return
    has = h5.get("has_stats") or a5.get("has_stats")
    st.markdown("**📊 Promedios últimos 5** (a favor / en contra)")
    rows = []
    for lab, key in _STAT_METRICS:
        if key != "goals" and not has:
            continue
        rows.append({"Métrica": lab, f"🏠 {home}": _stat_cell(h5, key),
                     f"✈️ {away}": _stat_cell(a5, key)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    if not has:
        st.caption("Esta liga no publica córners/tiros/tarjetas — solo goles disponibles.")


def _pair(f, a) -> str:
    fs = f"{f:g}" if (f is not None and pd.notna(f)) else "—"
    ags = f"{a:g}" if (a is not None and pd.notna(a)) else "—"
    return f"{fs} / {ags}"


def _render_team_detail(det: list, label: str) -> None:
    st.markdown(f"**Últimos {len(det)} — {label}**  ·  columnas = a favor / en contra")
    rows = []
    for r in det:
        rows.append({
            "Fecha": r["Fecha"], "Rival": r["Rival"], "Marc.": r["Marcador"],
            "Res": _RESULT_EMOJI.get(r["Res"], "") + r["Res"],
            "Córners": _pair(r["Córners"], r["Córners_c"]),
            "Tiros P": _pair(r["TirosP"], r["TirosP_c"]),
            "Tarj": f"{r['Tarj']:g}" if (r["Tarj"] is not None and pd.notna(r["Tarj"])) else "—",
            "xG": _pair(r["xG"], r["xG_c"]),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 column_config={"Res": st.column_config.TextColumn(width="small")})


def _render_corner_signal(u: dict) -> None:
    """Córners: esperado combinado (ataque+defensa) vs línea real, con σ y
    etiqueta de confianza. Informativo — aún sin backtest de líneas históricas."""
    ca = u["_d"]["stats"].get("corners")
    cl = u.get("corners")
    if not ca and not cl:
        return
    if ca is None:
        if cl:
            st.caption(f"⚽ Córners — línea de la casa {cl['line']} (faltan stats para proyectar)")
        return

    proj, std, conf = ca["proj"], ca["std"], ca["confidence"]
    n_txt = f"n={ca['n_home']}/{ca['n_away']}"
    sigma_txt = f"{std}" if std is not None else "n/a"

    if cl is not None:
        line = cl["line"]
        gap = round(proj - line, 1)
        # notable solo con dispersión real (σ>0) y gap que la supere — evita
        # marcar señal falsa cuando la muestra es degenerada (σ=0).
        notable = std is not None and std > 0 and abs(gap) >= 0.5 * std
        if notable:
            lean = "Over" if gap > 0 else "Under"
            st.markdown(
                f"**🚩 Córners** — esperado combinado **{proj}** vs línea **{line}** "
                f"→ **{lean}** (gap {gap:+.1f}, supera 0.5σ)  ·  cuotas O {cl['over']} / U {cl['under']}"
            )
        else:
            st.markdown(
                f"Córners — esperado combinado {proj} vs línea {line} "
                f"(gap {gap:+.1f}: dentro de rango normal, sin inclinación clara)  ·  "
                f"cuotas O {cl['over']} / U {cl['under']}"
            )
        st.caption(
            f"σ muestra ≈ {sigma_txt} · {n_txt} · confianza {conf} · "
            "informativo (sin backtest de líneas históricas todavía)"
        )
    else:
        st.markdown(f"Córners — esperado combinado {proj} (la casa no publica línea aquí)")
        st.caption(f"σ ≈ {sigma_txt} · {n_txt} · confianza {conf} · informativo")


_TREND_GROUPS = [
    ("Córners individuales",   "corners_ind"),
    ("Córners totales",        "corners_tot"),
    ("Tiros a puerta indiv.",  "sot_ind"),
    ("Tiros a puerta totales", "sot_tot"),
]


def _tr_mark(hits: int, n: int) -> str:
    if not n:
        return ""
    r = hits / n
    return " 🔥" if r >= 0.8 else (" ❄️" if r <= 0.2 else "")


def _tr_cell(hn) -> str:
    if hn is None:
        return "—"
    h, n = hn
    return f"{h}/{n}{_tr_mark(h, n)}"


def _trend_highlights(trends: dict, thr_map: dict) -> list:
    """Rachas fuertes (u10) por grupo: el umbral MÁS ALTO que supera de forma
    fiable (≥80%) y el MÁS BAJO bajo el que se queda (≤20%). Máx 2 por grupo,
    sin sesgo por orden de lista."""
    from models.match_analysis import is_strong
    out = []
    for label, key in _TREND_GROUPS:
        rates = trends[key][10]
        overs = [t for t in thr_map[key]
                 if rates[t] and is_strong(*rates[t]) and rates[t][0] / rates[t][1] >= 0.8]
        unders = [t for t in thr_map[key]
                  if rates[t] and is_strong(*rates[t]) and rates[t][0] / rates[t][1] <= 0.2]
        if overs:
            t = max(overs)
            h, n = rates[t]
            out.append(f"🔥 {label}: supera **+{t}** en {h}/{n}")
        if unders:
            t = min(unders)
            h, n = rates[t]
            out.append(f"❄️ {label}: por debajo de **+{t}** en {n - h}/{n}")
    return out


# Tendencias booleanas de goles/tiempos (rate simple, no por umbral).
_GOAL_TREND_LABELS = [
    ("Marca +1.5 goles",        "goals_for15"),
    ("Recibe +1.5 goles",       "goals_ag15"),
    ("Marcó en ambos tiempos",  "scored_both_halves"),
    ("Ganó ambos tiempos",      "won_both_halves"),
]


def _bar(hn, width: int = 10) -> str:
    """Barra de texto ████░░ h/n para una racha (h, n). '—' si no hay muestra."""
    if not hn or not hn[1]:
        return "—"
    h, n = hn
    filled = round(width * h / n)
    return "█" * filled + "░" * (width - filled) + f"  {h}/{n}"


def _goal_highlights(trends: dict) -> list:
    """Rachas fuertes (u10) de goles/tiempos como strings cortos."""
    from models.match_analysis import is_strong
    out = []
    for label, key in _GOAL_TREND_LABELS:
        hn = trends[key][10]
        if hn and is_strong(*hn):
            h, n = hn
            mark = "🔥" if h / n >= 0.8 else "❄️"
            out.append(f"{mark} {label}: {h}/{n}")
    return out


def _render_result_rates(th: dict, ta: dict, home: str, away: str) -> None:
    """BTTS / Over 2.5 / portería a cero de cada equipo (últimos 10) — junto a
    los promedios. Salen del marcador, así que aplican a todas las ligas."""
    if not (th["n_goals"] or ta["n_goals"]):
        return

    def pct(tr, key):
        hn = tr[key][10]
        return f"{round(100 * hn[0] / hn[1])}%" if hn else "—"

    st.markdown("**📌 Resultados recientes (últimos 10)**")
    rows = ["| Equipo | BTTS | Over 2.5 | Portería a cero |", "|:--|:--|:--|:--|"]
    for emoji, name, tr in (("🏠", home, th), ("✈️", away, ta)):
        safe = str(name).replace("|", "\\|")   # un '|' partiría la fila markdown
        rows.append(
            f"| {emoji} **{safe}** | {pct(tr, 'btts')} | "
            f"{pct(tr, 'over25')} | {pct(tr, 'clean_sheet')} |"
        )
    st.markdown("\n".join(rows))
    st.caption(
        "% de los últimos 10 partidos del equipo · BTTS = ambos marcaron · "
        "Over 2.5 = +3 goles totales · Portería a cero = el equipo no recibió gol."
    )


def _thr_table(trends: dict, key: str, thrs) -> str:
    """Tabla markdown de un grupo de umbrales: Umbral | u10 | u5. Nativa (respeta
    el tema, sin el recuadro ni el scroll de un widget por grupo)."""
    rows = ["| Umbral | u10 | u5 |", "|:--|:--|:--|"]
    for t in thrs:
        u10 = _tr_cell(trends[key][10].get(t))
        u5 = _tr_cell(trends[key][5].get(t))
        rows.append(f"| **+{t}** | {u10} | {u5} |")
    return "\n".join(rows)


def _render_team_trends_full(trends: dict, label: str, thr_map: dict) -> None:
    st.markdown(
        f"**{label}**  ·  goles: {trends['n_goals']} · córners: "
        f"{trends['n_corners']} · tiros: {trends['n_sot']} partidos con datos"
    )
    # Goles / tiempos (todas las ligas) — con barra visual (u10)
    st.caption("Goles / tiempos (u10)")
    for glabel, key in _GOAL_TREND_LABELS:
        st.markdown(f"`{_bar(trends[key][10])}`  {glabel}")
    # Córners / tiros (donde hay stats) — una tabla compacta por grupo.
    if trends["n_corners"] or trends["n_sot"]:
        st.caption("Córners / tiros — por umbral (u10 · u5)")
        for glabel, key in _TREND_GROUPS:
            thrs = thr_map[key]
            if not any(trends[key][10].get(t) for t in thrs):
                continue   # grupo sin datos (p. ej. tiros donde la liga no publica)
            st.markdown(f"**{glabel}**")
            st.markdown(_thr_table(trends, key, thrs))


def _team_strong_trends(tr: dict, cap: int = 4) -> list:
    """Rachas notables (u10) de un equipo — goles/tiempos + córners/tiros."""
    from models.match_analysis import TREND_THRESHOLDS
    return (_goal_highlights(tr) + _trend_highlights(tr, TREND_THRESHOLDS))[:cap]


def _render_trends(u: dict, th: dict, ta: dict) -> None:
    """Tablas completas de tendencias de ambos equipos, lado a lado. Las rachas
    fuertes ya se resumen en el panel de arriba; acá está todo el detalle."""
    from models.match_analysis import TREND_THRESHOLDS
    home, away = u["_d"]["home"], u["_d"]["away"]
    if not (th["n_goals"] or ta["n_goals"]):
        st.info("Sin datos de tendencias para este partido.")
        return

    st.caption(
        "Descriptivo ('8/10'), **no** probabilidad. 🔥 = pasa seguido (≥80%) · "
        "❄️ = pasa rara vez (≤20%) — es **frecuencia**, no bueno/malo (ej. 'recibe "
        "+1.5 goles 🔥' = concede seguido). Ventanas fijas. Goles/tiempos en todas "
        "las ligas; córners/tiros donde la liga publica stats."
    )
    cl = u.get("corners")
    if cl:
        st.caption(
            f"📏 Línea de córners de la casa: **{cl['line']}** "
            f"(O {cl['over']} / U {cl['under']}) — compará con 'Córners totales'."
        )

    c1, c2 = st.columns(2)
    with c1:
        _render_team_trends_full(th, f"🏠 {home}", TREND_THRESHOLDS)
    with c2:
        _render_team_trends_full(ta, f"✈️ {away}", TREND_THRESHOLDS)


def _low_season(u: dict) -> bool:
    """True si algún equipo tiene pocos partidos de la temporada en curso — el
    modelo de goles arrastra forma de la temporada anterior."""
    from models.match_analysis import GOALS_MIN_SEASON_GAMES
    gi = u["_d"].get("goal_internals")
    if not gi:
        return False
    for g in (gi.get("home_season_games"), gi.get("away_season_games")):
        if g is not None and g < GOALS_MIN_SEASON_GAMES:
            return True
    return False


def _render_goal_internals(u: dict) -> None:
    """Base del Poisson de goles: λ y multiplicadores ataque/defensa (como
    tarjetas), con aviso si la muestra de la temporada actual es chica.
    Transparencia, no cambia el cálculo."""
    from models.match_analysis import GOALS_MIN_SEASON_GAMES
    gi = u["_d"].get("goal_internals")
    if not gi:
        return
    home, away = u["_d"]["home"], u["_d"]["away"]
    hg, ag = gi.get("home_season_games"), gi.get("away_season_games")

    st.markdown("#### ⚙️ Base del modelo de goles (Over/Under · BTTS)")
    for name, g in ((home, hg), (away, ag)):
        if g is not None and g < GOALS_MIN_SEASON_GAMES:
            st.warning(
                f"⚠️ **{name}: {g} partido{'s' if g != 1 else ''} esta temporada** "
                "— el modelo arrastra forma de la temporada anterior; su confianza "
                "en los mercados de goles puede estar inflada por falta de muestra "
                "actual."
            )

    c1, c2 = st.columns(2)
    for col, emoji, name, atk, dfn, lam, g in (
        (c1, "🏠", home, gi["home_attack"], gi["home_defense"], gi["lambda_home"], hg),
        (c2, "✈️", away, gi["away_attack"], gi["away_defense"], gi["lambda_away"], ag),
    ):
        with col:
            st.markdown(f"**{emoji} {name}**")
            m1, m2, m3 = st.columns(3)
            m1.metric("λ goles esp.", f"{lam:.2f}")
            m2.metric("Ataque", f"{atk:.2f}×")
            m3.metric("Defensa", f"{dfn:.2f}×")
            gtxt = "—" if g is None else str(g)
            warn = " ⚠️" if (g is not None and g < GOALS_MIN_SEASON_GAMES) else ""
            st.caption(f"Partidos esta temporada: **{gtxt}**{warn}")

    total = gi["lambda_home"] + gi["lambda_away"]
    ha, ag_ = gi.get("home_advantage"), gi.get("avg_goals")
    factores = (f" · ventaja local ×{ha} · media liga {ag_}"
                if ha is not None and ag_ is not None else "")
    st.caption(
        f"λ = goles esperados = ataque × defensa rival × (ventaja local, solo "
        f"local) × media liga{factores}. Multiplicador **1.00× = promedio de la "
        f"liga** (más bajo = mejor defensa / peor ataque). **Total esperado ≈ "
        f"{total:.2f} goles** → el Over/Under y el BTTS salen de esta base con un "
        "ajuste Dixon-Coles (por eso no coinciden exacto con λ). Diagnóstico: "
        "expone la base, no la cambia."
    )


def _render_glance(u: dict, th: dict, ta: dict) -> None:
    """Panel 'de un vistazo' — lo esencial arriba sin scroll: señal, λ, córners,
    aviso de muestra y las rachas notables de cada equipo."""
    d = u["_d"]
    home, away = d["home"], d["away"]

    m1, m2, m3 = st.columns(3)
    bl = _best_lean(u["_sig"]) if (d["known"] and u["_sig"]) else None
    if bl:
        mkt, side, mp, kp, gap = bl
        m1.metric(f"Señal · {mkt}", f"{side} {mp:.0%}",
                  f"casa {kp:.0%} · +{gap:.0%}", delta_color="off")
    else:
        m1.metric("Señal del modelo", "— (sin modelo)")
    gi = d.get("goal_internals")
    m2.metric("Goles esperados (λ)",
              f"{gi['lambda_home'] + gi['lambda_away']:.2f}" if gi else "—",
              help="Total de goles que espera el modelo; de acá sale el Over/Under.")
    ca, cl = d["stats"].get("corners"), u.get("corners")
    if ca:
        m3.metric("Córners proyectados", f"{ca['proj']}",
                  f"línea {cl['line']}" if cl else "sin línea", delta_color="off")
    else:
        m3.metric("Córners proyectados", "—")

    if _low_season(u):
        st.caption("⚠️ Muestra de temporada chica — la base de goles arrastra "
                   "forma vieja (detalle en pestaña Resumen ⚙️).")

    g1, g2 = st.columns(2)
    for col, name, tr, emoji in ((g1, home, th, "🏠"), (g2, away, ta, "✈️")):
        with col:
            hi = _team_strong_trends(tr)
            if hi:
                st.markdown(f"{emoji} **{name}** — rachas notables:")
                for x in hi:
                    st.markdown(f"- {x}")
            else:
                st.markdown(f"{emoji} **{name}** — sin rachas notables")
    st.caption("Diagnóstico, **no** apuesta. El modelo no le gana al mercado en "
               "backtest y no está calibrado — el juicio es tuyo.")


def _render_detail(u: dict, matches_df, stats_df) -> None:
    from models.match_analysis import team_recent_detail, h2h_detail, team_trends
    d = u["_d"]
    home, away = d["home"], d["away"]
    before = u["dt"]
    th = team_trends(stats_df, matches_df, home, before)
    ta = team_trends(stats_df, matches_df, away, before)

    st.markdown(f"### {home}  vs  {away}")
    cty = f" ({u['country']})" if u.get("country") else ""
    st.caption(f"{u['league']}{cty} · {d['date']} · {_hhmm(u['dt'])} · cuotas: {u['book']}")

    _render_glance(u, th, ta)

    tab_res, tab_tend, tab_hist = st.tabs(["📊 Resumen", "📈 Tendencias", "📋 Historial"])

    with tab_res:
        if d["known"] and u["_sig"]:
            st.markdown("**📊 Modelo vs mercado — dónde más difieren** (diagnóstico)")
            rows = ["| Mercado | Elección modelo | Prob. modelo | Prob. mercado | Δ máx |",
                    "|:--|:--|:--|:--|--:|"]
            for s in sorted(u["_sig"], key=lambda x: x["delta"], reverse=True):
                pick = max(s["model"], key=s["model"].get)
                odds = _lean_odds(u, s["mkt"], pick)
                lean = f"{pick} {s['model'][pick]:.0%}"
                if odds is not None:
                    lean += f" · @{odds:.2f}"
                rows.append(
                    f"| {s['mkt']} | **{lean}** | {_dist_str(s['model'])} | "
                    f"{_dist_str(s['market'])} | {s['delta']:.2%} |"
                )
            st.markdown("\n".join(rows))
            cs = " · ".join(f"{c['score']} ({c['prob']:.0%})" for c in d["correct_scores"])
            st.caption(f"Marcadores más probables del modelo: {cs}")
        _render_corner_signal(u)
        _render_goal_internals(u)

    with tab_tend:
        _render_trends(u, th, ta)
        _render_result_rates(th, ta, home, away)

    with tab_hist:
        _render_stats_table(d)
        st.markdown("#### 📋 Últimos partidos (partido a partido)")
        dh = team_recent_detail(stats_df, matches_df, home, before, 6)
        da = team_recent_detail(stats_df, matches_df, away, before, 6)
        h1, h2 = st.columns(2)
        with h1:
            if dh:
                _render_team_detail(dh, f"🏠 {home}")
            else:
                st.caption(f"🏠 {home}: sin detalle disponible")
        with h2:
            if da:
                _render_team_detail(da, f"✈️ {away}")
            else:
                st.caption(f"✈️ {away}: sin detalle disponible")

        hh = h2h_detail(stats_df, matches_df, home, away, before, 6)
        if hh:
            st.markdown("#### 🤝 Enfrentamientos directos (detalle)")
            st.dataframe(pd.DataFrame(hh), hide_index=True, use_container_width=True)

        rh, ra = d["rest"]["home"], d["rest"]["away"]
        if rh["days_rest"] is not None or ra["days_rest"] is not None:
            st.caption(
                f"😴 Descanso — {home}: {rh['days_rest']}d ({rh['games_14d']} en 14d)  ·  "
                f"{away}: {ra['days_rest']}d ({ra['games_14d']} en 14d)"
            )


@st.cache_resource(max_entries=2, show_spinner="Cargando análisis…")
def _analyzed_cached(analysis_mtime: float):
    """Carga el análisis precalculado por refresh.py (instantáneo). Si el
    artefacto falta, lo arma en vivo (lento) como respaldo. Cacheado por mtime
    del .pkl: cuando refresh.py lo reescribe, el mtime cambia y la caché se
    invalida sola (el dashboard recoge el análisis nuevo sin reiniciar)."""
    loaded = pipeline.load_analysis_artifact()
    if loaded is not None:
        return loaded  # (analyzed, label, n_matches)

    # Respaldo en vivo — solo si nunca se corrió refresh.py.
    from models.analysis_builder import load_upcoming, build_analysis
    model, teams, label, n_matches = _get_live_model()
    matches_df = _get_matches()
    xg_df = _get_xg()
    stats_df = _get_stats_df()
    upcoming = build_analysis(model, matches_df, xg_df, stats_df, load_upcoming())
    return upcoming, label, n_matches


def _get_analyzed():
    mtime = (pipeline.ANALYSIS_PATH.stat().st_mtime
             if pipeline.ANALYSIS_PATH.exists() else 0.0)
    return _analyzed_cached(mtime)


_WEEKDAYS_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


def _day_label(date_str: str) -> str:
    """'Hoy 2026-08-16' / 'Mañana …' / 'Sábado …' según la fecha (hora local)."""
    d = pd.Timestamp(date_str).date()
    delta = (d - datetime.now().date()).days
    if delta == 0:
        prefix = "Hoy"
    elif delta == 1:
        prefix = "Mañana"
    elif delta == -1:
        prefix = "Ayer"
    else:
        prefix = _WEEKDAYS_ES[d.weekday()].capitalize()
    return f"{prefix} · {date_str}"


def _hhmm(dt) -> str:
    """Hora local del partido como 'HH:MM' (dt ya viene en hora local)."""
    return pd.Timestamp(dt).strftime("%H:%M") if dt is not None else "—"


def _day_selectbox(items: list, key: str, container=st, reverse: bool = False) -> str:
    """Selectbox 'Día' con las fechas distintas de items (+ 'Todos'). Devuelve la
    selección ('Todos' o 'YYYY-MM-DD'). reverse=True ordena de más reciente.

    Por defecto arranca en HOY si hay partidos hoy; si no, en 'Todos'. Tras la
    primera carga respeta lo que elija el usuario (Streamlit persiste por key)."""
    days = ["Todos"] + sorted({x["date"] for x in items}, reverse=reverse)
    today = str(datetime.now().date())
    idx = days.index(today) if today in days else 0
    return container.selectbox(
        "Día", days, index=idx, key=key,
        format_func=lambda d: "Todos los días" if d == "Todos" else _day_label(d))


def _lean_odds(u: dict, mkt: str, side: str):
    """Cuota decimal de la casa para el lado señalado (o None si no está)."""
    o1 = u.get("odds_1x2") or {}
    ou = u.get("ou25") or {}
    bt = u.get("btts") or {}
    table = {
        "1X2": {"Local": o1.get("home_win"), "Empate": o1.get("draw"),
                "Visit.": o1.get("away_win")},
        "Goles O/U 2.5": {"Over": ou.get("over"), "Under": ou.get("under")},
        "BTTS": {"Sí": bt.get("yes"), "No": bt.get("no")},
    }
    return table.get(mkt, {}).get(side)


def _best_lean(sigs: list) -> tuple | None:
    """Entre TODOS los mercados del partido, el lado que el modelo más FAVORECE
    frente a la casa (máxima diferencia positiva modelo − casa). Ese es el lado
    accionable: 'el modelo le da más probabilidad que la casa'.

    Devuelve (mercado, lado, prob_modelo, prob_casa, gap) o None."""
    best = None
    for s in sigs:
        for side in s["model"]:
            gap = s["model"][side] - s["market"][side]
            if best is None or gap > best[4]:
                best = (s["mkt"], side, s["model"][side], s["market"][side], gap)
    return best


def _render_analysis() -> None:
    st.header("🔍 Análisis por partido — triaje diario")
    st.caption(
        "Diagnóstico, **no** sugerencias de apuesta. Los partidos se ordenan por "
        "dónde el modelo (sin calibrar) más se aleja del mercado. El modelo no le "
        "gana al mercado en backtest — el juicio es tuyo."
    )

    try:
        upcoming, label, n_matches = _get_analyzed()
    except Exception as exc:
        st.error(f"No se pudo analizar: {exc}")
        return

    if not upcoming:
        st.info("No hay partidos próximos con odds. Corre "
                "`python collectors/apifootball_collector.py --upcoming` para recolectarlos.")
        return

    matches_df = _get_matches()   # cacheado — para la vista de detalle
    stats_df = _get_stats_df()    # cacheado

    c1, c2, c3, c4, c5 = st.columns([2, 2, 1.4, 1.2, 1.6])
    leagues = ["Todas"] + sorted({u["league"] for u in upcoming})
    sel = c1.selectbox("Liga", leagues)
    sel_day = _day_selectbox(upcoming, "ana_dia", c2)
    # Hora primero → orden por defecto (no confiamos ciego en la discrepancia).
    orden = c3.selectbox("Ordenar por", ["Hora", "Discrepancia"], key="ana_orden")
    solo_prox = c4.checkbox("Solo próximos", value=True,
                            help="Oculta los partidos cuya hora ya pasó (ya empezaron o terminaron).")
    only_signals = c5.checkbox("Solo con discrepancia ≥ 5%", value=False)

    now_local = pd.Timestamp(datetime.now())
    view = [u for u in upcoming
            if (sel == "Todas" or u["league"] == sel)
            and (sel_day == "Todos" or u["date"] == sel_day)
            and (not solo_prox or u["dt"] > now_local)]
    # (partido, lean) donde lean = (mercado, lado, prob_modelo, prob_casa, gap)
    known = [(u, _best_lean(u["_sig"]))
             for u in view if u["_d"]["known"] and u["_sig"]]
    known = [(u, bl) for u, bl in known if bl is not None]
    if only_signals:
        known = [(u, bl) for u, bl in known if bl[4] >= 0.05]
    if orden == "Hora":
        known.sort(key=lambda p: p[0]["dt"])            # cronológico, próximos primero
    else:
        known.sort(key=lambda p: p[1][4], reverse=True)  # mayor discrepancia primero
    unknown = [u for u in view if not (u["_d"]["known"] and u["_sig"])]

    if known:
        rows = []
        for u, (mkt, side, mp, kp, gap) in known:
            partido = f"{u['home']} vs {u['away']}"
            if _low_season(u):
                partido += " ⚠️"
            rows.append({
                "Partido": partido,
                "Liga": u["league"],
                "Fecha": u["date"],
                "Hora": _hhmm(u["dt"]),
                "Mercado": mkt,
                "Señal": side,
                "Cuota": _lean_odds(u, mkt, side),
                "Modelo": mp,
                "Casa": kp,
                "Δ": gap,
            })
        event = st.dataframe(
            pd.DataFrame(rows), hide_index=True, use_container_width=True,
            on_select="rerun", selection_mode="single-row",
            key=f"ana_tbl_{sel}_{sel_day}_{orden}_{only_signals}_{solo_prox}",
            column_config={
                "Cuota": st.column_config.NumberColumn(
                    format="%.2f", help="Cuota decimal de la casa para el lado señalado"),
                "Modelo": st.column_config.NumberColumn(
                    format="percent", help="Prob. del modelo para ese lado"),
                "Casa": st.column_config.NumberColumn(
                    format="percent", help="Prob. implícita de la casa (sin vig) para ese lado"),
                "Δ": st.column_config.NumberColumn(
                    format="percent", help="Cuánto MÁS probable lo ve el modelo que la casa"),
            },
        )
        st.caption(
            f"{len(known)} partidos · **Señal** = el lado (Over/Under, Sí/No, "
            "Local/Empate/Visit.) al que el modelo le da MÁS probabilidad que la "
            "casa; **Modelo/Casa** = probabilidad de ese lado, **Δ** = cuánto más. "
            "**⚠️** = algún equipo con pocos partidos esta temporada (el modelo "
            "arrastra forma vieja — pesa más en Over/Under y BTTS que en 1X2; ver "
            f"detalle). Diagnóstico, no apuesta — modelo sin calibrar ({n_matches:,} partidos)."
        )

        sel_rows = event.selection.rows if (event and event.selection) else []
        idx = sel_rows[0] if (sel_rows and sel_rows[0] < len(known)) else 0

        st.divider()
        u_sel = known[idx][0]
        st.subheader(f"🔎 Detalle — {u_sel['home']} vs {u_sel['away']}")
        st.caption("👆 Hacé clic en una fila de la tabla para ver el detalle de otro partido.")
        _render_detail(u_sel, matches_df, stats_df)
    else:
        st.info("Ningún partido próximo con modelo + mercado para el filtro actual.")

    if unknown:
        with st.expander(f"⚠️ {len(unknown)} partidos sin datos de modelo (equipo no entrenado)"):
            for u in unknown:
                st.markdown(f"- {u['home']} vs {u['away']} · {u['league']}")


# ---------------------------------------------------------------------------
# Section: Resultados (partidos finalizados recientes)
# ---------------------------------------------------------------------------

def _render_results() -> None:
    st.header("✅ Resultados recientes")
    st.caption(
        "Partidos finalizados (últimos días) con el desenlace de tus mercados "
        "(Over/Under 2.5, BTTS) calculado del marcador real. Refresca con "
        "`python collectors/apifootball_collector.py --results`."
    )

    fin = _get_recent_finished(3)
    if not fin:
        st.info("No hay partidos finalizados recientes en la BD.")
        return

    c1, c2 = st.columns(2)
    leagues = ["Todas"] + sorted({f["league"] for f in fin})
    sel = c1.selectbox("Liga", leagues, key="res_liga")
    sel_day = _day_selectbox(fin, "res_dia", c2, reverse=True)
    view = [f for f in fin
            if (sel == "Todas" or f["league"] == sel)
            and (sel_day == "Todos" or f["date"] == sel_day)]

    rows = []
    for f in view:
        rows.append({
            "Fecha": f["date"],
            "Hora": _hhmm(f["dt"]),
            "Partido": f"{f['home']} vs {f['away']}",
            "Liga": f["league"],
            "Marcador": f["score"],
            "Over 2.5": "✅ Over" if f["over25"] else "🔻 Under",
            "BTTS": "✅ Sí" if f["btts"] else "🔻 No",
            "Córners": f"{f['corners']:g}" if f["corners"] is not None else "—",
            "Tarjetas": f"{f['cards']:g}" if f["cards"] is not None else "—",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=560)
    st.caption(
        f"{len(view)} partidos finalizados · Córners/Tarjetas = total del partido "
        "(donde la liga publica stats)"
    )


# ---------------------------------------------------------------------------
# Section: Predicciones (bitácora de validación del modelo)
# ---------------------------------------------------------------------------

_1X2_LBL = {"home": "Local", "draw": "Empate", "away": "Visit."}


def _mark(hit) -> str:
    return "✅" if hit is True else ("❌" if hit is False else "—")


def _graded_cell(cell, odds=None, lblmap=None, line=None) -> str:
    """'Local ✅ @2.00' / 'Over 10.5 ❌ @1.85' / '—' según la llamada del modelo,
    si acertó y la cuota sellada (si está disponible). `line` añade la línea de
    la casa junto al lado (para córners, donde el mercado ES la línea)."""
    call = cell["model"]
    if call is None:
        return "—"
    txt = lblmap.get(call, call) if lblmap else call
    if line is not None:
        txt = f"{txt} {line:g}"
    suffix = f" @{odds:.2f}" if odds is not None else ""
    return f"{txt} {_mark(cell['model_hit'])}{suffix}"


_MK_LBL = {"1x2": "1X2", "ou25": "O/U 2.5", "btts": "BTTS", "corners": "Córners"}


def _acc_str(hits: int, n: int) -> str:
    return f"{hits}/{n} ({hits / n:.0%})" if n else "—"


def _roi_str(pnl: float, bets: int) -> str:
    """ROI con staking plano de 1u: % por apuesta + P&L total + nº de apuestas
    (su propio denominador, que puede diferir del de acierto si faltan cuotas)."""
    return f"{pnl / bets:+.0%} ({pnl:+.1f}u · {bets})" if bets else "—"


def _summary_table(summary: dict, pnl: dict) -> "pd.DataFrame":
    rows = []
    tot_mh = tot_mn = tot_kh = tot_kn = 0
    tot_mp = tot_kp = 0.0
    tot_mb = tot_kb = 0
    for mk in ("1x2", "ou25", "btts", "corners"):
        s, p = summary[mk], pnl[mk]
        rows.append({
            "Mercado": _MK_LBL[mk],
            "Acierto modelo": _acc_str(s["model_hits"], s["model_n"]),
            "ROI modelo": _roi_str(p["model_pnl"], p["model_bets"]),
            "Acierto mercado": _acc_str(s["market_hits"], s["market_n"]),
            "ROI mercado": _roi_str(p["market_pnl"], p["market_bets"]),
        })
        tot_mh += s["model_hits"]; tot_mn += s["model_n"]
        tot_kh += s["market_hits"]; tot_kn += s["market_n"]
        tot_mp += p["model_pnl"]; tot_mb += p["model_bets"]
        tot_kp += p["market_pnl"]; tot_kb += p["market_bets"]
    rows.append({
        "Mercado": "**Total**",
        "Acierto modelo": _acc_str(tot_mh, tot_mn),
        "ROI modelo": _roi_str(tot_mp, tot_mb),
        "Acierto mercado": _acc_str(tot_kh, tot_kn),
        "ROI mercado": _roi_str(tot_kp, tot_kb),
    })
    return pd.DataFrame(rows)


def _render_predictions() -> None:
    st.header("📋 Predicciones del modelo")
    st.caption(
        "Bitácora de validación: lo que el modelo predijo **antes** de cada "
        "partido, con su cuota sellada. Al finalizar se califica contra el "
        "resultado real — acierto direccional (✅/❌) **y** ROI (¿le habría "
        "ganado a esa cuota?). Es **diagnóstico** con muestra aún chica, **no** "
        "un backtest cerrado ni una recomendación de apuesta."
    )

    graded, pending, summary, pnl, n_leaked = _get_predictions()
    n_clean = len(graded) - n_leaked   # calificados limpios (cuentan para el ROI)

    if not graded and not pending:
        st.info(
            "No hay predicciones registradas. Ejecuta `python refresh.py` para "
            "sellar el snapshot de los partidos próximos."
        )
        return

    # Nota de transparencia: snapshots contaminados por el bug de fuga (corregido).
    if n_leaked:
        st.warning(
            f"⚠️ **{n_leaked} predicciones excluidas del cálculo** — su snapshot se "
            "tomó *después* de iniciado el partido, así que el modelo se “predijo” a "
            "sí mismo con el resultado ya conocido (fuga). Inflaban el ROI/acierto de "
            "forma irreal. Bug de `load_upcoming()` corregido el **2026-08-16**; se "
            "marcan, no se borran — siguen abajo en Calificados, señaladas con 🚩."
        )

    # --- Resumen: acierto + ROI (modelo vs mercado) ---
    st.subheader("Acierto y ROI acumulados")
    if n_clean == 0:
        extra = (f" ({n_leaked} calificadas quedaron excluidas por la fuga)"
                 if n_leaked else "")
        st.info(
            f"Aún no hay partidos calificados *limpios* — {len(pending)} predicciones "
            f"esperando que se jueguen{extra}. El acierto y el ROI honestos "
            "aparecerán aquí cuando maduren los snapshots limpios."
        )
    else:
        st.dataframe(_summary_table(summary, pnl), hide_index=True,
                     use_container_width=True)
        st.caption(
            f"{n_clean} partidos calificados limpios. **ROI** = apostar 1u plana al "
            "lado que llama cada uno, a la cuota sellada antes del partido "
            "(gana → cuota−1, pierde → −1). 'Mercado' = apostar al favorito de la "
            "cuota, como referencia. Córners: solo modelo vs línea. **Total** = los "
            "4 mercados combinados (misma unidad, 1u por apuesta), para ver de un "
            "vistazo cómo va el modelo en general. "
            "⚠️ Muestra chica — es diagnóstico, **no** una recomendación de apuesta."
        )

    # --- Calificados ---
    if graded:
        st.subheader(f"Calificados ({len(graded)})")
        gsel_day = _day_selectbox(graded, "pred_grad_dia", reverse=True)
        gview = [x for x in graded if gsel_day == "Todos" or x["date"] == gsel_day]
        rows = []
        for x in gview:
            g, od = x["g"], x["odds"]
            ct = x["corners_total"]
            corner_real = f"{ct:g}" if ct is not None else "—"
            rows.append({
                "Fecha": x["date"],
                "Hora": _hhmm(x["dt"]),
                "Partido": ("🚩 " if x.get("leaked") else "")
                           + f"{x['home']} vs {x['away']}",
                "Marcador": x["score"],
                "1X2": _graded_cell(g["1x2"], od["1x2"], _1X2_LBL),
                "O/U 2.5": _graded_cell(g["ou25"], od["ou25"]),
                "BTTS": _graded_cell(g["btts"], od["btts"]),
                "Córners": _graded_cell(g["corners"], od["corners"],
                                        line=x.get("corner_line")),
                "C. total": corner_real,
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     use_container_width=True, height=420)
        st.caption(
            f"{len(gview)} de {len(graded)} calificados. Cada celda = lado más "
            "probable del modelo + si salió + la cuota sellada (@). 'C. total' = "
            "córners reales del partido (donde la liga publica stats). "
            "🚩 = snapshot contaminado (excluido del ROI/acierto de arriba)."
        )

    # --- Pendientes ---
    if pending:
        st.subheader(f"Pendientes ({len(pending)})")
        c1, c2 = st.columns(2)
        leagues = ["Todas"] + sorted({x["league"] for x in pending})
        sel = c1.selectbox("Liga", leagues, key="pred_liga")
        sel_day = _day_selectbox(pending, "pred_pend_dia", c2)
        view = [x for x in pending
                if (sel == "Todas" or x["league"] == sel)
                and (sel_day == "Todos" or x["date"] == sel_day)]

        def _at(o):
            return f" @{o:.2f}" if o is not None else ""

        rows = []
        for x in view:
            mc = x["mc"]
            # prob del lado llamado en 1X2 + su cuota sellada
            p1x2 = {"home": x["m_home"], "draw": x["m_draw"],
                    "away": x["m_away"]}.get(mc["1x2"])
            o1x2 = {"home": x.get("o_home"), "draw": x.get("o_draw"),
                    "away": x.get("o_away")}.get(mc["1x2"])
            c1x2 = (f"{_1X2_LBL[mc['1x2']]} {p1x2:.0%}{_at(o1x2)}"
                    if mc["1x2"] and p1x2 is not None else "—")
            if mc["ou25"] and x["m_over25"] is not None:
                p_ou = x["m_over25"] if mc["ou25"] == "Over" else 1 - x["m_over25"]
                o_ou = x.get("o_over25") if mc["ou25"] == "Over" else x.get("o_under25")
                cou = f"{mc['ou25']} {p_ou:.0%}{_at(o_ou)}"
            else:
                cou = "—"
            if mc["btts"] and x["m_btts_yes"] is not None:
                p_bt = x["m_btts_yes"] if mc["btts"] == "Sí" else 1 - x["m_btts_yes"]
                o_bt = x.get("o_btts_yes") if mc["btts"] == "Sí" else x.get("o_btts_no")
                cbt = f"{mc['btts']} {p_bt:.0%}{_at(o_bt)}"
            else:
                cbt = "—"
            if x["corner_proj"] is not None and x["corner_line"] is not None:
                # El modelo llama Over si la proyección supera la línea, Under si no;
                # mostramos ese lado con su cuota sellada + la proj vs línea de fondo.
                over = x["corner_proj"] > x["corner_line"]
                side = "Over" if over else "Under"
                o_co = x.get("o_corner_over") if over else x.get("o_corner_under")
                cco = (f"{side}{_at(o_co)} · "
                       f"{x['corner_proj']:.1f} vs {x['corner_line']:g}")
            elif x["corner_proj"] is not None:
                cco = f"{x['corner_proj']:.1f} (sin línea)"
            else:
                cco = "—"
            rows.append({
                "Fecha": x["date"],
                "Hora": _hhmm(x["dt"]),
                "Partido": f"{x['home']} vs {x['away']}",
                "Liga": x["league"],
                "1X2 modelo": c1x2,
                "O/U 2.5": cou,
                "BTTS": cbt,
                "Córners": cco,
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     use_container_width=True, height=480)
        st.caption(
            f"{len(view)} partidos · lado más probable del modelo con su "
            "probabilidad y la **cuota sellada** (@) · **Córners** = lado que llama "
            "el modelo (Over si la proyección supera la línea) con su cuota sellada, "
            "y de fondo *proyección vs línea de la casa*. Se calificarán al finalizar."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Asistente de Análisis Pre-Partido",
        page_icon="⚽",
        layout="wide",
    )
    st.title("⚽ Asistente de Análisis Pre-Partido")

    tab_analysis, tab_results, tab_pred, tab_live, tab_perf = st.tabs(
        ["🔍 Análisis", "✅ Resultados", "📋 Predicciones",
         "🎯 Picks en vivo", "📈 Rendimiento"]
    )
    with tab_analysis:
        _render_analysis()
    with tab_results:
        _render_results()
    with tab_pred:
        _render_predictions()
    with tab_live:
        _render_live_picks()
    with tab_perf:
        _render_performance()

    st.sidebar.divider()
    st.sidebar.caption(
        f"Actualizado: {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}"
    )


if __name__ == "__main__":
    main()
