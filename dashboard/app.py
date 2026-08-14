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

@st.cache_resource(show_spinner="Entrenando EnsembleModel (BD + xG + features)…")
def _get_live_model():
    """Entrena una vez el EnsembleModel por sesión (cacheado como recurso)."""
    model, teams, label, df = pipeline.build_live_model()
    return model, teams, label, len(df)


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
        st.info("Ejecuta `python collectors/odds_collector.py` para obtener odds actuales.")
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Football EV Dashboard",
        page_icon="⚽",
        layout="wide",
    )
    st.title("⚽ Football EV Betting System")

    tab_live, tab_perf = st.tabs(["🎯 Picks en vivo", "📈 Rendimiento"])
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
