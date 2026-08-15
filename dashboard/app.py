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


@st.cache_data(show_spinner=False)
def _get_matches():
    """Partidos finalizados de la BD (para forma/H2H/descanso)."""
    from models.poisson_model import load_matches
    return load_matches()


@st.cache_data(ttl=3600, show_spinner=False)
def _get_xg():
    """xG de Understat (caché 1h en la app; el colector cachea 48h)."""
    from collectors.understat_collector import load_all_xg
    try:
        return load_all_xg()
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def _get_stats_df():
    """Estadísticas por-partido (team_statistics) para promedios rodantes."""
    from sqlalchemy import select
    from database.db import get_session
    from database.models import TeamStatistic
    with get_session() as s:
        rows = s.execute(
            select(TeamStatistic.match_id, TeamStatistic.team_id,
                   TeamStatistic.type, TeamStatistic.value_num)
        ).all()
    return pd.DataFrame(rows, columns=["match_id", "team_id", "type", "value_num"])


@st.cache_data(ttl=600, show_spinner=False)
def _get_recent_finished(days_back: int = 3):
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
        for m in ms:
            md = pd.Timestamp(m.match_date)
            if md.tzinfo is not None:
                md = md.tz_localize(None)
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


@st.cache_data(ttl=600, show_spinner=False)
def _get_upcoming() -> list[dict]:
    """Partidos próximos con odds (desde la BD API-Football): 1X2 + O/U 2.5 + BTTS."""
    from sqlalchemy import select
    from database.db import get_session
    from database.models import Match, Odds, MarketOdds

    out: list[dict] = []
    with get_session() as s:
        for o in s.execute(select(Odds).where(Odds.market == "1x2")).scalars().all():
            m = s.get(Match, o.match_id)
            if m is None:
                continue
            md = pd.Timestamp(m.match_date)
            if md.tzinfo is not None:
                md = md.tz_localize(None)
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


def _devig2(a: float, b: float) -> tuple[float, float]:
    ia, ib = 1.0 / a, 1.0 / b
    tot = ia + ib
    return ia / tot, ib / tot


def _signals(d: dict, ou25: dict | None, btts: dict | None) -> list[dict]:
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
        ov, un = _devig2(ou25["over"], ou25["under"])
        model = {"Over": gm["over25"], "Under": gm["under25"]}
        market = {"Over": ov, "Under": un}
        sig.append({"mkt": "Goles O/U 2.5", "model": model, "market": market,
                    "delta": abs(model["Over"] - market["Over"])})
    if gm and btts:
        y, n = _devig2(btts["yes"], btts["no"])
        model = {"Sí": gm["btts_yes"], "No": gm["btts_no"]}
        market = {"Sí": y, "No": n}
        sig.append({"mkt": "BTTS", "model": model, "market": market,
                    "delta": abs(model["Sí"] - market["Sí"])})
    return sig


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


def _render_detail(u: dict, matches_df, stats_df) -> None:
    from models.match_analysis import team_recent_detail, h2h_detail
    d = u["_d"]
    home, away = d["home"], d["away"]
    before = u["dt"]

    st.markdown(f"### {home}  vs  {away}")
    cty = f" ({u['country']})" if u.get("country") else ""
    st.caption(f"{u['league']}{cty} · {d['date']} · cuotas: {u['book']}")
    st.caption(
        "Diagnóstico de análisis, **no** sugerencias de apuesta. El modelo no está "
        "validado contra resultados (no le gana al mercado en backtest). El juicio es tuyo."
    )

    if d["known"] and u["_sig"]:
        st.markdown(
            "**📊 Dónde el modelo (sin calibrar) más se aleja del mercado** "
            "— diagnóstico, no apuesta sugerida"
        )
        rows = [{"Mercado": s["mkt"], "Modelo": _dist_str(s["model"]),
                 "Mercado ": _dist_str(s["market"]), "Δ máx": s["delta"]}
                for s in sorted(u["_sig"], key=lambda x: x["delta"], reverse=True)]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
            column_config={"Δ máx": st.column_config.NumberColumn(format="percent")})
        cs = " · ".join(f"{c['score']} ({c['prob']:.0%})" for c in d["correct_scores"])
        st.caption(f"Marcadores más probables del modelo: {cs}")

    # Resumen de promedios (arriba) + detalle partido-a-partido (abajo)
    _render_stats_table(d)
    _render_corner_signal(u)

    st.markdown("#### 📋 Últimos partidos (partido a partido)")
    dh = team_recent_detail(stats_df, matches_df, home, before, 6)
    da = team_recent_detail(stats_df, matches_df, away, before, 6)
    if dh:
        _render_team_detail(dh, f"🏠 {home}")
    if da:
        _render_team_detail(da, f"✈️ {away}")

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


@st.cache_resource(show_spinner="Analizando partidos (una sola vez)…")
def _get_analyzed():
    """Construye TODOS los dossiers + señales una sola vez (cacheado).

    Así filtrar/seleccionar en la UI no recalcula nada → respuesta instantánea.
    """
    from models.match_analysis import build_dossier
    model, teams, label, n_matches = _get_live_model()
    matches_df = _get_matches()
    xg_df = _get_xg()
    stats_df = _get_stats_df()
    upcoming = _get_upcoming()
    for u in upcoming:
        d = build_dossier(model, matches_df, xg_df, u["home"], u["away"], u["dt"],
                          u["odds_1x2"], stats_df=stats_df)
        u["_d"] = d
        u["_sig"] = _signals(d, u["ou25"], u["btts"]) if d["known"] else []
        u["_top"] = max((s["delta"] for s in u["_sig"]), default=None)
    return upcoming, label, n_matches


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

    c1, c2 = st.columns([2, 1])
    leagues = ["Todas"] + sorted({u["league"] for u in upcoming})
    sel = c1.selectbox("Liga", leagues)
    only_signals = c2.checkbox("Solo con discrepancia ≥ 5%", value=False)

    view = [u for u in upcoming if sel == "Todas" or u["league"] == sel]
    known = [u for u in view if u["_d"]["known"] and u["_top"] is not None]
    known.sort(key=lambda u: u["_top"], reverse=True)
    if only_signals:
        known = [u for u in known if u["_top"] >= 0.05]
    unknown = [u for u in view if not (u["_d"]["known"] and u["_top"] is not None)]

    if known:
        rows = []
        for u in known:
            top_mkt = max(u["_sig"], key=lambda s: s["delta"])["mkt"]
            rows.append({
                "Partido": f"{u['home']} vs {u['away']}",
                "Liga": u["league"],
                "Fecha": u["date"],
                "Mayor discrepancia": top_mkt,
                "Δ": u["_top"],
            })
        st.dataframe(
            pd.DataFrame(rows), hide_index=True, use_container_width=True,
            column_config={
                "Δ": st.column_config.NumberColumn(
                    format="percent",
                    help="Máxima divergencia |modelo − mercado| del partido"),
            },
        )
        st.caption(
            f"{len(known)} partidos · modelo entrenado con {n_matches:,} partidos "
            "(sin calibrar — la discrepancia es un diagnóstico, no un edge)"
        )

        st.divider()
        st.subheader("🔎 Detalle del partido")
        labels = [f"{u['home']} vs {u['away']} · {u['league']} · Δ{u['_top']:.0%}"
                  for u in known]
        idx = st.selectbox("Elige un partido para ver el detalle completo",
                           range(len(known)), format_func=lambda i: labels[i])
        _render_detail(known[idx], matches_df, stats_df)
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

    leagues = ["Todas"] + sorted({f["league"] for f in fin})
    sel = st.selectbox("Liga", leagues, key="res_liga")
    view = [f for f in fin if sel == "Todas" or f["league"] == sel]

    rows = []
    for f in view:
        rows.append({
            "Fecha": f["date"],
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Asistente de Análisis Pre-Partido",
        page_icon="⚽",
        layout="wide",
    )
    st.title("⚽ Asistente de Análisis Pre-Partido")

    tab_analysis, tab_results, tab_live, tab_perf = st.tabs(
        ["🔍 Análisis", "✅ Resultados", "🎯 Picks en vivo", "📈 Rendimiento"]
    )
    with tab_analysis:
        _render_analysis()
    with tab_results:
        _render_results()
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
