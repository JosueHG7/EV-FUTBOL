import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
import config
from models.poisson_model import PoissonModel, load_matches
from models.ensemble_model import EnsembleModel
from models.feature_engineering import build_features
from collectors.understat_collector import load_all_xg
from models.ev_calculator import find_value_bets
from models.kelly import recommended_stake

_LEAGUE_LABELS = {
    "soccer_spain_la_liga":         "La Liga",
    "soccer_epl":                   "Premier League",
    "soccer_germany_bundesliga":    "Bundesliga",
    "soccer_italy_serie_a":         "Serie A",
    "soccer_france_ligue_one":      "Ligue 1",
    "soccer_uefa_champs_league":    "Champions League",
}

_BET_LABELS = {
    "home_win":  "Local",
    "draw":      "Empate",
    "away_win":  "Visitante",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bookmaker_h2h_odds(
    bookmakers: list, home_team: str, away_team: str, bookmaker_key: str
) -> dict | None:
    """Cuotas h2h del bookmaker especificado. Devuelve None si no está disponible."""
    bk = next((b for b in bookmakers if b["title"] == bookmaker_key), None)
    if bk is None:
        return None

    h2h = next((m for m in bk["markets"] if m["key"] == "h2h"), None)
    if h2h is None:
        return None

    prices: dict[str, float] = {o["name"]: float(o["price"]) for o in h2h["outcomes"]}

    home_price = prices.get(home_team)
    draw_price = prices.get("Draw")
    away_price = prices.get(away_team)

    if None in (home_price, draw_price, away_price):
        return None

    return {
        "home_win": home_price,
        "draw":     draw_price,
        "away_win": away_price,
    }


def _fmt_date(iso: str) -> str:
    return iso[:10] if iso else "?"


# ---------------------------------------------------------------------------
# Reusable pipeline steps (shared by the CLI and the Streamlit dashboard)
# ---------------------------------------------------------------------------

def build_live_model() -> tuple[object, set, str, "pd.DataFrame"]:
    """
    Carga la BD, construye features (con xG si está disponible) y entrena el
    EnsembleModel para predicción en vivo.

    Devuelve (model, teams, model_label, matches_df). Lanza RuntimeError si la
    BD está vacía. Cae a PoissonModel si el Ensemble falla.
    """
    df = load_matches()
    if df.empty:
        raise RuntimeError(
            "No hay partidos en la BD. Ejecuta database/ingest.py primero."
        )

    try:
        xg_df = load_all_xg()
    except Exception:
        xg_df = None

    feat_df = build_features(df, xg_df=xg_df)

    model_label = "Ensemble (Poisson 0.451 / XGBoost 0.549)"
    try:
        model = EnsembleModel()
        model.fit(df, feat_df, weights={"poisson": 0.451, "xgboost": 0.549})
        teams = model.teams
    except Exception as exc:
        model = PoissonModel()
        model.fit(df)
        teams = model.teams
        model_label = f"PoissonModel (fallback: {exc})"

    return model, teams, model_label, df


def load_raw_odds() -> dict:
    """Carga data/odds_raw.json. Lanza FileNotFoundError si no existe."""
    odds_path = config.DATA_DIR / "odds_raw.json"
    if not odds_path.exists():
        raise FileNotFoundError(
            f"{odds_path} no encontrado. Ejecuta collectors/odds_collector.py primero."
        )
    with open(odds_path, encoding="utf-8") as f:
        return json.load(f)


def scan_picks(
    model: object,
    teams: set,
    raw_odds: dict,
    min_ev: float = config.MIN_EV_THRESHOLD,
    bankroll: float = config.DEFAULT_BANKROLL,
) -> tuple[list[dict], dict]:
    """
    Escanea todos los partidos con odds y devuelve los value bets del modelo.

    Devuelve (picks_ordenados_por_ev, stats), donde stats contiene los
    contadores de diagnóstico: skipped_model, skipped_odds, filtered_odds,
    total_matches.
    """
    all_picks: list[dict] = []
    skipped_model = 0
    skipped_odds  = 0
    filtered_odds = 0

    for league_key, matches in raw_odds["data"].items():
        league = _LEAGUE_LABELS.get(league_key, league_key)

        for match in matches:
            home = match["home_team"]
            away = match["away_team"]

            if home not in teams or away not in teams:
                skipped_model += 1
                continue

            market_odds = _bookmaker_h2h_odds(
                match["bookmakers"], home, away, config.TARGET_BOOKMAKER
            )
            if market_odds is None:
                skipped_odds += 1
                continue

            out_of_range = [
                f"{k}={v}" for k, v in market_odds.items()
                if not (config.MIN_ODDS <= v <= config.MAX_ODDS)
            ]
            if out_of_range:
                filtered_odds += len(out_of_range)
                continue

            model_probs = model.predict_1x2(home, away)
            value_bets  = find_value_bets(model_probs, market_odds, min_ev=min_ev)

            for vb in value_bets:
                stake_info = recommended_stake(vb["model_prob"], vb["odds"], bankroll)
                all_picks.append({
                    "league": league,
                    "home":   home,
                    "away":   away,
                    "date":   match.get("commence_time", ""),
                    **vb,
                    **stake_info,
                })

    all_picks.sort(key=lambda x: x["ev"], reverse=True)
    stats = {
        "skipped_model": skipped_model,
        "skipped_odds":  skipped_odds,
        "filtered_odds": filtered_odds,
        "total_matches": sum(len(v) for v in raw_odds["data"].values()),
    }
    return all_picks, stats


# ---------------------------------------------------------------------------
# Pipeline (CLI)
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    # 1. Cargar datos históricos + entrenar
    print("Cargando datos y entrenando modelo Ensemble...")
    print("  (BD + xG Understat caché 48h + features)")
    try:
        model, teams, model_label, df = build_live_model()
    except RuntimeError as exc:
        print(f"  [ERROR] {exc}")
        return

    print(f"  {len(df)} partidos | {len(teams)} equipos | {model_label}\n")

    # 2. Cargar odds
    try:
        raw_odds = load_raw_odds()
    except FileNotFoundError as exc:
        print(f"  [ERROR] {exc}")
        return

    collected_at = raw_odds.get("collected_at", "desconocido")
    total_matches = sum(len(v) for v in raw_odds["data"].values())
    print(f"Odds cargadas — {total_matches} partidos | recopiladas: {collected_at}\n")

    # 3. Escanear partidos
    all_picks, stats = scan_picks(model, teams, raw_odds)
    skipped_model = stats["skipped_model"]
    skipped_odds  = stats["skipped_odds"]
    filtered_odds = stats["filtered_odds"]

    # 5. Reporte
    SEP = "=" * 68
    print(SEP)
    print(f"  PICKS DEL DIA  |  Bankroll: {config.DEFAULT_BANKROLL:.0f}  |  Min EV: {config.MIN_EV_THRESHOLD:.0%}  |  Book: {config.TARGET_BOOKMAKER}")
    print(f"  Modelo: {model_label}")
    print(SEP)

    if not all_picks:
        print("\n  Sin value bets hoy.\n")
    else:
        for i, p in enumerate(all_picks, 1):
            bet   = _BET_LABELS.get(p["bet_type"], p["bet_type"])
            stake = f"{p['stake']:.2f} ({p['pct_bankroll']:.1%})" if p["stake"] > 0 else "No apostar (Kelly insuficiente)"
            print(
                f"\n  #{i}  {p['home']} vs {p['away']}\n"
                f"       Liga  : {p['league']}  |  Fecha: {_fmt_date(p['date'])}\n"
                f"       Apuesta: {bet} @ {p['odds']}\n"
                f"       Modelo : {p['model_prob']:.1%}  |  Implícita: {p['implied_prob']:.1%}  |  Edge: {p['edge']:+.1%}\n"
                f"       EV     : {p['ev']:+.2%}  |  Overround: {p['overround']:.4f}\n"
                f"       Stake  : {stake}"
            )

    print(f"\n{SEP}")
    print(
        f"  Picks encontrados : {len(all_picks)}\n"
        f"  Sin modelo        : {skipped_model} partidos (nombres no coinciden)\n"
        f"  Sin {config.TARGET_BOOKMAKER:<10}   : {skipped_odds} partidos\n"
        f"  Cuotas filtradas  : {filtered_odds} outcomes fuera de [{config.MIN_ODDS}, {config.MAX_ODDS}]"
    )
    print(SEP)


if __name__ == "__main__":
    run_pipeline()
