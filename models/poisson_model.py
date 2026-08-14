import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sqlalchemy import select

sys.path.append(str(Path(__file__).parent.parent))
from database.db import get_session
from database.models import Match

_MAX_GOALS = 10   # dimensión de la matriz de marcadores


# ---------------------------------------------------------------------------
# Dixon-Coles correction
# ---------------------------------------------------------------------------

def _tau(x: int, y: int, lambda_h: float, lambda_a: float, rho: float) -> float:
    """
    Correction factor for low-scoring cells (Dixon & Coles, 1997).

    Adjusts the joint probability of results where x+y <= 2 to account for
    the observed over-representation of draws and narrow wins in real data.
    rho < 0 lowers 0-0 / 1-1 and raises 1-0 / 0-1.
    """
    if x == 0 and y == 0:
        return 1.0 - lambda_h * lambda_a * rho
    if x == 1 and y == 0:
        return 1.0 + lambda_a * rho
    if x == 0 and y == 1:
        return 1.0 + lambda_h * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_matches(league_id: int | None = None) -> pd.DataFrame:
    """
    Lee partidos finalizados de la BD y devuelve un DataFrame.
    Filtra por liga si se especifica league_id.
    """
    with get_session() as session:
        stmt = select(Match).where(
            Match.status == "finished",
            Match.home_goals.is_not(None),
            Match.away_goals.is_not(None),
        )
        if league_id is not None:
            stmt = stmt.where(Match.league_id == league_id)

        rows = session.scalars(stmt).all()
        records = [
            {
                "home_team":  m.home_team_name,
                "away_team":  m.away_team_name,
                "home_goals": m.home_goals,
                "away_goals": m.away_goals,
                "league_id":  m.league_id,
                "season":     m.season,
                "match_date": m.match_date,
            }
            for m in rows
        ]
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PoissonModel:

    def __init__(self) -> None:
        self.teams: dict[str, dict] = {}
        self.avg_goals: float = 0.0
        self.home_advantage: float = 1.0
        self.rho: float = 0.0          # Dixon-Coles correlation parameter
        self.decay_factor: float = 0.98
        self._fitted = False

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        league_id: int | None = None,
        decay_factor: float = 0.98,
    ) -> None:
        """
        Entrena el modelo con los partidos del DataFrame.
        Si league_id se especifica, filtra solo esa liga.
        decay_factor controla el peso temporal: weight = decay_factor ^ días_desde_partido.
        Un valor de 0.98 significa que un partido de hace 35 días vale ~50% de uno reciente.
        """
        if league_id is not None:
            df = df[df["league_id"] == league_id].copy()

        if df.empty:
            raise ValueError("No hay partidos para entrenar el modelo.")

        self.decay_factor = decay_factor
        self._fit_core(df, decay_factor)
        self.rho = self.fit_rho(df)

    # ------------------------------------------------------------------
    # _fit_core  (weighted parameter estimation — no recursion)
    # ------------------------------------------------------------------

    def _fit_core(self, df: pd.DataFrame, decay_factor: float) -> None:
        """
        Fits team attack/defense parameters using exponentially weighted
        match history.  Called by fit() and fit_decay().
        """
        max_date = df["match_date"].max()
        days_since = (max_date - df["match_date"]).dt.days.clip(lower=0)
        weights = decay_factor ** days_since

        total_w = float(weights.sum())
        avg_home = float((df["home_goals"] * weights).sum()) / total_w
        avg_away = float((df["away_goals"] * weights).sum()) / total_w
        self.avg_goals      = (avg_home + avg_away) / 2
        self.home_advantage = avg_home / self.avg_goals

        self.teams = {}
        for team in set(df["home_team"]) | set(df["away_team"]):
            hm = df["home_team"] == team
            am = df["away_team"] == team
            hw = weights[hm]
            aw = weights[am]
            tw = float(hw.sum() + aw.sum())
            if tw == 0:
                continue
            scored = (
                float((df.loc[hm, "home_goals"] * hw).sum()) +
                float((df.loc[am, "away_goals"] * aw).sum())
            )
            conceded = (
                float((df.loc[hm, "away_goals"] * hw).sum()) +
                float((df.loc[am, "home_goals"] * aw).sum())
            )
            self.teams[team] = {
                "attack":  (scored   / tw) / self.avg_goals,
                "defense": (conceded / tw) / self.avg_goals,
                "games":   int(hm.sum() + am.sum()),
            }

        self._fitted = True

    # ------------------------------------------------------------------
    # fit_decay
    # ------------------------------------------------------------------

    def fit_decay(self, df: pd.DataFrame) -> float:
        """
        Finds the optimal temporal decay factor by maximising log-likelihood
        on a held-out test set (last 20% of matches by date).

        Trains on the first 80%, evaluates out-of-sample Poisson log-likelihood
        on the remaining 20%.  Search range: [0.90, 0.999].
        """
        df_sorted = df.sort_values("match_date").reset_index(drop=True)
        split     = int(len(df_sorted) * 0.8)
        train_df  = df_sorted.iloc[:split].copy()
        eval_df   = df_sorted.iloc[split:].copy()

        if train_df.empty or eval_df.empty:
            return 0.98

        def neg_log_lik(decay: float) -> float:
            m = PoissonModel()
            m._fit_core(train_df, decay)

            total = 0.0
            for _, row in eval_df.iterrows():
                home, away = row["home_team"], row["away_team"]
                if home not in m.teams or away not in m.teams:
                    continue
                hp = m._team_params(home)
                ap = m._team_params(away)
                lh = hp["attack"] * ap["defense"] * m.home_advantage * m.avg_goals
                la = ap["attack"] * hp["defense"] * m.avg_goals
                x, y = int(row["home_goals"]), int(row["away_goals"])
                total += math.log(max(float(poisson.pmf(x, lh)), 1e-10))
                total += math.log(max(float(poisson.pmf(y, la)), 1e-10))
            return -total

        result = minimize_scalar(neg_log_lik, bounds=(0.90, 0.999), method="bounded")
        return float(result.x)

    # ------------------------------------------------------------------
    # fit_rho
    # ------------------------------------------------------------------

    def fit_rho(self, df: pd.DataFrame) -> float:
        """
        Finds the optimal Dixon-Coles rho by maximising the log-likelihood
        of the tau correction over historical low-scoring results.

        Only results where x+y <= 2 contribute to the likelihood (tau = 1
        for all other scores, so their log-contribution is zero).
        Search range: rho ∈ [-0.2, 0.0]  (paper found ≈ -0.13).
        """
        # Pre-compute (x, y, λ_h, λ_a) for low-score matches only
        low_score_data: list[tuple[int, int, float, float]] = []
        for _, row in df.iterrows():
            home, away = row["home_team"], row["away_team"]
            if home not in self.teams or away not in self.teams:
                continue
            x, y = int(row["home_goals"]), int(row["away_goals"])
            if x > 1 or y > 1:
                continue
            hp = self._team_params(home)
            ap = self._team_params(away)
            lh = hp["attack"] * ap["defense"] * self.home_advantage * self.avg_goals
            la = ap["attack"] * hp["defense"] * self.avg_goals
            low_score_data.append((x, y, lh, la))

        if not low_score_data:
            return 0.0

        def neg_log_lik(rho: float) -> float:
            total = 0.0
            for x, y, lh, la in low_score_data:
                t = _tau(x, y, lh, la, rho)
                if t <= 0.0:
                    return 1e10
                total += math.log(t)
            return -total

        result = minimize_scalar(neg_log_lik, bounds=(-0.2, 0.0), method="bounded")
        return float(result.x)

    # ------------------------------------------------------------------
    # predict_score_matrix
    # ------------------------------------------------------------------

    def predict_score_matrix(
        self, home_team: str, away_team: str
    ) -> np.ndarray:
        """Devuelve matriz _MAX_GOALS × _MAX_GOALS de probabilidades de marcadores."""
        self._check_fitted()
        hp = self._team_params(home_team)
        ap = self._team_params(away_team)

        lambda_home = hp["attack"] * ap["defense"] * self.home_advantage * self.avg_goals
        lambda_away = ap["attack"] * hp["defense"] * self.avg_goals

        home_pmf = poisson.pmf(range(_MAX_GOALS), lambda_home)
        away_pmf = poisson.pmf(range(_MAX_GOALS), lambda_away)

        matrix = np.outer(home_pmf, away_pmf)

        # Apply Dixon-Coles correction to low-score cells
        if self.rho != 0.0:
            for x in range(2):
                for y in range(2):
                    matrix[x, y] = max(
                        0.0,
                        matrix[x, y] * _tau(x, y, lambda_home, lambda_away, self.rho),
                    )
            matrix /= matrix.sum()   # renormalise to sum to 1

        return matrix

    # ------------------------------------------------------------------
    # predict_1x2
    # ------------------------------------------------------------------

    def predict_1x2(
        self, home_team: str, away_team: str
    ) -> dict[str, float]:
        """Devuelve probabilidades de victoria local, empate y victoria visitante."""
        matrix = self.predict_score_matrix(home_team, away_team)

        p_home = float(np.sum(np.tril(matrix, -1)))   # home_goals > away_goals
        p_draw = float(np.sum(np.diag(matrix)))        # home_goals == away_goals
        p_away = float(np.sum(np.triu(matrix,  1)))   # away_goals > home_goals

        return {"home_win": p_home, "draw": p_draw, "away_win": p_away}

    # ------------------------------------------------------------------
    # predict_over_under
    # ------------------------------------------------------------------

    def predict_over_under(
        self, home_team: str, away_team: str, line: float = 2.5
    ) -> dict[str, float]:
        """P(over/under X.5 goles totales)."""
        matrix = self.predict_score_matrix(home_team, away_team)

        total = np.array(
            [[i + j for j in range(_MAX_GOALS)] for i in range(_MAX_GOALS)]
        )
        p_over = float(np.sum(matrix[total > line]))
        return {"over": p_over, "under": 1.0 - p_over}

    # ------------------------------------------------------------------
    # predict_btts
    # ------------------------------------------------------------------

    def predict_btts(self, home_team: str, away_team: str) -> dict[str, float]:
        """P(ambos equipos anotan) y P(no)."""
        matrix = self.predict_score_matrix(home_team, away_team)
        p_btts = float(np.sum(matrix[1:, 1:]))
        return {"yes": p_btts, "no": 1.0 - p_btts}

    # ------------------------------------------------------------------
    # get_team_params
    # ------------------------------------------------------------------

    def get_team_params(self, team_name: str) -> dict:
        """Devuelve attack, defense y partidos jugados de un equipo."""
        self._check_fitted()
        if team_name not in self.teams:
            raise KeyError(f"Equipo no encontrado en el modelo: '{team_name}'")
        p = self.teams[team_name]
        return {"attack": p["attack"], "defense": p["defense"], "games": p["games"]}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("El modelo no ha sido entrenado. Llama a fit() primero.")

    def _team_params(self, team: str) -> dict:
        if team not in self.teams:
            raise KeyError(f"Equipo no encontrado en el modelo: '{team}'")
        return self.teams[team]


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Cargando partidos de la BD...")
    df = load_matches()
    print(f"  {len(df)} partidos finalizados en {df['league_id'].nunique()} ligas\n")

    # --- Buscar decay óptimo ---
    print("Buscando decay_factor optimo (train 80% / eval 20%)...")
    _tmp = PoissonModel()
    _tmp._fit_core(df, 0.98)          # bootstrap needed by fit_decay internals
    optimal_decay = _tmp.fit_decay(df)
    print(f"  decay_factor optimo = {optimal_decay:.4f}\n")

    # --- Entrenar modelo sin decay (decay=1.0 → pesos iguales) ---
    model_flat = PoissonModel()
    model_flat.fit(df, decay_factor=1.0)

    # --- Entrenar modelo con decay óptimo ---
    model_decay = PoissonModel()
    model_decay.fit(df, decay_factor=optimal_decay)

    print(f"  Modelo plano    : avg_goals={model_flat.avg_goals:.3f}  "
          f"home_adv={model_flat.home_advantage:.3f}  rho={model_flat.rho:.4f}")
    print(f"  Modelo con decay: avg_goals={model_decay.avg_goals:.3f}  "
          f"home_adv={model_decay.home_advantage:.3f}  rho={model_decay.rho:.4f}\n")

    # --- Partido de prueba con modelo decay ---
    HOME, AWAY = "Real Madrid", "Barcelona"
    print(f"--- {HOME} vs {AWAY} (modelo con decay) ---")

    p1x2 = model_decay.predict_1x2(HOME, AWAY)
    print(f"  1X2          : Local {p1x2['home_win']:.1%} | Empate {p1x2['draw']:.1%} | Visitante {p1x2['away_win']:.1%}")

    ou = model_decay.predict_over_under(HOME, AWAY, 2.5)
    print(f"  Over/Under 2.5: Over {ou['over']:.1%} | Under {ou['under']:.1%}")

    matrix = model_decay.predict_score_matrix(HOME, AWAY)
    peak = np.unravel_index(matrix.argmax(), matrix.shape)
    print(f"  Marcador mas probable: {peak[0]}-{peak[1]} ({matrix[peak]:.1%})\n")

    # --- Comparación Dixon-Coles con modelo decay ---
    print("Comparacion marcadores bajos CON vs SIN Dixon-Coles (modelo decay):")
    print(f"  {'Marcador':<10} {'Sin DC':>8} {'Con DC':>8} {'Dif':>8}")
    print("  " + "-" * 36)

    rho_saved = model_decay.rho
    model_decay.rho = 0.0
    matrix_no_dc = model_decay.predict_score_matrix(HOME, AWAY)
    model_decay.rho = rho_saved
    matrix_dc = model_decay.predict_score_matrix(HOME, AWAY)

    for x, y in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        p_no = matrix_no_dc[x, y]
        p_dc = matrix_dc[x, y]
        print(f"  {x}-{y:<8}  {p_no:>7.2%}  {p_dc:>7.2%}  {p_dc - p_no:>+7.3%}")

    print()

    # --- Comparación top 5 ataque: sin decay vs con decay ---
    print("Top 5 ATAQUE — sin decay (igual peso todos los partidos):")
    flat_df = pd.DataFrame(
        [{"team": t, **v} for t, v in model_flat.teams.items()]
    ).query("games >= 10")
    for _, r in flat_df.nlargest(5, "attack").iterrows():
        print(f"  {r['team']:<28} attack={r['attack']:.3f}  games={r['games']}")

    print(f"\nTop 5 ATAQUE — con decay={optimal_decay:.4f} (equipos en forma reciente suben):")
    decay_df = pd.DataFrame(
        [{"team": t, **v} for t, v in model_decay.teams.items()]
    ).query("games >= 10")
    for _, r in decay_df.nlargest(5, "attack").iterrows():
        print(f"  {r['team']:<28} attack={r['attack']:.3f}  games={r['games']}")

    print("\nTop 5 DEFENSA — con decay (menos goles recibidos ponderados):")
    for _, r in decay_df.nsmallest(5, "defense").iterrows():
        print(f"  {r['team']:<28} defense={r['defense']:.3f}  games={r['games']}")
