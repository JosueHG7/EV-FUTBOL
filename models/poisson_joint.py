"""
models/poisson_joint.py — PROTOTYPE, not wired into refresh.py or the dashboard.

The production model (`models/poisson_model.py::PoissonModel._fit_core`) sets
each team's attack/defense from that team's OWN scoring rate vs the global
average — it never looks at who they played. That's fine within one league,
but it means the model has no basis to compare a Bundesliga team's strength
to an Úrvalsdeild team's: nothing connects the two ratings.

`JointPoissonModel` fits attack/defense the way Dixon & Coles (1997) actually
described it: maximize one joint (weighted) Poisson log-likelihood over ALL
teams and ALL matches at once. A team's rating is then pulled by every match
it (or its opponents, transitively) played — including cross-competition
matches (Champions/Europa/Conference League, Libertadores, Sudamericana),
which is exactly the data added on 2026-08-20 to make this possible. Ridge
regularization toward "average team" (attack=defense=1) both resolves the
model's scale ambiguity and shrinks small samples (a team with 1-2 matches
this season, e.g. an early Champions League qualifier) instead of letting
them take on a wild estimate.

Only promote this to production if it measurably improves out-of-sample
accuracy on cross-league matches vs the baseline — see PROJECT_STATUS.md
pendiente #2. Do not import this from refresh.py / dashboard / main.py.
"""

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from models.poisson_model import PoissonModel

_LOG_LAMBDA_CLIP = 20.0   # numerical safety during line search (exp(20) ~ 5e8)


class JointPoissonModel(PoissonModel):

    def __init__(self, reg: float = 0.01) -> None:
        super().__init__()
        self.reg = reg              # ridge strength on log(attack)/log(defense)
        self.n_teams_: int | None = None
        self.n_iter_: int | None = None
        self.converged_: bool | None = None

    # ------------------------------------------------------------------
    # _fit_core — same signature/outputs as PoissonModel, different estimator
    # ------------------------------------------------------------------

    def _fit_core(self, df: pd.DataFrame, decay_factor: float) -> None:
        max_date = df["match_date"].max()
        days_since = (max_date - df["match_date"]).dt.days.clip(lower=0)
        time_w = (decay_factor ** days_since).to_numpy()

        total_w = float(time_w.sum())
        avg_home = float((df["home_goals"].to_numpy() * time_w).sum()) / total_w
        avg_away = float((df["away_goals"].to_numpy() * time_w).sum()) / total_w
        self.avg_goals = (avg_home + avg_away) / 2
        self.home_advantage = avg_home / self.avg_goals

        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        self.n_teams_ = n

        home_idx = df["home_team"].map(idx).to_numpy()
        away_idx = df["away_team"].map(idx).to_numpy()
        y_h = df["home_goals"].to_numpy(dtype=float)
        y_a = df["away_goals"].to_numpy(dtype=float)

        log_c1 = math.log(self.home_advantage * self.avg_goals)   # lambda_home constant term
        log_c2 = math.log(self.avg_goals)                          # lambda_away constant term

        def nll_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
            alpha, delta = x[:n], x[n:]   # log(attack), log(defense)

            log_lh = np.clip(alpha[home_idx] + delta[away_idx] + log_c1,
                             -_LOG_LAMBDA_CLIP, _LOG_LAMBDA_CLIP)
            log_la = np.clip(alpha[away_idx] + delta[home_idx] + log_c2,
                             -_LOG_LAMBDA_CLIP, _LOG_LAMBDA_CLIP)
            lh, la = np.exp(log_lh), np.exp(log_la)

            loss = float(np.sum(time_w * (lh - y_h * log_lh + la - y_a * log_la)))
            loss += self.reg * float(np.sum(alpha ** 2) + np.sum(delta ** 2))

            g_alpha = np.zeros(n)
            g_delta = np.zeros(n)
            wh = time_w * (lh - y_h)
            wa = time_w * (la - y_a)
            np.add.at(g_alpha, home_idx, wh)
            np.add.at(g_delta, away_idx, wh)
            np.add.at(g_alpha, away_idx, wa)
            np.add.at(g_delta, home_idx, wa)
            g_alpha += 2 * self.reg * alpha
            g_delta += 2 * self.reg * delta
            return loss, np.concatenate([g_alpha, g_delta])

        result = minimize(nll_grad, np.zeros(2 * n), jac=True, method="L-BFGS-B",
                          options={"maxiter": 500})
        self.n_iter_ = int(result.nit)
        self.converged_ = bool(result.success)

        alpha, delta = result.x[:n], result.x[n:]
        attack = np.exp(alpha)
        defense = np.exp(delta)

        games = (pd.concat([df["home_team"], df["away_team"]])
                 .value_counts())

        self.teams = {
            t: {"attack": float(attack[i]), "defense": float(defense[i]),
                "games": int(games.get(t, 0))}
            for t, i in idx.items()
        }
        self._fitted = True


if __name__ == "__main__":
    import time
    from models.poisson_model import PoissonModel, load_matches

    print("Cargando partidos de la BD...")
    df = load_matches()
    print(f"  {len(df)} partidos · {len(set(df['home_team']) | set(df['away_team']))} equipos\n")

    print("Ajustando modelo base (independiente)...")
    t0 = time.time()
    base = PoissonModel()
    base.fit(df, decay_factor=0.98)
    print(f"  listo en {time.time()-t0:.1f}s\n")

    print("Ajustando modelo joint MLE (prototipo)...")
    t0 = time.time()
    joint = JointPoissonModel(reg=0.01)
    joint.fit(df, decay_factor=0.98)
    print(f"  listo en {time.time()-t0:.1f}s · convergió={joint.converged_} · iter={joint.n_iter_}\n")

    for home, away in [("Real Madrid", "Barcelona"),
                       ("Motherwell", "Freiburg"),
                       ("Bayern Munich", "Paris Saint Germain")]:
        try:
            pb = base.predict_1x2(home, away)
            pj = joint.predict_1x2(home, away)
        except KeyError as exc:
            print(f"{home} vs {away}: {exc}")
            continue
        print(f"{home} vs {away}")
        print(f"  base : L{pb['home_win']:.1%} E{pb['draw']:.1%} V{pb['away_win']:.1%}")
        print(f"  joint: L{pj['home_win']:.1%} E{pj['draw']:.1%} V{pj['away_win']:.1%}")
