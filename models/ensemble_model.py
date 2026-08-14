"""
models/ensemble_model.py

Weighted ensemble that combines PoissonModel (Dixon-Coles + decay)
with XGBoostModel.  Optimal weights are found via a single walk-forward
pass — both models are trained per window, predictions are stored, then
Brent's method minimises the Brier Score over the Poisson weight w ∈ [0,1].
"""

import contextlib
import io
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.append(str(Path(__file__).parent.parent))
from models.feature_engineering import build_features
from models.poisson_model import PoissonModel, load_matches
from models.xgboost_model import (
    XGBoostModel,
    _TARGET_MAP,
    _TARGET_INV,
    _brier_score,
    prepare_dataset,
)
from collectors.understat_collector import load_all_xg


# ---------------------------------------------------------------------------
# Walk-forward prediction collector (trains both models per window)
# ---------------------------------------------------------------------------

def _walk_forward_predictions(
    matches_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single walk-forward pass.  For each window trains PoissonModel (with
    Dixon-Coles rho and decay_factor=0.98) and XGBoostModel, then collects
    out-of-sample predictions.

    Timeline: 12-month warmup, retrain every 4 weeks, eval next 4 weeks.

    Returns
    -------
    labels    : int ndarray (n,)   — 0=H, 1=D, 2=A
    poi_probs : float ndarray (n, 3)
    xgb_probs : float ndarray (n, 3)
    """
    X, y = prepare_dataset(matches_df, features_df)
    dates = pd.to_datetime(features_df["match_date"])

    min_date   = dates.min()
    max_date   = dates.max()
    warmup_end = min_date + timedelta(days=365)
    step       = timedelta(weeks=4)

    all_labels:    list[int]         = []
    all_poi_probs: list[list[float]] = []
    all_xgb_probs: list[list[float]] = []

    cursor  = warmup_end
    period  = 0
    n_total = int(((max_date - warmup_end) / step))

    while cursor + step <= max_date:
        train_mask = dates <= cursor
        eval_mask  = (dates > cursor) & (dates <= cursor + step)

        if train_mask.sum() < 100 or eval_mask.sum() < 5:
            cursor += step
            continue

        period += 1
        print(f"  Ventana {period:>2}/{n_total}  "
              f"train={train_mask.sum()}  eval={eval_mask.sum()}", end="\r")

        # --- Poisson ---
        tr_matches = matches_df[train_mask.values].reset_index(drop=True)
        pm = PoissonModel()
        try:
            pm.fit(tr_matches)
        except Exception:
            cursor += step
            continue

        # --- XGBoost ---
        xm = XGBoostModel()
        xm.fit(X[train_mask], y[train_mask])

        # --- Collect predictions ---
        ev_feat = features_df[eval_mask].reset_index(drop=True)
        ev_X    = X[eval_mask].reset_index(drop=True)
        ev_y    = y[eval_mask].reset_index(drop=True)
        xgb_all = xm.predict_proba(ev_X)     # (n_eval, 3) — batch

        for i in range(len(ev_feat)):
            r = ev_feat.iloc[i]
            try:
                p = pm.predict_1x2(r["home_team"], r["away_team"])
                all_labels.append(int(ev_y.iloc[i]))
                all_poi_probs.append([p["home_win"], p["draw"], p["away_win"]])
                all_xgb_probs.append(xgb_all[i].tolist())
            except (KeyError, RuntimeError):
                pass

        cursor += step

    print()   # newline after \r progress
    return (
        np.array(all_labels),
        np.array(all_poi_probs, dtype=float),
        np.array(all_xgb_probs, dtype=float),
    )


# ---------------------------------------------------------------------------
# EnsembleModel
# ---------------------------------------------------------------------------

class EnsembleModel:
    """
    Weighted ensemble: Poisson (Dixon-Coles + decay) + XGBoost.

    Default weights favour Poisson (0.6) — it shows slightly lower
    Brier Score on walk-forward validation.  Call optimize_weights()
    to find the data-driven optimum.
    """

    DEFAULT_WEIGHTS: dict[str, float] = {"poisson": 0.6, "xgboost": 0.4}

    def __init__(self) -> None:
        self.poisson = PoissonModel()
        self.xgboost = XGBoostModel()
        self.weights: dict[str, float] = dict(self.DEFAULT_WEIGHTS)
        self._fitted = False
        self._feat_df: pd.DataFrame | None = None
        self._X: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        matches_df: pd.DataFrame,
        features_df: pd.DataFrame,
        weights: dict[str, float] | None = None,
    ) -> None:
        """
        Train both sub-models on the full dataset.

        Parameters
        ----------
        matches_df  : raw match data from load_matches()
        features_df : output of build_features(matches_df)
        weights     : {'poisson': w, 'xgboost': 1-w}; None → default 0.6/0.4
        """
        if weights is not None:
            self.weights = weights

        self.poisson.fit(matches_df)

        X, y = prepare_dataset(matches_df, features_df)
        self.xgboost.fit(X, y)

        self._feat_df = features_df
        self._X = X
        self._fitted = True

    # ------------------------------------------------------------------
    # predict_proba
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        home_team: str,
        away_team: str,
        feature_row: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Weighted average of Poisson and XGBoost probabilities.

        Parameters
        ----------
        home_team   : home team name as stored in the DB
        away_team   : away team name
        feature_row : single-row DataFrame with the 33 feature columns

        Returns
        -------
        {'home_win': float, 'draw': float, 'away_win': float}
        """
        if not self._fitted:
            raise RuntimeError("EnsembleModel not fitted — call fit() first.")

        poi = self.poisson.predict_1x2(home_team, away_team)
        poi_arr = np.array([poi["home_win"], poi["draw"], poi["away_win"]])
        xgb_arr = self.xgboost.predict_proba(feature_row)[0]

        w_p = self.weights["poisson"]
        w_x = self.weights["xgboost"]
        combined = w_p * poi_arr + w_x * xgb_arr

        return {
            "home_win": round(float(combined[0]), 4),
            "draw":     round(float(combined[1]), 4),
            "away_win": round(float(combined[2]), 4),
        }

    # ------------------------------------------------------------------
    # teams property + predict_1x2 — PoissonModel-compatible interface
    # ------------------------------------------------------------------

    @property
    def teams(self) -> set:
        return self.poisson.teams

    def predict_1x2(self, home_team: str, away_team: str) -> dict[str, float]:
        """
        Drop-in replacement for PoissonModel.predict_1x2(home, away).

        Looks up the best available historical feature row (exact H2H first,
        then last home-team appearance) and delegates to predict_proba.
        Falls back to Poisson-only if no feature data is found.
        """
        if not self._fitted:
            raise RuntimeError("EnsembleModel not fitted — call fit() first.")

        if self._feat_df is not None and self._X is not None:
            fd = self._feat_df
            available_idx = set(self._X.index)
            for mask in [
                (fd["home_team"] == home_team) & (fd["away_team"] == away_team),
                fd["home_team"] == home_team,
            ]:
                candidates = fd[mask & fd.index.isin(available_idx)]
                if not candidates.empty:
                    last_idx = candidates.index[-1]
                    return self.predict_proba(
                        home_team, away_team, self._X.loc[[last_idx]]
                    )

        return self.poisson.predict_1x2(home_team, away_team)

    # ------------------------------------------------------------------
    # optimize_weights
    # ------------------------------------------------------------------

    def optimize_weights(
        self,
        matches_df: pd.DataFrame,
        features_df: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Find the Poisson weight w ∈ [0,1] that minimises walk-forward Brier.

        Runs a single walk-forward pass, then uses Brent's method (fast,
        O(1) per evaluation once predictions are cached).

        Returns
        -------
        {'poisson': w, 'xgboost': 1-w}
        """
        labels, poi_probs, xgb_probs = _walk_forward_predictions(
            matches_df, features_df
        )

        def brier_w(w: float) -> float:
            return _brier_score(w * poi_probs + (1 - w) * xgb_probs, labels)

        result = minimize_scalar(brier_w, bounds=(0.0, 1.0), method="bounded")
        w = float(result.x)
        self.weights = {"poisson": round(w, 4), "xgboost": round(1.0 - w, 4)}
        return self.weights


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Cargando partidos de la BD...")
    matches = load_matches()
    print(f"  {len(matches)} partidos | {matches['league_id'].nunique()} ligas\n")

    print("Cargando xG de Understat (con caché 48h)...")
    try:
        xg_df = load_all_xg()
        print(f"  {len(xg_df)} partidos con xG\n")
    except Exception as exc:
        print(f"  Fallo al cargar xG ({exc}) — sin features xG\n")
        xg_df = None

    print("Construyendo features...")
    feat_df = build_features(matches, xg_df=xg_df)

    # ---- Walk-forward pass (shared across all comparisons) ---------------
    print("Walk-forward: entrenando Poisson + XGBoost por ventana...")
    labels, poi_probs, xgb_probs = _walk_forward_predictions(matches, feat_df)
    print(f"  Partidos evaluados out-of-sample: {len(labels)}\n")

    # ---- Optimise weights ------------------------------------------------
    print("Optimizando peso del ensemble (Brent sobre Brier walk-forward)...")

    def _brier_w(w: float) -> float:
        return _brier_score(w * poi_probs + (1 - w) * xgb_probs, labels)

    opt_result = minimize_scalar(_brier_w, bounds=(0.0, 1.0), method="bounded")
    w_opt = float(opt_result.x)
    optimal_weights = {
        "poisson":  round(w_opt, 4),
        "xgboost":  round(1.0 - w_opt, 4),
    }
    print(f"  Pesos optimos: Poisson={optimal_weights['poisson']}  "
          f"XGBoost={optimal_weights['xgboost']}\n")

    # ---- Brier Score comparison -------------------------------------------
    naive_probs = np.full((len(labels), 3), 1.0 / 3.0)
    ens_probs   = w_opt * poi_probs + (1.0 - w_opt) * xgb_probs

    bs_ens   = _brier_score(ens_probs,   labels)
    bs_poi   = _brier_score(poi_probs,   labels)
    bs_xgb   = _brier_score(xgb_probs,  labels)
    bs_naive = _brier_score(naive_probs, labels)
    bs_best  = min(bs_ens, bs_poi, bs_xgb)

    print("Brier Score (walk-forward, out-of-sample):")
    print(f"  {'Modelo':<24} {'Brier':>7}  {'vs Naive':>9}  {'vs Poisson':>11}")
    print(f"  {'-'*55}")
    for name, bs in [
        (f"Ensemble (w={w_opt:.2f}/{1-w_opt:.2f})", bs_ens),
        ("Poisson (DC + decay)",                    bs_poi),
        ("XGBoost",                                 bs_xgb),
        ("Naive (1/3 cada)",                        bs_naive),
    ]:
        d_naive = bs - bs_naive
        d_poi   = bs - bs_poi
        marker  = "  <<< mejor" if bs == bs_best else ""
        print(f"  {name:<24} {bs:.4f}  {d_naive:>+9.4f}  {d_poi:>+11.4f}{marker}")

    # ---- Weight sensitivity sweep ----------------------------------------
    print(f"\nSensibilidad al peso de Poisson:")
    print(f"  {'w_poi':>6} {'w_xgb':>6} {'Brier':>8}")
    print(f"  {'-'*24}")
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        bs = _brier_w(w)
        marker = " <<" if abs(w - w_opt) < 0.06 else ""
        print(f"  {w:>6.1f} {1-w:>6.1f} {bs:>8.4f}{marker}")
    print(f"  {'optimo':>6}        {bs_ens:>8.4f}  (w={w_opt:.3f})")

    # ---- Train final ensemble on all data --------------------------------
    print(f"\nEntrenando ensemble final con todos los datos...")
    ensemble = EnsembleModel()
    ensemble.fit(matches, feat_df, weights=optimal_weights)
    bi = getattr(ensemble.xgboost.model, "best_iteration", None)
    n_trees = (bi + 1) if (bi is not None and bi >= 0) else 300
    print(f"  Poisson : rho={ensemble.poisson.rho:.4f}  "
          f"avg_goals={ensemble.poisson.avg_goals:.3f}  "
          f"home_adv={ensemble.poisson.home_advantage:.3f}")
    print(f"  XGBoost : {n_trees} arboles")

    # ---- Compare predictions: Real Madrid vs Barcelona -------------------
    HOME, AWAY = "Real Madrid", "Barcelona"
    print(f"\nPredicciones: {HOME} vs {AWAY}")
    print("-" * 60)

    h2h_mask = (
        (feat_df["home_team"] == HOME) & (feat_df["away_team"] == AWAY)
    ) | (
        (feat_df["home_team"] == AWAY) & (feat_df["away_team"] == HOME)
    )
    h2h_rows = feat_df[h2h_mask]

    if h2h_rows.empty:
        print(f"  No se encontraron partidos {HOME} vs {AWAY}.")
    else:
        last = h2h_rows.iloc[-1]
        X_all, _ = prepare_dataset(matches, feat_df)
        feat_row  = X_all.loc[[last.name]]

        actual      = last["result"]
        actual_code = _TARGET_MAP.get(actual, -1)

        poi_p = ensemble.poisson.predict_1x2(last["home_team"], last["away_team"])
        xgb_p = ensemble.xgboost.predict_proba(feat_row)[0]
        ens_p = ensemble.predict_proba(last["home_team"], last["away_team"], feat_row)

        print(f"  Ultimo partido : {last['home_team']} vs {last['away_team']}")
        print(f"  Fecha          : {last['match_date'].date()}")
        print(f"  Resultado real : {actual} "
              f"({int(last['home_goals'])}-{int(last['away_goals'])})\n")

        print(f"  {'Resultado':<18} {'Ensemble':>10} {'Poisson':>10} {'XGBoost':>10}")
        print(f"  {'-'*52}")
        poi_arr_match = [poi_p["home_win"], poi_p["draw"], poi_p["away_win"]]
        ens_arr_match = [ens_p["home_win"], ens_p["draw"], ens_p["away_win"]]
        for code in range(3):
            marker = "  <-- real" if code == actual_code else ""
            print(f"  {_TARGET_INV[code]:<18} "
                  f"{ens_arr_match[code]:>9.1%} "
                  f"{poi_arr_match[code]:>9.1%} "
                  f"{xgb_p[code]:>9.1%}{marker}")

        if actual_code >= 0:
            e_arr = np.array([ens_arr_match])
            p_arr = np.array([poi_arr_match])
            x_arr = xgb_p.reshape(1, 3)
            lab   = np.array([actual_code])
            print(f"\n  Brier Score (este partido):")
            print(f"    Ensemble : {_brier_score(e_arr, lab):.4f}")
            print(f"    Poisson  : {_brier_score(p_arr, lab):.4f}")
            print(f"    XGBoost  : {_brier_score(x_arr, lab):.4f}")

    # ---- Final summary ---------------------------------------------------
    winner = (
        "Ensemble" if bs_best == bs_ens
        else ("Poisson" if bs_best == bs_poi else "XGBoost")
    )
    print(f"\n{'='*52}")
    print("  RESUMEN FINAL")
    print(f"{'='*52}")
    print(f"  Pesos optimos  : Poisson {optimal_weights['poisson']}"
          f" / XGBoost {optimal_weights['xgboost']}")
    print(f"  Brier Ensemble : {bs_ens:.4f}")
    print(f"  Brier Poisson  : {bs_poi:.4f}")
    print(f"  Brier XGBoost  : {bs_xgb:.4f}")
    print(f"  Brier Naive    : {bs_naive:.4f}")
    imp_vs_poi = (bs_poi - bs_ens) / bs_poi * 100
    imp_vs_xgb = (bs_xgb - bs_ens) / bs_xgb * 100
    print(f"\n  Mejora Ensemble vs Poisson : {imp_vs_poi:+.2f}%")
    print(f"  Mejora Ensemble vs XGBoost : {imp_vs_xgb:+.2f}%")
    print(f"\n  Mejor modelo   : {winner} ({bs_best:.4f})")
    print(f"{'='*52}")
