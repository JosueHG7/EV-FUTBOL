"""
models/xgboost_model.py

XGBoost multiclass classifier for football match outcome prediction (H/D/A).

Uses the 33 features produced by feature_engineering.py with a strict
walk-forward training protocol — the model never sees data from the future
of any evaluation window.
"""

import sys
from datetime import timedelta
from pathlib import Path

import contextlib
import io
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).parent.parent))
from models.feature_engineering import build_features
from models.poisson_model import PoissonModel, load_matches

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TARGET_MAP = {"H": 0, "D": 1, "A": 2}
_TARGET_INV = {0: "Home win (H)", 1: "Draw (D)", 2: "Away win (A)"}

_ID_COLS = frozenset({
    "home_team", "away_team", "home_goals", "away_goals",
    "match_date", "league_id", "season", "result",
})

_H2H_PROB_COLS = ["h2h_home_wins", "h2h_draws", "h2h_away_wins"]
_POISSON_COLS  = [
    "poisson_home_win", "poisson_draw", "poisson_away_win",
    "lambda_home", "lambda_away",
]


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_dataset(
    matches_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build X (feature matrix) and y (encoded targets) from build_features() output.

    Imputation strategy
    -------------------
    - H2H win/draw/away proportions : 0.33  (uniform prior)
    - H2H avg_goals                 : global column median
    - Poisson columns               : global column mean
    - Remaining nulls (form/venue/fatigue) : per-league median, then global median

    Parameters
    ----------
    matches_df  : raw match data returned by load_matches()
    features_df : output of build_features(matches_df)

    Returns
    -------
    X : DataFrame, shape (n, 33), NaN-free
    y : Series of int — 0=H, 1=D, 2=A
    """
    df = features_df.copy()
    df["match_date"] = pd.to_datetime(df["match_date"])

    y = df["result"].map(_TARGET_MAP).astype(int)

    feat_cols = [c for c in df.columns if c not in _ID_COLS]
    X = df[feat_cols].copy()

    # H2H probability proportions → uniform prior
    for col in _H2H_PROB_COLS:
        if col in X.columns:
            X[col] = X[col].fillna(0.33)

    # H2H avg_goals → global median
    if "h2h_avg_goals" in X.columns:
        X["h2h_avg_goals"] = X["h2h_avg_goals"].fillna(X["h2h_avg_goals"].median())

    # Poisson columns → global column mean
    for col in _POISSON_COLS:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].mean())

    # Remaining nulls → per-league median, fallback to global median
    X["_league"] = df["league_id"].values
    for col in feat_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(
                X.groupby("_league")[col].transform("median")
            ).fillna(X[col].median())
    X = X.drop(columns=["_league"])

    return X, y


# ---------------------------------------------------------------------------
# Brier Score
# ---------------------------------------------------------------------------

def _brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Multiclass Brier Score.
    BS = mean_i( sum_k (f_ik - o_ik)^2 ).  Range [0, 2]; lower is better.
    """
    n = len(labels)
    onehot = np.zeros((n, 3))
    onehot[np.arange(n), labels] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


# ---------------------------------------------------------------------------
# XGBoostModel
# ---------------------------------------------------------------------------

class XGBoostModel:
    """
    Thin wrapper around XGBClassifier for 3-class match outcome prediction.
    """

    def __init__(self) -> None:
        self.model: XGBClassifier | None = None
        self.feature_names: list[str] = []
        self.importances: pd.Series | None = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        """
        Train XGBClassifier with early stopping.

        If X_val / y_val are not provided the last 20% of X_train (by row
        order, which should be chronological) is used as the validation set.
        """
        self.feature_names = list(X_train.columns)

        self.model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            early_stopping_rounds=20,
            verbosity=0,
        )

        if X_val is not None and y_val is not None:
            fit_X, fit_y = X_train, y_train
            eval_set = [(X_val.values, y_val.values)]
        else:
            n_val  = max(1, int(len(X_train) * 0.2))
            fit_X  = X_train.iloc[:-n_val]
            fit_y  = y_train.iloc[:-n_val]
            eval_set = [(X_train.iloc[-n_val:].values,
                         y_train.iloc[-n_val:].values)]

        _sink = io.StringIO()
        with contextlib.redirect_stdout(_sink), contextlib.redirect_stderr(_sink):
            self.model.fit(fit_X.values, fit_y.values, eval_set=eval_set)

        self.importances = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names,
        ).sort_values(ascending=False)

    # ------------------------------------------------------------------
    # predict_proba
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns shape (n, 3) array: [p_home_win, p_draw, p_away_win]."""
        if self.model is None:
            raise RuntimeError("Model not fitted — call fit() first.")
        return self.model.predict_proba(X.values)

    # ------------------------------------------------------------------
    # get_feature_importance
    # ------------------------------------------------------------------

    def get_feature_importance(self, top_n: int = 15) -> pd.Series:
        """Top N features sorted by importance (descending)."""
        if self.importances is None:
            raise RuntimeError("Model not fitted — call fit() first.")
        return self.importances.head(top_n)


# ---------------------------------------------------------------------------
# Walk-forward training
# ---------------------------------------------------------------------------

def walk_forward_train(
    matches_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Strict walk-forward training protocol.

    Timeline
    --------
    - Warmup  : first 12 months — training only, no evaluation
    - Retrain : every 4 weeks, using ALL data up to that point
    - Evaluate: the following 4-week window (out-of-sample)

    Early stopping uses an internal 20% tail of each training window
    (never the evaluation window) to avoid leakage.

    Returns
    -------
    DataFrame with: period_start, period_end, n_matches,
                    brier_xgb, brier_naive
    """
    X, y = prepare_dataset(matches_df, features_df)
    dates = pd.to_datetime(features_df["match_date"])

    min_date   = dates.min()
    max_date   = dates.max()
    warmup_end = min_date + timedelta(days=365)
    step       = timedelta(weeks=4)

    records: list[dict] = []
    cursor = warmup_end

    while cursor + step <= max_date:
        train_mask = dates <= cursor
        eval_mask  = (dates > cursor) & (dates <= cursor + step)

        if train_mask.sum() < 100 or eval_mask.sum() < 5:
            cursor += step
            continue

        model = XGBoostModel()
        model.fit(X[train_mask], y[train_mask])

        probs  = model.predict_proba(X[eval_mask])
        labels = y[eval_mask].values

        records.append({
            "period_start": cursor,
            "period_end":   cursor + step,
            "n_matches":    int(eval_mask.sum()),
            "brier_xgb":    round(_brier_score(probs, labels), 4),
            "brier_naive":  round(_brier_score(
                                np.full((len(labels), 3), 1.0 / 3.0), labels
                            ), 4),
        })

        cursor += step

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Cargando partidos de la BD...")
    matches = load_matches()
    print(f"  {len(matches)} partidos | {matches['league_id'].nunique()} ligas\n")

    print("Construyendo features (orden cronologico, sin data leakage)...")
    feat_df = build_features(matches)

    print("Preparando dataset...")
    X, y = prepare_dataset(matches, feat_df)
    print(f"  {len(X)} muestras | {len(X.columns)} features")
    dist = y.value_counts().sort_index()
    print(f"  Distribucion: H={dist.get(0,0)} ({dist.get(0,0)/len(y):.1%})  "
          f"D={dist.get(1,0)} ({dist.get(1,0)/len(y):.1%})  "
          f"A={dist.get(2,0)} ({dist.get(2,0)/len(y):.1%})\n")

    # ---- Walk-forward evaluation (XGBoost) --------------------------------
    print("Walk-forward training XGBoost (reentrenamiento cada 4 semanas)...")
    wf = walk_forward_train(matches, feat_df)
    mean_brier_xgb   = wf["brier_xgb"].mean()
    mean_brier_naive = wf["brier_naive"].mean()
    print(f"  Periodos evaluados : {len(wf)}")
    print(f"  Partidos eval.     : {wf['n_matches'].sum()}")
    print(f"  Brier XGBoost      : {mean_brier_xgb:.4f}")
    print(f"  Brier Naive (1/3)  : {mean_brier_naive:.4f}\n")

    # ---- Walk-forward evaluation (Poisson) --------------------------------
    print("Calculando Brier Score walk-forward para Poisson (referencia)...")
    dates_s    = pd.to_datetime(feat_df["match_date"])
    cursor_poi = dates_s.min() + timedelta(days=365)
    step_poi   = timedelta(weeks=4)
    poi_briers: list[float] = []

    while cursor_poi + step_poi <= dates_s.max():
        tr_mask = (dates_s <= cursor_poi).values
        ev_mask = ((dates_s > cursor_poi) & (dates_s <= cursor_poi + step_poi)).values

        if tr_mask.sum() < 100 or ev_mask.sum() < 5:
            cursor_poi += step_poi
            continue

        pm = PoissonModel()
        try:
            pm._fit_core(matches[tr_mask].reset_index(drop=True), 1.0)
        except ValueError:
            cursor_poi += step_poi
            continue

        probs_poi, labels_poi = [], []
        for _, r in feat_df[ev_mask].iterrows():
            try:
                p = pm.predict_1x2(r["home_team"], r["away_team"])
                probs_poi.append([p["home_win"], p["draw"], p["away_win"]])
                labels_poi.append(_TARGET_MAP[r["result"]])
            except (KeyError, RuntimeError):
                pass

        if probs_poi:
            poi_briers.append(
                _brier_score(np.array(probs_poi), np.array(labels_poi))
            )

        cursor_poi += step_poi

    mean_brier_poi = float(np.mean(poi_briers)) if poi_briers else float("nan")
    print(f"  Brier Poisson      : {mean_brier_poi:.4f}\n")

    # ---- Final model on all data ------------------------------------------
    print("Entrenando modelo final con todos los datos...")
    final = XGBoostModel()
    final.fit(X, y)
    bi = getattr(final.model, "best_iteration", None)
    n_trees = (bi + 1) if (bi is not None and bi >= 0) else 300
    print(f"  Arboles usados: {n_trees}\n")

    # ---- Top 15 features --------------------------------------------------
    print("Top 15 features por importancia:")
    top = final.get_feature_importance(15)
    max_imp = top.max()
    for feat, imp in top.items():
        bar = "#" * int(imp / max_imp * 30)
        print(f"  {feat:<42} {imp:.4f}  {bar}")

    # ---- XGBoost vs Poisson: Real Madrid vs Barcelona ---------------------
    HOME, AWAY = "Real Madrid", "Barcelona"
    print(f"\nComparacion XGBoost vs Poisson — {HOME} vs {AWAY}")
    print("-" * 58)

    h2h_mask = (
        (feat_df["home_team"] == HOME) & (feat_df["away_team"] == AWAY)
    ) | (
        (feat_df["home_team"] == AWAY) & (feat_df["away_team"] == HOME)
    )
    h2h_rows = feat_df[h2h_mask]

    if h2h_rows.empty:
        print(f"  No se encontraron partidos {HOME} vs {AWAY} en el dataset.")
    else:
        last = h2h_rows.iloc[-1]
        feat_row = X.loc[[last.name]]

        xgb_probs = final.predict_proba(feat_row)[0]

        poi_final = PoissonModel()
        poi_final.fit(matches)
        poi_p = poi_final.predict_1x2(HOME, AWAY)
        poi_probs = np.array([poi_p["home_win"], poi_p["draw"], poi_p["away_win"]])

        print(f"  Ultimo partido : {last['home_team']} vs {last['away_team']}")
        print(f"  Fecha          : {last['match_date'].date()}")
        print(f"  Resultado real : {last['result']} "
              f"({last['home_goals']}-{last['away_goals']})\n")

        actual_code = _TARGET_MAP.get(last["result"], -1)
        print(f"  {'Resultado':<18} {'XGBoost':>10} {'Poisson':>10}")
        print(f"  {'-'*40}")
        for code, label in _TARGET_INV.items():
            marker = "  <-- real" if code == actual_code else ""
            print(f"  {label:<18} {xgb_probs[code]:>9.1%} {poi_probs[code]:>9.1%}{marker}")

        if actual_code >= 0:
            bs_xgb = _brier_score(xgb_probs.reshape(1, 3), np.array([actual_code]))
            bs_poi = _brier_score(poi_probs.reshape(1, 3), np.array([actual_code]))
            print(f"\n  Brier Score (este partido):")
            print(f"    XGBoost : {bs_xgb:.4f}")
            print(f"    Poisson : {bs_poi:.4f}")

    # ---- Summary ----------------------------------------------------------
    print(f"\n{'='*48}")
    print("  RESUMEN BRIER SCORE (walk-forward)")
    print(f"{'='*48}")
    print(f"  XGBoost : {mean_brier_xgb:.4f}")
    print(f"  Poisson : {mean_brier_poi:.4f}")
    print(f"  Naive   : {mean_brier_naive:.4f}")
    improvement = (mean_brier_poi - mean_brier_xgb) / mean_brier_poi * 100
    print(f"\n  Mejora XGBoost vs Poisson: {improvement:+.1f}%")
    print(f"{'='*48}")
