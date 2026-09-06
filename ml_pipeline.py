"""
Tennis match prediction — machine-learning pipeline (Tier 2).
-----------------------------------------------------------------
Leak-free by construction. For every match we build features known
BEFORE the match starts:
  - Elo (global + surface) going into the match
  - rolling PRE-match serve/return averages per player (last N matches)
  - rank + rank points as of the match
  - head-to-head record before this match
  - surface, best_of

The target is randomized (player_a is not always the winner), so the
model must actually learn, not read the result off the row.

Serve-stat columns are 0% populated before ~1991. Where they're
missing, those features are NaN and XGBoost handles them natively —
so this runs on any year and gets stronger as stat-rich years are added.

Usage:
    import ml_pipeline as ml
    df = ml.load_matches(["atp_matches_2023.csv", "atp_matches_2024.csv"])
    result = ml.run(df)              # trains + evaluates vs Elo baseline
"""

import numpy as np
import pandas as pd
from collections import defaultdict, deque

# reuse the Elo engine as one of the features
import elo_engine as elo

SERVE_COLS = ["ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced"]
ROLL_N = 20   # rolling window for pre-match form


def load_matches(paths):
    """Load one or more Sackmann match CSVs, chronologically sorted."""
    frames = []
    for p in paths:
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    need = {"winner_name", "loser_name", "surface", "tourney_date"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df = df.dropna(subset=["winner_name", "loser_name", "surface"])
    df = df.sort_values("tourney_date", kind="stable").reset_index(drop=True)
    return df


def _rolling_mean(dq):
    return float(np.mean(dq)) if len(dq) else np.nan


def build_features(df, seed=1):
    """
    Walk matches in time order. For each, emit ONE row of pre-match features
    for a (player_a, player_b) pair with a randomized target y.
    All state is updated AFTER emitting the row -> no leakage.
    """
    rng = np.random.default_rng(seed)
    model = elo.EloModel(k=32, surf_weight=0.6)

    # rolling per-player serve-stat history (pre-match)
    hist = {c: defaultdict(lambda: deque(maxlen=ROLL_N)) for c in SERVE_COLS}
    h2h = defaultdict(lambda: [0, 0])   # (a,b) sorted -> [a_wins, b_wins]
    matches_seen = defaultdict(int)

    feats, targets = [], []

    have_stats = False
    for c in SERVE_COLS:
        col = f"w_{c}"
        if col in df.columns and df[col].notna().any():
            have_stats = True
            break

    for _, r in df.iterrows():
        w, l, surf = r["winner_name"], r["loser_name"], r["surface"]

        # --- assemble pre-match features for both players ---
        def player_block(p):
            return {c: _rolling_mean(hist[c][p]) for c in SERVE_COLS}

        wb, lb = player_block(w), player_block(l)
        w_elo = model.rating(w, surf)
        l_elo = model.rating(l, surf)
        w_rank = r.get("winner_rank", np.nan)
        l_rank = r.get("loser_rank", np.nan)

        key = tuple(sorted([w, l]))
        a_wins, b_wins = h2h[key]
        # h2h from winner's perspective
        w_h2h = a_wins if key[0] == w else b_wins
        l_h2h = b_wins if key[0] == w else a_wins

        # randomize which side is "A" so y isn't always 1
        if rng.random() < 0.5:
            a, b, y = w, l, 1
            a_elo, b_elo, ab, bb = w_elo, l_elo, wb, lb
            a_rank, b_rank, a_h2h, b_h2h = w_rank, l_rank, w_h2h, l_h2h
        else:
            a, b, y = l, w, 0
            a_elo, b_elo, ab, bb = l_elo, w_elo, lb, wb
            a_rank, b_rank, a_h2h, b_h2h = l_rank, w_rank, l_h2h, w_h2h

        row = {
            "elo_diff": a_elo - b_elo,
            "rank_diff": (b_rank - a_rank) if pd.notna(a_rank) and pd.notna(b_rank) else np.nan,
            "h2h_diff": a_h2h - b_h2h,
            "best_of": r.get("best_of", np.nan),
            "surface": surf,
        }
        # serve-stat differentials (pre-match rolling)
        for c in SERVE_COLS:
            row[f"{c}_diff"] = (ab[c] - bb[c]) if (ab[c] == ab[c] and bb[c] == bb[c]) else np.nan

        feats.append(row)
        targets.append(y)

        # --- update state AFTER emitting (so features stay pre-match) ---
        model.update(w, l, surf)
        if key[0] == w:
            h2h[key][0] += 1
        else:
            h2h[key][1] += 1
        matches_seen[w] += 1
        matches_seen[l] += 1
        if have_stats:
            for c in SERVE_COLS:
                wv, lv = r.get(f"w_{c}", np.nan), r.get(f"l_{c}", np.nan)
                if pd.notna(wv):
                    hist[c][w].append(wv)
                if pd.notna(lv):
                    hist[c][l].append(lv)

    X = pd.DataFrame(feats)
    X["surface"] = X["surface"].astype("category")
    y = np.array(targets)
    return X, y, have_stats


def run(df, test_frac=0.25, seed=1):
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

    X, y, have_stats = build_features(df, seed=seed)
    n = len(X)
    split = int(n * (1 - test_frac))
    # TIME-ORDERED split: train on the past, test on the future
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = y[:split], y[split:]

    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        enable_categorical=True, eval_metric="logloss",
    )
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]

    # Elo-only baseline = using just elo_diff through a logistic curve
    elo_p = 1 / (1 + 10 ** (-(Xte["elo_diff"].fillna(0)) / 400))

    # calibration: bucket test predictions vs actual outcomes
    calib = []
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        mask = (p >= lo) & (p < hi) if b < 9 else (p >= lo) & (p <= hi)
        if mask.sum():
            calib.append({"bucket": f"{b*10}-{b*10+10}%",
                          "avg_pred": float(p[mask].mean()),
                          "actual": float(yte[mask].mean()),
                          "n": int(mask.sum())})

    out = {
        "n_matches": n,
        "n_test": len(Xte),
        "have_stats": have_stats,
        "ml_accuracy": accuracy_score(yte, p > 0.5),
        "ml_logloss": log_loss(yte, p),
        "ml_brier": brier_score_loss(yte, p),
        "elo_accuracy": accuracy_score(yte, elo_p > 0.5),
        "elo_logloss": log_loss(yte, elo_p.clip(1e-9, 1 - 1e-9)),
        "elo_brier": brier_score_loss(yte, elo_p),
        "feature_importance": sorted(
            zip(X.columns, clf.feature_importances_), key=lambda t: -t[1])[:10],
        "calibration": calib,
        "predictor": Predictor(clf, df, seed=seed),
    }
    return out


class Predictor:
    """Holds the trained model + final per-player state for live matchup
    predictions. Rebuilds end-of-data Elo and rolling stats once."""
    def __init__(self, clf, df, seed=1):
        self.clf = clf
        self.model = elo.EloModel(k=32, surf_weight=0.6)
        self.hist = {c: defaultdict(lambda: deque(maxlen=ROLL_N)) for c in SERVE_COLS}
        self.h2h = defaultdict(lambda: [0, 0])
        self.rank = {}
        have_stats = any(f"w_{c}" in df.columns and df[f"w_{c}"].notna().any()
                         for c in SERVE_COLS)
        for _, r in df.iterrows():
            w, l, surf = r["winner_name"], r["loser_name"], r["surface"]
            self.model.update(w, l, surf)
            key = tuple(sorted([w, l]))
            self.h2h[key][0 if key[0] == w else 1] += 1
            if pd.notna(r.get("winner_rank", np.nan)): self.rank[w] = r["winner_rank"]
            if pd.notna(r.get("loser_rank", np.nan)):  self.rank[l] = r["loser_rank"]
            if have_stats:
                for c in SERVE_COLS:
                    if pd.notna(r.get(f"w_{c}", np.nan)): self.hist[c][w].append(r[f"w_{c}"])
                    if pd.notna(r.get(f"l_{c}", np.nan)): self.hist[c][l].append(r[f"l_{c}"])
        self.players = sorted(self.model.global_elo.keys())

    def predict(self, a, b, surface, best_of=3):
        ra, rb = self.model.rating(a, surface), self.model.rating(b, surface)
        key = tuple(sorted([a, b]))
        a_h2h = self.h2h[key][0 if key[0] == a else 1]
        b_h2h = self.h2h[key][1 if key[0] == a else 0]
        arank, brank = self.rank.get(a, np.nan), self.rank.get(b, np.nan)
        row = {
            "elo_diff": ra - rb,
            "rank_diff": (brank - arank) if pd.notna(arank) and pd.notna(brank) else np.nan,
            "h2h_diff": a_h2h - b_h2h,
            "best_of": best_of,
            "surface": surface,
        }
        for c in SERVE_COLS:
            av = np.mean(self.hist[c][a]) if len(self.hist[c][a]) else np.nan
            bv = np.mean(self.hist[c][b]) if len(self.hist[c][b]) else np.nan
            row[f"{c}_diff"] = (av - bv) if (av == av and bv == bv) else np.nan
        X = pd.DataFrame([row])
        X["surface"] = pd.Categorical(X["surface"], categories=["Hard", "Clay", "Grass", "Carpet"])
        return float(self.clf.predict_proba(X)[:, 1][0])
