"""
Surface-weighted Elo engine for tennis prediction.
Importable core shared by the CLI and the Streamlit app.
No printing here -> backtest() returns a results dict.
"""

import math, random
from collections import defaultdict

BASE_ELO = 1500


def expected(r_a, r_b):
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def _base_elo():
    # module-level (not a lambda) so EloModel stays picklable for st.cache_data
    return BASE_ELO


class EloModel:
    def __init__(self, k=32, surf_weight=0.60, surf_ramp=20):
        self.k = k
        self.sw = surf_weight
        self.ramp = surf_ramp
        self.global_elo = defaultdict(_base_elo)
        self.surf_elo = defaultdict(_base_elo)
        self.matches_played = defaultdict(int)

    def rating(self, player, surface):
        g = self.global_elo[player]
        s = self.surf_elo[(player, surface)]
        n = self.matches_played[(player, surface)]
        w = self.sw * min(n, self.ramp) / self.ramp
        return (1 - w) * g + w * s

    def predict(self, a, b, surface):
        return expected(self.rating(a, surface), self.rating(b, surface))

    def update(self, winner, loser, surface):
        gw, gl = self.global_elo[winner], self.global_elo[loser]
        e = expected(gw, gl)
        self.global_elo[winner] = gw + self.k * (1 - e)
        self.global_elo[loser] = gl + self.k * (0 - (1 - e))
        sw_, sl_ = self.surf_elo[(winner, surface)], self.surf_elo[(loser, surface)]
        es = expected(sw_, sl_)
        self.surf_elo[(winner, surface)] = sw_ + self.k * (1 - es)
        self.surf_elo[(loser, surface)] = sl_ + self.k * (0 - (1 - es))
        self.matches_played[(winner, surface)] += 1
        self.matches_played[(loser, surface)] += 1


def make_demo(n_players=120, n_matches=12000, seed=7):
    rnd = random.Random(seed)
    surfaces = ["Hard", "Clay", "Grass"]
    players = [f"Player_{i:03d}" for i in range(n_players)]
    true_skill = {p: rnd.gauss(0, 1) for p in players}
    surf_bonus = {(p, s): rnd.gauss(0, 0.4) for p in players for s in surfaces}
    rows = []
    for m in range(n_matches):
        a, b = rnd.sample(players, 2)
        s = rnd.choices(surfaces, weights=[5, 3, 1])[0]
        sa = true_skill[a] + surf_bonus[(a, s)]
        sb = true_skill[b] + surf_bonus[(b, s)]
        p_a = 1 / (1 + math.exp(-(sa - sb) * 1.6))
        w, l = (a, b) if rnd.random() < p_a else (b, a)
        rows.append((f"{m:06d}", w, l, s))
    return rows


def rows_from_dataframe(df):
    """Accepts a Sackmann-style DataFrame -> list of (date, winner, loser, surface)."""
    need = {"winner_name", "loser_name", "surface"}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV missing columns: {need - set(df.columns)}")
    date_col = "tourney_date" if "tourney_date" in df.columns else None
    sub = df[["winner_name", "loser_name", "surface"] + ([date_col] if date_col else [])].dropna(
        subset=["winner_name", "loser_name", "surface"])
    rows = []
    for _, r in sub.iterrows():
        d = str(r[date_col]) if date_col else "0"
        rows.append((d, r["winner_name"], r["loser_name"], r["surface"]))
    rows.sort(key=lambda x: x[0])
    return rows


def backtest(rows, k=32, surf_weight=0.60, warmup_frac=0.15, seed=1):
    """Walk-forward backtest. Returns a dict of metrics + per-match records."""
    model = EloModel(k=k, surf_weight=surf_weight)
    rnd = random.Random(seed)
    n = len(rows)
    warmup = int(n * warmup_frac)

    correct = tot = 0
    logloss = brier = 0.0
    bins = defaultdict(lambda: [0, 0.0, 0])   # bucket -> [target_hits, prob_sum, count]
    preds = []                                # (index, p, y) for scored matches

    for i, (_, w, l, surf) in enumerate(rows):
        if rnd.random() < 0.5:
            p, y = model.predict(w, l, surf), 1
        else:
            p, y = model.predict(l, w, surf), 0
        if i >= warmup:
            tot += 1
            correct += ((p > 0.5) == (y == 1))
            pc = min(max(p, 1e-9), 1 - 1e-9)
            logloss += -(y * math.log(pc) + (1 - y) * math.log(1 - pc))
            brier += (p - y) ** 2
            b = min(int(p * 10), 9)
            bins[b][0] += y; bins[b][1] += p; bins[b][2] += 1
            preds.append((i, p, y))
        model.update(w, l, surf)

    calib = []
    for b in range(10):
        hits, psum, cnt = bins[b]
        if cnt:
            calib.append({"bucket": f"{b*10}-{b*10+10}%",
                          "avg_pred": psum / cnt,
                          "actual": hits / cnt,
                          "n": cnt})

    top = sorted(model.global_elo.items(), key=lambda x: -x[1])[:15]

    return {
        "n_total": n,
        "n_scored": tot,
        "warmup": warmup,
        "accuracy": correct / tot if tot else 0,
        "logloss": logloss / tot if tot else 0,
        "brier": brier / tot if tot else 0,
        "calibration": calib,
        "top_players": top,
        "model": model,
    }
