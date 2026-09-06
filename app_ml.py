"""
Tennis ML Predictor — Streamlit UI
Run with:  streamlit run app_ml.py

Wraps ml_pipeline.py (XGBoost). Upload one or more Sackmann
atp_matches_YYYY.csv / wta_matches_YYYY.csv files. The app trains a
leak-free model, compares it to an Elo baseline, shows calibration
and feature importance, and lets you predict any matchup.

Serve-stat features switch on automatically for 1991+ files.
"""
import io
import pandas as pd
import streamlit as st
import ml_pipeline as ml

st.set_page_config(page_title="Tennis ML Predictor", page_icon="🎾", layout="wide")

st.title("🎾 Tennis ML Predictor")
st.caption("XGBoost on Sackmann match data — leak-free, time-split, "
           "benchmarked against an Elo baseline. Upload match CSVs to train.")

with st.sidebar:
    st.header("Data")
    files = st.file_uploader("atp_matches / wta_matches CSV(s)",
                             type="csv", accept_multiple_files=True,
                             help="Real Sackmann match files. Upload several "
                                  "years for a stronger model. 1991+ adds serve stats.")
    st.divider()
    st.caption("Download files (paste in browser address bar):")
    st.code("huggingface.co/datasets/Aneeshers/"
            "tennis-sackmann-archive/resolve/main/"
            "atp/atp_matches_2024.csv", language=None)

if not files:
    st.info("⬅️ Upload one or more Sackmann match CSVs in the sidebar to train the model.")
    st.stop()

@st.cache_data(show_spinner=False)
def parse_files(file_bytes):
    frames = []
    report = []
    for name, b in file_bytes:
        try:
            d = pd.read_csv(io.BytesIO(b))
            if not {"winner_name", "loser_name", "surface"}.issubset(d.columns):
                report.append((name, 0, "not a match-results file")); continue
            frames.append(d); report.append((name, len(d), None))
        except Exception as e:
            report.append((name, 0, str(e)[:40]))
    if not frames:
        return None, report
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["winner_name", "loser_name", "surface"])
    df = df.sort_values("tourney_date", kind="stable").reset_index(drop=True)
    return df, report

@st.cache_resource(show_spinner="Training model…")
def train_model(df_key, _df):
    # df_key is a lightweight cache key (row count + col hash); _df is not hashed
    return ml.run(_df)

df, report = parse_files([(f.name, f.getvalue()) for f in files])
res = None
if df is not None:
    key = f"{len(df)}-{hash(tuple(df.columns))}-{df['tourney_date'].iloc[0]}-{df['tourney_date'].iloc[-1]}"
    res = train_model(key, df)

ok = [r for r in report if r[2] is None]
bad = [r for r in report if r[2] is not None]
if ok:
    st.success("Loaded: " + ", ".join(f"{n} ({c:,})" for n, c, _ in ok))
if bad:
    st.warning("Skipped: " + ", ".join(f"{n} ({e})" for n, _, e in bad))
if res is None:
    st.error("No valid match files found. Need winner_name, loser_name, surface columns.")
    st.stop()

# --------------------------------------------------------------- metrics
st.subheader("Model vs Elo baseline")
if not res["have_stats"]:
    st.caption("⚠️ These files have no serve stats (pre-1991). Add 1991+ years "
               "to switch on serve-stat features and widen the gap over Elo.")
c1, c2, c3 = st.columns(3)
lift = res["ml_accuracy"] - res["elo_accuracy"]
c1.metric("ML accuracy", f"{res['ml_accuracy']:.1%}", f"{lift:+.1%} vs Elo")
c2.metric("ML log-loss", f"{res['ml_logloss']:.4f}",
          f"{res['ml_logloss']-res['elo_logloss']:+.4f} vs Elo", delta_color="inverse")
c3.metric("Test matches", f"{res['n_test']:,}", help="Most recent slice, unseen in training")

st.divider()
tab_pred, tab_cal, tab_feat = st.tabs(["⚔️ Predict", "📊 Calibration", "🔑 Features"])

with tab_pred:
    pr = res["predictor"]
    players = pr.players
    if len(players) >= 2:
        ca, cb, cs = st.columns(3)
        pa = ca.selectbox("Player A", players, index=0)
        pb = cb.selectbox("Player B", players, index=1)
        surf = cs.selectbox("Surface", ["Hard", "Clay", "Grass", "Carpet"])
        bo = st.radio("Best of", [3, 5], horizontal=True)
        if pa != pb:
            p = pr.predict(pa, pb, surf, best_of=bo)
            st.metric(f"P({pa} beats {pb})", f"{p:.1%}")
            st.progress(p)
        else:
            st.warning("Pick two different players.")

with tab_cal:
    st.caption("Predicted probability vs actual win rate on the held-out test set.")
    cal = pd.DataFrame(res["calibration"])
    if not cal.empty:
        st.bar_chart(cal.set_index("bucket")[["avg_pred", "actual"]])

with tab_feat:
    st.caption("Which inputs the model relied on most.")
    fi = pd.DataFrame([(n, round(v, 3)) for n, v in res["feature_importance"] if v > 0],
                      columns=["Feature", "Importance"])
    st.dataframe(fi, hide_index=True, use_container_width=True)

st.divider()
st.caption("Elo remains a strong baseline; the ML edge grows with more stat-rich data. "
           "Beating bookmaker closing odds is the real test.")
