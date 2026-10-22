"""Universal Predictor Studio  -  run with:  streamlit run app.py"""
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

import engine as E

st.set_page_config(page_title="Universal Predictor Studio", page_icon="🧠", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;800&family=JetBrains+Mono:wght@500&display=swap');
html, body, [class*="css"] {font-family:'Sora',sans-serif;}
.stApp {background:linear-gradient(120deg,#060a14,#0b1530,#1a0f33,#060a14);background-size:300% 300%;animation:bg 22s ease infinite;color:#e8ecf8;}
@keyframes bg {0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
[data-testid="stSidebar"] {background:rgba(8,12,26,.88);border-right:1px solid rgba(25,240,180,.15);}
.hero h1 {font-weight:800;font-size:3rem;margin:0;background:linear-gradient(90deg,#19f0b4,#8b5cf6,#ffc857,#19f0b4);background-size:300% 100%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;animation:shine 6s linear infinite;}
@keyframes shine {to{background-position:300% 0}}
.hero p {color:#9fb0d6;font-size:1.05rem;margin:.2rem 0 1rem}
.stepper {display:flex;gap:6px;flex-wrap:wrap;margin:.4rem 0 1.4rem}
.node {padding:.45rem .8rem;border-radius:99px;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.12);font-size:.8rem;color:#9fb0d6;opacity:0;animation:pop .5s forwards,glow 2.4s ease-in-out infinite}
.node.done {border-color:#19f0b4;color:#19f0b4;animation:pop .5s forwards}
@keyframes pop {from{opacity:0;transform:translateY(10px) scale(.9)}to{opacity:1;transform:none}}
@keyframes glow {0%,100%{box-shadow:0 0 0 rgba(139,92,246,0)}50%{box-shadow:0 0 14px rgba(139,92,246,.7)}}
.badge {display:inline-block;font-family:'JetBrains Mono',monospace;font-size:.75rem;letter-spacing:.12em;color:#060a14;background:#19f0b4;padding:.2rem .6rem;border-radius:6px;margin-top:1.2rem}
.card {background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.09);border-radius:16px;padding:1rem 1.2rem;backdrop-filter:blur(8px);animation:pop .6s both}
.board {display:flex;flex-direction:column;gap:10px;margin:.6rem 0}
.brow {display:grid;grid-template-columns:200px 1fr 70px;align-items:center;gap:12px;animation:pop .5s both}
.bname {font-weight:600}.bval {font-family:'JetBrains Mono',monospace;color:#ffc857;text-align:right}
.btrack {background:rgba(255,255,255,.06);border-radius:99px;height:16px;overflow:hidden}
.bar {height:100%;border-radius:99px;transform-origin:left;background:linear-gradient(90deg,#8b5cf6,#4f8cff);animation:grow 1.1s cubic-bezier(.2,.9,.2,1) both}
.bar.win {background:linear-gradient(90deg,#ffc857,#19f0b4);box-shadow:0 0 18px rgba(255,200,87,.7);animation:grow 1.1s cubic-bezier(.2,.9,.2,1) both,pulse 1.8s ease-in-out infinite 1.2s}
@keyframes grow {from{transform:scaleX(0)}to{transform:scaleX(1)}}
@keyframes pulse {50%{filter:brightness(1.35)}}
.winner {text-align:center;padding:1.2rem;border-radius:18px;background:radial-gradient(circle at 50% 0,rgba(255,200,87,.25),rgba(255,255,255,.03));border:1px solid #ffc857;animation:pop .7s both}
.winner b {font-size:2rem;color:#ffc857}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

STEPS = ["Load", "First look", "Clean", "Statistics", "Plots", "Choose inputs", "5-model race", "Predict"]


def stepper(done):
    h = "".join(f'<span class="node{" done" if done else ""}" style="animation-delay:{i*.25}s">'
                f'{"✓ " if done else ""}{i+1}. {s}</span>' for i, s in enumerate(STEPS))
    return f'<div class="stepper">{h}</div>'


def badge(n, text):
    st.markdown(f'<span class="badge">STEP {n} · {text.upper()}</span>', unsafe_allow_html=True)


def board_html(rows):
    h = '<div class="board">'
    for i, (name, sc) in enumerate(sorted(rows, key=lambda r: -r[1])):
        w = max(0.0, min(1.0, sc)) * 100
        h += (f'<div class="brow" style="animation-delay:{i*.12}s"><div class="bname">{"👑 " if i == 0 else ""}{name}</div>'
              f'<div class="btrack"><div class="bar{" win" if i == 0 else ""}" style="width:{w:.1f}%;animation-delay:{i*.12+.2}s"></div></div>'
              f'<div class="bval">{sc:.3f}</div></div>')
    return h + "</div>"


def plot(fig, h=380):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=h, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data(show_spinner=False)
def read_table(data, name, sheet):
    if name.lower().endswith((".xlsx", ".xls", ".xlsm")):
        return pd.read_excel(io.BytesIO(data), sheet_name=sheet)
    return pd.read_csv(io.BytesIO(data))


RESULT_TPL = """
<style>*{box-sizing:border-box;font-family:Sora,Segoe UI,sans-serif}body{margin:0;color:#e8ecf8}
.c{text-align:center;padding:26px 14px;border-radius:20px;border:1px solid #19f0b4;background:radial-gradient(circle at 50% 0,rgba(25,240,180,.22),rgba(10,16,34,.95));animation:in .8s both;box-shadow:0 0 40px rgba(25,240,180,.2)}
@keyframes in{from{opacity:0;transform:scale(.92)}to{opacity:1;transform:none}}
.l{color:#9fb0d6;letter-spacing:.15em;font-size:12px}.v{font-size:60px;font-weight:800;color:#19f0b4;margin:6px 0}
.s{color:#9fb0d6;font-size:14px}.t{height:10px;background:rgba(255,255,255,.1);border-radius:9px;margin:18px 8px 4px;position:relative}
.f{height:100%;border-radius:9px;background:linear-gradient(90deg,#8b5cf6,#19f0b4);width:0;transition:width 1.4s cubic-bezier(.2,.9,.2,1)}
.p{display:flex;align-items:center;gap:10px;margin:7px 10px;font-size:14px}.p .n{width:110px;text-align:left}.p .k{flex:1;height:12px;background:rgba(255,255,255,.1);border-radius:9px;overflow:hidden}
.p .k i{display:block;height:100%;width:0;background:linear-gradient(90deg,#8b5cf6,#ffc857);transition:width 1.2s}</style>
<div class="c"><div class="l">__LABEL__</div><div class="v" id="v">__START__</div><div class="s">__SUB__</div>__EXTRA__</div>
<script>
const target=__NUM__,dec=__DEC__,el=document.getElementById('v');
if(__ANIM__){let t0=null;function step(t){if(!t0)t0=t;const p=Math.min((t-t0)/1300,1),e=1-Math.pow(1-p,4);
el.textContent=(target*e).toFixed(dec);if(p<1)requestAnimationFrame(step);else el.textContent=target.toFixed(dec);}requestAnimationFrame(step);}
setTimeout(()=>{document.querySelectorAll('[data-w]').forEach(x=>x.style.width=x.dataset.w+'%')},100);
</script>
"""


def regression_card(val, mae, lo, hi, unit_name):
    pct = 0 if hi == lo else float(np.clip((val - lo) / (hi - lo), 0, 1)) * 100
    extra = f'<div class="t"><div class="f" data-w="{pct:.0f}"></div></div><div class="s">position inside the range seen in your data ({lo:.3g} → {hi:.3g})</div>'
    dec = 2 if abs(val) < 100 else 1
    return (RESULT_TPL.replace("__LABEL__", f"PREDICTED {unit_name.upper()}").replace("__START__", "0")
            .replace("__SUB__", f"typical error on unseen data: ± {mae:.3g}").replace("__EXTRA__", extra)
            .replace("__NUM__", f"{val:.6g}").replace("__DEC__", str(dec)).replace("__ANIM__", "true"))


def class_card(label, labels, proba, target):
    bars = "".join(f'<div class="p"><div class="n">{l}</div><div class="k"><i data-w="{p*100:.0f}"></i></div><div>{p*100:.0f}%</div></div>'
                   for l, p in sorted(zip(labels, proba), key=lambda x: -x[1]))
    return (RESULT_TPL.replace("__LABEL__", f"PREDICTED {target.upper()}").replace("__START__", str(label))
            .replace("__SUB__", "model confidence").replace("__EXTRA__", bars)
            .replace("__NUM__", "0").replace("__DEC__", "0").replace("__ANIM__", "false"))


# ===================================================================== header
st.markdown('<div class="hero"><h1>Universal Predictor Studio</h1>'
            '<p>Drop any spreadsheet. Five models race. The winner predicts for you.</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 1 · Your data")
    up = st.file_uploader("Upload .xlsx / .csv", type=["xlsx", "xls", "xlsm", "csv"])

if up is None:
    st.markdown(stepper(False), unsafe_allow_html=True)
    st.markdown('<div class="card">👈 Upload a <b>.csv</b> or <b>.xlsx</b> file in the sidebar (UCS data, students data, anything with a column you want to predict). '
                'The app will clean it, pick the inputs, train 5 models, crown the best one and build a prediction dashboard automatically.</div>',
                unsafe_allow_html=True)
    st.stop()

data = up.getvalue()
sheet = 0
if up.name.lower().endswith((".xlsx", ".xls", ".xlsm")):
    names = pd.ExcelFile(io.BytesIO(data)).sheet_names
    with st.sidebar:
        sheet = st.selectbox("Sheet", names)
raw = read_table(data, up.name, sheet)
sig = f"{up.name}|{up.size}|{sheet}"
if st.session_state.get("sig") != sig:
    st.session_state.clear()
    st.session_state["sig"] = sig

cleaned, log = E.clean(raw)
with st.sidebar:
    st.markdown("### 2 · What to predict")
    target = st.selectbox("Target column (output)", list(cleaned.columns), index=len(cleaned.columns) - 1)
    work = cleaned.dropna(subset=[target]).reset_index(drop=True)
    auto_task = E.detect_task(work[target]) if len(work) else "regression"
    pick = st.radio("Problem type", ["Auto", "regression", "classification"], horizontal=True)
    task = auto_task if pick == "Auto" else pick
    st.caption(f"Using **{task}**" + (" (auto-detected)" if pick == "Auto" else ""))
    thr = st.slider("Drop inputs weaker than", 0.0, 0.5, 0.05, 0.01, help="Relevance 0 to 1 between an input and the target")

if len(work) < 10:
    st.error("Fewer than 10 usable rows after cleaning. Check the target column.")
    st.stop()
if task == "regression" and not E.is_num(work[target]):
    st.error("Regression needs a numeric target. Choose 'classification' or another column.")
    st.stop()

auto_sel, dropped, rel, warn = E.select_features(work, target, task, thr)
with st.sidebar:
    st.markdown("### 3 · Inputs")
    feats = st.multiselect("Inputs used by the models", [c for c in work.columns if c != target],
                           default=auto_sel, key=f"f_{sig}_{target}_{task}_{thr}",
                           help="Auto-chosen. Add or remove columns if you know better.")
    go_btn = st.button("🚀 Run the 5-model race", type="primary", use_container_width=True)

if go_btn:
    if not feats:
        st.error("Choose at least one input column.")
        st.stop()
    st.session_state.pop("res", None)
    st.markdown(stepper(False), unsafe_allow_html=True)
    prog, holder, rows = st.progress(0.0, "Training models…"), st.empty(), []

    def live(row, key):
        rows.append((row["Model"], row[key]))
        prog.progress(len(rows) / 5, f"Trained {row['Model']}")
        holder.markdown(board_html(rows), unsafe_allow_html=True)

    st.session_state["res"] = E.run_models(work, feats, target, task, live)
    st.session_state["meta"] = dict(feats=feats, spec=E.input_spec(work, feats), tmin=float(work[target].min()) if task == "regression" else 0,
                                    tmax=float(work[target].max()) if task == "regression" else 1)
    prog.empty(); holder.empty()

res = st.session_state.get("res")
st.markdown(stepper(res is not None), unsafe_allow_html=True)
tab_story, tab_race, tab_pred, tab_batch = st.tabs(["📖 Pipeline story", "🏆 Model race", "🎯 Live predictor", "📦 Batch predict"])

# ============================================================ PIPELINE STORY
with tab_story:
    badge(1, "Load"); st.markdown(f"**`{up.name}`** was read with `pandas`. It has **{raw.shape[0]} rows × {raw.shape[1]} columns**.")
    st.dataframe(raw.head(), use_container_width=True)

    badge(2, "First look")
    st.markdown("Before touching anything we look at the **data types, missing cells and duplicates**. A column that should be numeric but shows as `object`/`str` "
                "contains text such as `-`, `5 cups` or `12.5 hrs`, and a model cannot use it yet.")
    st.dataframe(pd.DataFrame({"type": raw.dtypes.astype(str), "missing": raw.isna().sum(), "unique values": raw.nunique()}), use_container_width=True)
    st.markdown(f"Duplicate rows found: **{int(raw.duplicated().sum())}**")

    badge(3, "Clean")
    st.markdown("Automatic cleaning rules: **remove duplicates → placeholders become real missing values (NaN) → text-numbers become numbers → label spellings are merged**. "
                f"Rows whose **target is missing are removed (the target is never filled in)**: {len(cleaned) - len(work)} removed.")
    st.dataframe(pd.DataFrame(log, columns=["What was fixed", "How many"]), use_container_width=True, hide_index=True)
    st.info("Fixes that need domain knowledge (minutes vs hours, feet vs cm, decimal slips like 245 instead of 24.5) can't be guessed by a universal tool. "
            "Look at the statistics below and fix those in your file if you see impossible values.")

    badge(4, "Statistics")
    st.markdown("Mean, spread and extremes of every numeric column. Impossible minimum/maximum values show up here.")
    st.dataframe(work.describe().T.round(2), use_container_width=True)
    if task == "regression":
        t = work[target]
        st.markdown(f"**{target}**: mean = {t.mean():.2f}, median = {t.median():.2f}, min = {t.min():.2f}, max = {t.max():.2f}, std = {t.std():.2f}")

    badge(5, "Plots")
    st.markdown("A picture of the **target** and of how the strongest inputs relate to it.")
    plot(px.histogram(work, x=target, color=None if task == "regression" else target, nbins=25, title=f"Distribution of {target}"), 300)
    top = [c for c in rel.index if c in feats][:2] or list(feats[:2])
    cols = st.columns(max(len(top), 1))
    for col, f in zip(cols, top):
        with col:
            fn = E.is_num(work[f])
            if fn and task == "regression": fig = px.scatter(work, x=f, y=target, opacity=.7)
            elif task == "regression": fig = px.box(work, x=f, y=target)
            elif fn: fig = px.box(work, x=target, y=f)
            else: fig = px.histogram(work, x=f, color=target, barmode="group")
            fig.update_layout(title=f"{target} vs {f}"); plot(fig, 330)

    badge(6, "Choose inputs")
    st.markdown("An input must be **relevant** to the target and must **not leak it**. Rules used: relevance (|Spearman ρ| for number-vs-number, correlation ratio or Cramér's V otherwise), "
                "**leakage** (almost identical to the target, or a column like *total/grade* calculated from it), **redundancy** (two inputs with |ρ| > 0.9: keep one), "
                "plus IDs, constants and columns that are mostly empty are dropped.")
    num_cols = [c for c in work.columns if E.is_num(work[c])]
    if len(num_cols) >= 2:
        corr = work[num_cols].corr(method="spearman")
        plot(px.imshow(corr.round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Spearman correlation heatmap"), 480)
    tbl = pd.DataFrame({"relevance": rel.round(3)}).reindex([c for c in work.columns if c != target])
    tbl["decision"] = ["✅ used" if c in feats else "❌ dropped" for c in tbl.index]
    tbl["reason"] = [dropped.get(c, "") if c not in feats else ("you added it manually" if c in dropped else "") for c in tbl.index]
    st.dataframe(tbl.sort_values("relevance", ascending=False), use_container_width=True)
    for w in warn:
        st.warning(f"`{w}` is suspiciously strong (relevance > 0.9). Make sure it is known *before* the outcome happens.")

    badge(7, "Models")
    if res is None:
        st.info("Press **🚀 Run the 5-model race** in the sidebar to continue the story.")
    else:
        tr = res["table"]; m = "R2" if res["task"] == "regression" else "Accuracy"
        st.markdown(f"**Split:** 80 % of the rows train the models, 20 % are kept as unseen **test** data. **Missing inputs** are filled with the *training* median "
                    "(most frequent label for text) and numbers are scaled, all inside one pipeline so the test data never leaks into the preparation.")
        st.markdown("**Baselines** are simple guesses every model must beat:")
        st.dataframe(pd.DataFrame(res["baselines"]).T.round(3), use_container_width=True)
        st.markdown("**Five models** were trained: a straight-line formula, a full decision tree, a 3-level tree, a random forest (many trees voting) and KNN (average of the 5 most similar rows). "
                    "Then **5-fold cross-validation** tests each one on 5 different slices of the data, so a lucky test split can't fool us.")
        st.dataframe(tr.round(3), use_container_width=True)
        dt = tr.loc["Decision Tree"]
        st.markdown(f"**Overfitting check:** the full Decision Tree scores {dt['Train ' + m]:.2f} on training rows but {dt['Test ' + m]:.2f} on unseen rows. "
                    "A big gap means it memorised instead of learning.")
        st.markdown(f"**Winner: {res['best']}** (highest cross-validated {'R²' if res['task'] == 'regression' else 'accuracy'}). It was then re-trained on *all* rows for the live predictor.")
        if res["task"] == "regression":
            yt, yp = res["y_test"], res["pred_test"]
            fig = px.scatter(x=yt, y=yp, labels={"x": "Actual", "y": "Predicted"}, title="Predicted vs actual (unseen test rows)")
            lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="perfect", line=dict(dash="dash", color="#ffc857")))
        else:
            fig = px.imshow(res["cm"], x=[str(l) for l in res["labels"]], y=[str(l) for l in res["labels"]], text_auto=True,
                            labels=dict(x="Predicted", y="Actual"), title="Confusion matrix (unseen test rows)", color_continuous_scale="Viridis")
        plot(fig, 420)
        imp = res["importance"].sort_values()
        plot(px.bar(x=imp.values, y=imp.index, orientation="h", labels={"x": "score drop when shuffled", "y": ""},
                    title="Which inputs matter most? (permutation importance)"), 320)

    badge(8, "Predict")
    st.markdown("Open the **🎯 Live predictor** tab: move the inputs and the answer updates instantly. Use **📦 Batch predict** for a whole new file.")

# ================================================================ MODEL RACE
with tab_race:
    if res is None:
        st.info("Set the target and inputs in the sidebar, then press **🚀 Run the 5-model race**.")
    else:
        key = res["key"]
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(f"#### Leaderboard · {key}")
            st.markdown(board_html([(n, v) for n, v in res["table"][key].items()]), unsafe_allow_html=True)
        with c2:
            ts = res["test_scores"]
            st.markdown(f'<div class="winner">👑<br><b>{res["best"]}</b><br>' + " · ".join(f"{k} {v:.3f}" for k, v in ts.items()) +
                        f'<br><small>on {len(res["y_test"])} unseen rows</small></div>', unsafe_allow_html=True)
        if res["table"][key].iloc[0] > 0.98:
            st.warning("A score this close to perfect is a warning sign. Check that no input is calculated from the target (data leakage).")
        if res["n"] < 30:
            st.warning(f"Only {res['n']} rows: scores are unstable. Treat the winner as a guide, not a proof.")

# =============================================================== LIVE PREDICT
with tab_pred:
    if res is None:
        st.info("Run the race first. The input form is built automatically from your columns.")
    else:
        meta, left, right = st.session_state["meta"], None, None
        left, right = st.columns([3, 2], gap="large")
        row = {}
        with left:
            st.markdown(f"#### Inputs · model: **{res['best']}**")
            grid = st.columns(2)
            for i, (f, sp) in enumerate(meta["spec"].items()):
                with grid[i % 2]:
                    if sp["kind"] == "num":
                        step = 1.0 if sp["int"] else max((sp["max"] - sp["min"]) / 200, 1e-6)
                        unk = st.checkbox("unknown", key=f"u_{f}", help="Leave this value out; the model uses a typical value")
                        v = st.slider(f, sp["min"], sp["max"], sp["default"], step, key=f"s_{f}", disabled=unk) if sp["max"] > sp["min"] else sp["default"]
                        row[f] = np.nan if unk else v
                    else:
                        row[f] = st.selectbox(f, sp["options"], key=f"s_{f}")
        with right:
            pred, proba = E.predict_rows(res, pd.DataFrame([row]))
            if res["task"] == "regression":
                html = regression_card(float(pred[0]), res["test_scores"]["MAE"], meta["tmin"], meta["tmax"], res["target"])
            else:
                html = class_card(pred[0], list(res["final"].classes_), proba[0], res["target"])
            components.html(html, height=430 if res["task"] == "classification" else 330)

# ============================================================== BATCH PREDICT
with tab_batch:
    if res is None:
        st.info("Run the race first.")
    else:
        st.markdown(f"Upload a new file with these columns: `{', '.join(res['feats'])}`. The winner ({res['best']}) fills in **{res['target']}** for every row.")
        nf = st.file_uploader("New data (.csv / .xlsx)", type=["xlsx", "xls", "csv"], key="batch")
        if nf is not None:
            newdf = read_table(nf.getvalue(), nf.name, 0)
            miss = [c for c in res["feats"] if c not in [str(x).strip() for x in newdf.columns]]
            if miss:
                st.error(f"Missing columns in the new file: {', '.join(miss)}")
            else:
                clean_new, _ = E.clean(newdf, train=False)
                p, pr = E.predict_rows(res, clean_new)
                out = newdf.copy()
                out[f"predicted_{res['target']}"] = np.round(p, 3) if res["task"] == "regression" else p
                if pr is not None:
                    out["confidence"] = pr.max(axis=1).round(3)
                st.dataframe(out, use_container_width=True)
                buf = io.BytesIO(); out.to_excel(buf, index=False)
                d1, d2 = st.columns(2)
                d1.download_button("⬇️ Download CSV", out.to_csv(index=False).encode(), "predictions.csv", use_container_width=True)
                d2.download_button("⬇️ Download Excel", buf.getvalue(), "predictions.xlsx", use_container_width=True)
