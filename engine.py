"""Universal predictor engine: works on any table (UCS, Students, ...). No UI code here."""
import re
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

PLACEHOLDERS = {"-", "--", "n/a", "na", "?", "", "null", "none", "nan", "-999"}
NUM_RE = re.compile(r"^\s*-?\d+(\.\d+)?\s*[A-Za-z%°/]*\.?\s*$")   # "12.5 hrs", "3 cups", "45MPa"
ID_HINTS = {"id", "roll", "serial", "sr", "index", "sno"}
LEAK_HINTS = {"total", "grade", "result", "remark"}


def has_hint(name, hints):
    return any(tok in hints for tok in re.split(r"[^a-z0-9]+", name.lower()))


def is_num(s):
    return pd.api.types.is_numeric_dtype(s)


# ---------------------------------------------------------------- Step 3: clean
def clean(raw, train=True):
    df, log = raw.copy(), []
    df.columns = [str(c).strip() for c in df.columns]
    n = len(df)
    if train:   # for new files to predict we must keep every row
        df = df.dropna(how="all").dropna(axis=1, how="all").drop_duplicates()
    log.append(("Empty / duplicate rows removed", n - len(df)))
    ph = conv = merged = dates = 0
    for c in list(df.columns):
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            df = df.drop(columns=c); dates += 1; continue
        if is_num(s):
            m = s == -999
            ph += int(m.sum()); df.loc[m, c] = np.nan
            continue
        s = s.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        m = s.str.lower().isin(PLACEHOLDERS) & s.notna()
        ph += int(m.sum()); s = s.mask(m)
        nn = s.dropna()
        if len(nn) and nn.str.match(NUM_RE).mean() >= 0.85:          # text that is really a number
            df[c] = pd.to_numeric(s.str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce").astype(float)
            conv += 1
            continue
        key = s.str.lower()                                           # merge 'Civil'/'civil '/'CIVIL'
        canon = s.groupby(key).agg(lambda x: x.value_counts().index[0])
        new = key.map(canon)
        merged += int(s.nunique() - new.nunique())
        df[c] = new.astype(object).where(new.notna(), np.nan)
    log += [("Placeholders ('-', 'n/a', -999 ...) turned into missing values", ph),
            ("Text columns converted to numbers (e.g. '12.5 hrs' -> 12.5)", conv),
            ("Spelling/case variants of labels merged", merged),
            ("Date columns dropped (not used as inputs)", dates)]
    return df.reset_index(drop=True), log


def detect_task(y):
    if not is_num(y):
        return "classification"
    v = y.dropna()
    if v.nunique() <= 10 and (v % 1 == 0).all() and v.nunique() / max(len(v), 1) < 0.2:
        return "classification"
    return "regression"


# ------------------------------------------------- Step 6: relevance + selection
def _eta(num, cat):
    d = pd.DataFrame({"v": num, "g": cat}).dropna()
    if d.g.nunique() < 2 or d.v.var() == 0:
        return 0.0
    sst = ((d.v - d.v.mean()) ** 2).sum()
    ssb = ((d.groupby("g").v.transform("mean") - d.v.mean()) ** 2).sum()
    return float(np.sqrt(ssb / sst))


def _cramers(a, b):
    t = pd.crosstab(a, b)
    if min(t.shape) < 2:
        return 0.0
    return float(np.sqrt(chi2_contingency(t)[0] / (t.values.sum() * (min(t.shape) - 1))))


def relevance(df, f, target, task):
    fn, yn = is_num(df[f]), task == "regression"
    if fn and yn:
        r = abs(df[f].astype(float).corr(df[target].astype(float), method="spearman"))
    elif fn:
        r = _eta(df[f], df[target])
    elif yn:
        r = _eta(df[target], df[f])
    else:
        r = _cramers(df[f], df[target])
    return 0.0 if pd.isna(r) else float(r)


def select_features(df, target, task, thr=0.05):
    n, drop, keep = len(df), {}, []
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        if s.isna().mean() > 0.6:
            drop[c] = "more than 60% missing"
        elif s.nunique() <= 1:
            drop[c] = "constant: same value for everyone"
        elif not is_num(s) and s.nunique() > max(0.5 * n, 9):
            drop[c] = "ID / name-like text (every row different)"
        elif is_num(s) and s.nunique() == n and (s.dropna() % 1 == 0).all() and has_hint(c, ID_HINTS):
            drop[c] = "ID-like number"
        else:
            keep.append(c)
    rel = pd.Series({c: relevance(df, c, target, task) for c in keep}).sort_values(ascending=False)
    warn = []
    for c in list(rel.index):
        if rel[c] > 0.98:
            drop[c] = "DATA LEAKAGE: almost identical to the target"
        elif rel[c] >= 0.8 and has_hint(c, LEAK_HINTS):
            drop[c] = "possible DATA LEAKAGE: name suggests it is calculated from the target"
        elif rel[c] > 0.9:
            warn.append(c)
    rel_ok = rel[[c for c in rel.index if c not in drop]]
    weak = [c for c in rel_ok.index if rel_ok[c] < thr]
    if len(rel_ok) - len(weak) >= 2:
        for c in weak:
            drop[c] = f"weak relationship with target (< {thr})"
    kept = []
    for c in rel_ok.index:
        if c in drop:
            continue
        twin = next((k for k in kept if is_num(df[c]) and is_num(df[k]) and
                     abs(df[c].astype(float).corr(df[k].astype(float), method="spearman")) > 0.9), None)
        if twin:
            drop[c] = f"redundant: almost the same information as '{twin}'"
        else:
            kept.append(c)
    return kept, drop, rel, warn


# ------------------------------------------------------------ Step 7: models
def _pre(X):
    num = [c for c in X if is_num(X[c])]
    cat = [c for c in X if c not in num]
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore")), cat)])


def zoo(task, n_train):
    k = max(1, min(5, n_train))
    if task == "regression":
        return {"Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Decision Tree (depth 3)": DecisionTreeRegressor(max_depth=3, random_state=42),
                "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
                "KNN (k=5)": KNeighborsRegressor(n_neighbors=k)}
    return {"Logistic Regression": LogisticRegression(max_iter=3000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Decision Tree (depth 3)": DecisionTreeClassifier(max_depth=3, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
            "KNN (k=5)": KNeighborsClassifier(n_neighbors=k)}


def scores(task, yt, yp):
    if task == "regression":
        return {"MAE": float(mean_absolute_error(yt, yp)),
                "RMSE": float(np.sqrt(mean_squared_error(yt, yp))), "R2": float(r2_score(yt, yp))}
    return {"Accuracy": float(accuracy_score(yt, yp)),
            "F1": float(f1_score(yt, yp, average="weighted", zero_division=0))}


def run_models(df, feats, target, task, on_model=None):
    X, y = df[feats], df[target]
    reg = task == "regression"
    vc = None if reg else y.value_counts().min()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42,
                                          stratify=y if (not reg and vc >= 2) else None)
    if reg or vc < 2:
        cv = KFold(min(5, len(df)), shuffle=True, random_state=42)
    else:
        cv = StratifiedKFold(min(5, int(vc)), shuffle=True, random_state=42)
    scoring = {"r2": "r2", "rmse": "neg_root_mean_squared_error"} if reg else {"acc": "accuracy"}
    key = "CV R2" if reg else "CV Accuracy"
    if reg:
        base = {"Mean baseline": scores(task, yte, np.full(len(yte), ytr.mean())),
                "Median baseline": scores(task, yte, np.full(len(yte), ytr.median()))}
    else:
        base = {"Majority-class baseline": scores(task, yte, np.full(len(yte), ytr.mode()[0]))}
    rows, fitted = [], {}
    for name, m in zoo(task, len(Xtr)).items():
        pipe = Pipeline([("pre", _pre(X)), ("m", m)]).fit(Xtr, ytr)
        fitted[name] = pipe
        tr, te = scores(task, ytr, pipe.predict(Xtr)), scores(task, yte, pipe.predict(Xte))
        cvr = cross_validate(Pipeline([("pre", _pre(X)), ("m", clone(m))]), X, y, cv=cv, scoring=scoring)
        row = {"Model": name, **{f"Train {k}": v for k, v in tr.items()},
               **{f"Test {k}": v for k, v in te.items()}}
        if reg:
            row["CV RMSE"] = float(-cvr["test_rmse"].mean()); row[key] = float(cvr["test_r2"].mean())
        else:
            row[key] = float(cvr["test_acc"].mean())
        rows.append(row)
        if on_model:
            on_model(row, key)
    table = pd.DataFrame(rows).set_index("Model").sort_values(key, ascending=False)
    best = table.index[0]
    bp = fitted[best]
    pred = bp.predict(Xte)
    imp = permutation_importance(bp, Xte, yte, n_repeats=10, random_state=42,
                                 scoring="r2" if reg else "accuracy").importances_mean
    out = {"task": task, "table": table, "key": key, "best": best, "baselines": base, "feats": feats,
           "target": target, "y_test": yte.values, "pred_test": pred, "n": len(df),
           "importance": pd.Series(imp, index=feats).sort_values(ascending=False),
           "final": clone(bp).fit(X, y), "test_scores": scores(task, yte, pred)}
    if not reg:
        labels = sorted(y.unique(), key=str)
        out["labels"] = labels
        out["cm"] = confusion_matrix(yte, pred, labels=labels)
    return out


# ------------------------------------------------------- Step 8: predict new
def input_spec(df, feats):
    spec = {}
    for c in feats:
        s = df[c]
        if is_num(s):
            spec[c] = {"kind": "num", "min": float(s.min()), "max": float(s.max()),
                       "default": float(s.median()), "int": bool((s.dropna() % 1 == 0).all())}
        else:
            vc = s.value_counts()
            spec[c] = {"kind": "cat", "options": list(vc.index), "default": vc.index[0]}
    return spec


def predict_rows(res, rows_df):
    """rows_df has the feature columns; missing cells (NaN) are filled by the pipeline."""
    X = rows_df.reindex(columns=res["feats"])
    pred = res["final"].predict(X)
    proba = res["final"].predict_proba(X) if res["task"] == "classification" else None
    return pred, proba
