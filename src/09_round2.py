"""Second-round revision analyses.

(a) Side-by-side temporal-validation comparison of the competition-only benchmark and machine-learning
    models (R2, RMSE, MAE with bootstrap CIs; paired differences vs the benchmark).
(b) Operating characteristics of the brand-retention (>=30%) classifier in the temporal test set, with the
    probability cut-off chosen on training data only (out-of-fold predictions).

Usage: python3 src/09_round2.py <results dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

res = Path(sys.argv[1])
tab = res / "tables"
SEED, N_BOOT = 42, 2000
rng = np.random.default_rng(SEED)

# ---------------- (a) temporal comparison ----------------
tp = pd.read_csv(tab / "temporal_predictions.csv")
OUTS = {"brand_share_t1": "Brand claims share", "log_gen_price_ratio_t1": "Log generic price ratio",
        "log_mol_spend_ratio_t1": "Log molecule spending ratio"}
MODELS = ["Competition-only linear", "OLS (all features)", "Elastic Net", "Random Forest", "XGBoost"]
n_test = tp.mol.nunique()
boot = [rng.integers(0, n_test, n_test) for _ in range(N_BOOT)]


def rmse(y, p):
    return mean_squared_error(y, p) ** 0.5


rows = []
for o in OUTS:
    for spec in ("S3", "S4"):
        sub = tp[(tp.outcome == o) & (tp.spec == spec)]
        ref = sub[sub.model == "Competition-only linear"].sort_values("mol")
        y, pr = ref.y.values, ref.pred.values
        for m in MODELS:
            s = sub[sub.model == m].sort_values("mol")
            p = s.pred.values
            r2 = [r2_score(y[i], p[i]) for i in boot]
            rm = [rmse(y[i], p[i]) for i in boot]
            ma = [mean_absolute_error(y[i], p[i]) for i in boot]
            d_r2 = [r2_score(y[i], p[i]) - r2_score(y[i], pr[i]) for i in boot]
            d_rm = [rmse(y[i], p[i]) - rmse(y[i], pr[i]) for i in boot]
            rows.append({"outcome": o, "spec": spec, "model": m,
                         "R2": r2_score(y, p), "R2_lo": np.quantile(r2, .025), "R2_hi": np.quantile(r2, .975),
                         "RMSE": rmse(y, p), "RMSE_lo": np.quantile(rm, .025), "RMSE_hi": np.quantile(rm, .975),
                         "MAE": mean_absolute_error(y, p), "MAE_lo": np.quantile(ma, .025), "MAE_hi": np.quantile(ma, .975),
                         "dR2": r2_score(y, p) - r2_score(y, pr), "dR2_lo": np.quantile(d_r2, .025), "dR2_hi": np.quantile(d_r2, .975),
                         "dRMSE": rmse(y, p) - rmse(y, pr), "dRMSE_lo": np.quantile(d_rm, .025), "dRMSE_hi": np.quantile(d_rm, .975)})
cmp_ = pd.DataFrame(rows)
cmp_.to_csv(tab / "temporal_comparison.csv", index=False)
print(cmp_[cmp_.spec == "S3"].round(3).to_string())

with open(tab / "table_cmp.tex", "w") as f:
    f.write("\\begin{tabular}{llcccc}\n\\toprule\nOutcome & Model & $R^2$ [95\\% CI] & RMSE [95\\% CI] & "
            "$\\Delta R^2$ vs benchmark [95\\% CI] & $\\Delta$RMSE vs benchmark [95\\% CI]\\\\\n\\midrule\n")
    for o, lab in OUTS.items():
        for i, m in enumerate(MODELS):
            r = cmp_[(cmp_.outcome == o) & (cmp_.spec == "S3") & (cmp_.model == m)].iloc[0]
            name = "Competition-only linear (benchmark)" if m == "Competition-only linear" else m
            dr2 = "--" if i == 0 else f"{r.dR2:+.2f} [{r.dR2_lo:+.2f}, {r.dR2_hi:+.2f}]"
            drm = "--" if i == 0 else f"{r.dRMSE:+.3f} [{r.dRMSE_lo:+.3f}, {r.dRMSE_hi:+.3f}]"
            f.write(f"{lab if i == 0 else ''} & {name} & {r.R2:.2f} [{r.R2_lo:.2f}, {r.R2_hi:.2f}] & "
                    f"{r.RMSE:.3f} [{r.RMSE_lo:.3f}, {r.RMSE_hi:.3f}] & {dr2} & {drm}\\\\\n")
        f.write("\\midrule\n" if o != list(OUTS)[-1] else "")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------------- (b) classifier operating characteristics ----------------
import json  # noqa: E402

d = pd.read_csv(tab / "analytic_sample.csv")
specs = json.load(open(tab / "spec_meta.json"))["specs"]
tr, te = (d.t0 <= 2019).values, (d.t0 >= 2020).values
yb = (d.brand_share_t1 >= 0.3).astype(int).values


def rf():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                         RandomForestClassifier(n_estimators=500, min_samples_leaf=3, max_features=0.5,
                                                random_state=SEED, n_jobs=-1))


def ops(y, p, cut):
    pred = (p >= cut).astype(int)
    tp_, fp = ((pred == 1) & (y == 1)).sum(), ((pred == 1) & (y == 0)).sum()
    fn, tn = ((pred == 0) & (y == 1)).sum(), ((pred == 0) & (y == 0)).sum()
    return {"sens": tp_ / (tp_ + fn) if tp_ + fn else np.nan, "specif": tn / (tn + fp) if tn + fp else np.nan,
            "ppv": tp_ / (tp_ + fp) if tp_ + fp else np.nan, "npv": tn / (tn + fn) if tn + fn else np.nan,
            "flagged": int(pred.sum()), "tp": int(tp_), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


op_rows = []
for spec in ("S1", "S3"):
    X = d[specs[spec]].astype(float)
    # cut-offs from out-of-fold probabilities within the training period only
    oof = cross_val_predict(rf(), X[tr], yb[tr], cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                            method="predict_proba")[:, 1]
    fpr, tpr, thr = roc_curve(yb[tr], oof)
    cuts = {"Youden (training)": thr[np.argmax(tpr - fpr)],
            "Specificity $\\geq$0.80 (training)": thr[np.where(fpr <= 0.20)[0].max()]}
    p_te = rf().fit(X[tr], yb[tr]).predict_proba(X[te])[:, 1]
    y_te = yb[te]
    for name, cut in cuts.items():
        base = ops(y_te, p_te, cut)
        bs = pd.DataFrame([ops(y_te[i], p_te[i], cut) for i in boot])
        row = {"spec": spec, "rule": name, "cutoff": cut, "n_test": len(y_te), "positives": int(y_te.sum()), **base}
        for k in ("sens", "specif", "ppv", "npv"):
            row[f"{k}_lo"], row[f"{k}_hi"] = bs[k].quantile(.025), bs[k].quantile(.975)
        op_rows.append(row)
op = pd.DataFrame(op_rows)
op.to_csv(tab / "classifier_operating_points.csv", index=False)
print(op.round(3).to_string())

with open(tab / "table_ops.tex", "w") as f:
    f.write("\\begin{tabular}{llcccccc}\n\\toprule\nSpecification & Cut-off rule (probability) & Flagged & "
            "Sensitivity & Specificity & PPV & NPV\\\\\n\\midrule\n")
    for r in op.itertuples():
        f.write(f"{r.spec} & {r.rule} ({r.cutoff:.2f}) & {r.flagged}/{r.n_test} & "
                f"{r.sens:.2f} [{r.sens_lo:.2f}, {r.sens_hi:.2f}] & {r.specif:.2f} [{r.specif_lo:.2f}, {r.specif_hi:.2f}] & "
                f"{r.ppv:.2f} [{r.ppv_lo:.2f}, {r.ppv_hi:.2f}] & {r.npv:.2f} [{r.npv_lo:.2f}, {r.npv_hi:.2f}]\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
