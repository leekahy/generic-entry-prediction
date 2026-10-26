"""Main analysis (revised): sample, feature-timing specifications, predictive models with uncertainty,
classification, SHAP, OLS (HC3), spending decomposition, and spline dose-response.

Usage: python3 src/05_analysis.py <events.csv> <atc.csv> <mfr_counts.csv> <results dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import shap
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LinearRegression, LogisticRegressionCV
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

ev_path, atc_path, mfr_path, res = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
(res / "tables").mkdir(parents=True, exist_ok=True)
SEED = 42
N_BOOT = 2000
rng = np.random.default_rng(SEED)

# ---------------- sample ----------------
d = (pd.read_csv(ev_path).merge(pd.read_csv(atc_path), on="mol", how="left")
     .merge(pd.read_csv(mfr_path), on="mol", how="left"))
d["lag"] = d.t0 - d.first_anda_year
flow = [("Molecules with first material generic uptake in 2014-2023 (>=1,000 pre-entry brand claims)", len(d))]
d = d[d.fda_match != "none"]; flow.append(("Linked to FDA Drugs@FDA application records", len(d)))
d = d[d.n_anda_sponsors_t1 >= 1]; flow.append(("At least one ANDA approved by end of t0+1 (independent generic entry)", len(d)))
d = d[d.lag <= 10]; flow.append(("First ANDA approval within 10 years before observed entry", len(d)))
d = d[d.n_brand_products <= 5]; flow.append(("At most five originator brand products (excludes branded-generic families)", len(d)))
d = d.reset_index(drop=True)
pd.DataFrame(flow, columns=["step", "n"]).to_csv(res / "tables/sample_flow.csv", index=False)

# ---------------- features ----------------
ATC_GROUPS = {"C": "Cardiovascular", "N": "Nervous system", "J": "Anti-infective", "L": "Antineoplastic/immunomod.",
              "A": "Alimentary/metabolic", "D": "Dermatological"}
d["atc_group"] = d.atc1.map(ATC_GROUPS).fillna("Other")
forms = d.nda_forms.fillna("").str.upper()
routes = d.nda_routes.fillna("").str.upper()
d["oral_solid"] = (forms.str.split("|").map(lambda fs: all(f.startswith(("TABLET", "CAPSULE")) for f in fs if f))
                   & routes.str.split("|").map(lambda rs: all(r == "ORAL" for r in rs if r))).astype(int)
d["log_pre_spend"] = np.log(d.pre_spend)
d["log_pre_cost_per_fill"] = np.log(d.pre_cost_per_fill)
d["log_pre_clms_per_prscrbr"] = np.log(d.pre_clms_per_prscrbr)
d["log_n_anda_tm1"] = np.log1p(d.n_anda_sponsors_tm1)
d["log_n_anda_t0"] = np.log1p(d.n_anda_sponsors_t0)
d["log_n_mfr_t0"] = np.log1p(d.n_mfr_1pct_t0)
d["log_gen_price_ratio_t1"] = np.log(d.gen_price_ratio_t1)
d["log_mol_spend_ratio_t1"] = np.log(d.mol_spend_ratio_t1)
d["log_mol_fills_ratio_t1"] = np.log(d.mol_fills_ratio_t1)
d["log_mol_cost_per_fill_ratio_t1"] = np.log(d.mol_cost_per_fill_ratio_t1)
for g in ATC_GROUPS.values():
    d[f"atc_{g.split('/')[0].split()[0].lower()}"] = (d.atc_group == g).astype(int)
atc_cols = [c for c in d.columns if c.startswith("atc_") and c not in ("atc_codes", "atc_group")]

BASE = ["log_pre_spend", "log_pre_cost_per_fill", "pre_fills_per_bene", "pre_fills_per_clm",
        "log_pre_clms_per_prscrbr", "pre_ge65_share", "pre_lis_share", "pre_growth", "market_age",
        "n_brand_products", "combination", "oral_solid", "t0"] + atc_cols
# Feature-timing specifications (prediction date in brackets)
SPECS = {
    "S1": BASE,                          # pre-entry market features [start of entry year]
    "S2": BASE + ["log_n_anda_tm1"],     # + ANDA sponsors approved by end of t0-1 [start of entry year]
    "S3": BASE + ["log_n_anda_t0"],      # + ANDA sponsors approved by end of t0 [end of entry year]
    "S4": BASE + ["log_n_mfr_t0"],       # + manufacturers with Part D claims in t0 [after entry-year claims]
}
COMP = {"S2": "log_n_anda_tm1", "S3": "log_n_anda_t0", "S4": "log_n_mfr_t0"}
SPEC_LABELS = {"S1": "S1: Pre-entry features", "S2": "S2: + approved sponsors by t0-1",
               "S3": "S3: + approved sponsors by t0", "S4": "S4: + marketed manufacturers in t0"}
LABELS = {
    "log_pre_spend": "Pre-entry brand spending (log)", "log_pre_cost_per_fill": "Pre-entry cost per 30-day fill (log)",
    "pre_fills_per_bene": "30-day fills per beneficiary", "pre_fills_per_clm": "30-day fills per claim",
    "log_pre_clms_per_prscrbr": "Claims per prescriber (log)", "pre_ge65_share": "Share of claims age 65+",
    "pre_lis_share": "LIS share of patient cost", "pre_growth": "Pre-entry claims growth",
    "market_age": "Years since NDA approval", "n_brand_products": "No. of originator brand products",
    "combination": "Fixed-dose combination", "oral_solid": "Oral solid dosage form", "t0": "Entry year",
    "log_n_anda_tm1": "Approved ANDA sponsors by t0-1 (log1p)",
    "log_n_anda_t0": "Approved ANDA sponsors by t0 (log1p)",
    "log_n_mfr_t0": "Marketed generic manufacturers in t0 (log1p)", "atc_cardiovascular": "ATC: cardiovascular",
    "atc_nervous": "ATC: nervous system", "atc_anti-infective": "ATC: anti-infective",
    "atc_antineoplastic": "ATC: antineoplastic/immunomod.", "atc_alimentary": "ATC: alimentary/metabolic",
    "atc_dermatological": "ATC: dermatological", "pre_growth_missing": "Pre-entry growth missing",
}
OUTCOMES = {"brand_share_t1": "Brand claims share, t0+1", "log_gen_price_ratio_t1": "Log generic-to-brand price ratio, t0+1",
            "log_mol_spend_ratio_t1": "Log molecule spending ratio, t0+1 vs t0-1"}
json.dump({"specs": SPECS, "labels": LABELS, "spec_labels": SPEC_LABELS}, open(res / "tables/spec_meta.json", "w"), indent=1)


def imp():
    return SimpleImputer(strategy="median", add_indicator=True)


def models(spec):
    m = {"Mean baseline": None}
    if spec in COMP:
        m["Competition-only linear"] = LinearRegression()   # single predictor, no imputation needed
    m.update({
        "OLS (all features)": make_pipeline(imp(), LinearRegression()),
        "Elastic Net": make_pipeline(imp(), StandardScaler(),
                                     ElasticNetCV(l1_ratio=[.1, .5, .9, 1], cv=5, random_state=SEED, max_iter=50000)),
        "Random Forest": make_pipeline(imp(), RandomForestRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.5,
                                                                    random_state=SEED, n_jobs=-1)),
        "XGBoost": XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8,
                                colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0, random_state=SEED, n_jobs=4),
    })
    return m


def fit_predict(name, mdl, X_tr, y_tr, X_te, spec):
    if mdl is None:
        return np.full(len(X_te), y_tr.mean())
    if name == "Competition-only linear":
        c = COMP[spec]
        return mdl.fit(X_tr[[c]], y_tr).predict(X_te[[c]])
    return mdl.fit(X_tr, y_tr).predict(X_te)


def clip(y_name, p):
    return np.clip(p, 0, 1) if y_name == "brand_share_t1" else p


def metrics(y, p):
    return {"RMSE": mean_squared_error(y, p) ** 0.5, "MAE": mean_absolute_error(y, p), "R2": r2_score(y, p)}


perf, temporal_preds = [], []
tr_t, te_t = (d.t0 <= 2019).values, (d.t0 >= 2020).values
boot_idx = [rng.integers(0, te_t.sum(), te_t.sum()) for _ in range(N_BOOT)]
for y_name in OUTCOMES:
    for spec, feats in SPECS.items():
        X, y = d[feats].astype(float), d[y_name].values
        # repeated 10-fold CV
        rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=SEED)
        scores = {m: [] for m in models(spec)}
        for tr, te in rkf.split(X):
            for name, mdl in models(spec).items():
                p = clip(y_name, fit_predict(name, mdl, X.iloc[tr], y[tr], X.iloc[te], spec))
                scores[name].append(metrics(y[te], p))
        for name, sc in scores.items():
            sc = pd.DataFrame(sc)
            perf.append({"outcome": y_name, "spec": spec, "model": name, "validation": "CV",
                         "R2": sc.R2.mean(), "RMSE": sc.RMSE.mean(), "MAE": sc.MAE.mean(),
                         "R2_sd": sc.R2.std(), "R2_lo": sc.R2.quantile(.025), "R2_hi": sc.R2.quantile(.975)})
        # temporal validation (train 2014-2019, test 2020-2023); hyperparameters fixed a priori
        yte = y[te_t]
        preds = {}
        for name, mdl in models(spec).items():
            preds[name] = clip(y_name, fit_predict(name, mdl, X[tr_t], y[tr_t], X[te_t], spec))
            temporal_preds.append(pd.DataFrame({"outcome": y_name, "spec": spec, "model": name,
                                                "mol": d.mol[te_t].values, "y": yte, "pred": preds[name]}))
        ref = "Competition-only linear" if spec in COMP else "Mean baseline"
        for name, p in preds.items():
            mt = metrics(yte, p)
            bs = np.array([r2_score(yte[i], p[i]) for i in boot_idx])
            diff = np.array([r2_score(yte[i], p[i]) - r2_score(yte[i], preds[ref][i]) for i in boot_idx])
            loo = [r2_score(np.delete(yte, j), np.delete(p, j)) for j in range(len(yte))]
            top5 = np.argsort(-np.abs(yte - p))[:5]
            perf.append({"outcome": y_name, "spec": spec, "model": name, "validation": "Temporal", **mt,
                         "R2_lo": np.quantile(bs, .025), "R2_hi": np.quantile(bs, .975),
                         "dR2_vs_ref": mt["R2"] - r2_score(yte, preds[ref]), "dR2_lo": np.quantile(diff, .025),
                         "dR2_hi": np.quantile(diff, .975), "ref": ref, "LOO_R2_min": min(loo), "LOO_R2_max": max(loo),
                         "R2_drop_top5": r2_score(np.delete(yte, top5), np.delete(p, top5))})
        print(y_name, spec, "done", flush=True)
perf = pd.DataFrame(perf)
perf.to_csv(res / "tables/model_performance.csv", index=False)
pd.concat(temporal_preds).to_csv(res / "tables/temporal_predictions.csv", index=False)

# ---------------- brand retention classification (threshold sweep) ----------------
def classifiers():
    return {
        "Elastic Net logistic": make_pipeline(imp(), StandardScaler(),
                                              LogisticRegressionCV(Cs=10, penalty="elasticnet", solver="saga",
                                                                   l1_ratios=[.1, .5, .9], cv=5, max_iter=20000,
                                                                   scoring="roc_auc", random_state=SEED)),
        "Random Forest": make_pipeline(imp(), RandomForestClassifier(n_estimators=500, min_samples_leaf=3, max_features=0.5,
                                                                     random_state=SEED, n_jobs=-1)),
    }


cls = []
for thr in (0.1, 0.2, 0.3, 0.4, 0.5):
    yb = (d.brand_share_t1 >= thr).astype(int).values
    for spec in ("S1", "S3", "S4"):
        X = d[SPECS[spec]].astype(float)
        sc = {m: [] for m in classifiers()}
        for tr, te in RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=SEED).split(X, yb):
            for name, mdl in classifiers().items():
                p = mdl.fit(X.iloc[tr], yb[tr]).predict_proba(X.iloc[te])[:, 1]
                sc[name].append({"AUC": roc_auc_score(yb[te], p), "Brier": brier_score_loss(yb[te], p)})
        # temporal AUC with bootstrap CI
        for name, mdl in classifiers().items():
            v = pd.DataFrame(sc[name])
            p = mdl.fit(X[tr_t], yb[tr_t]).predict_proba(X[te_t])[:, 1]
            yt = yb[te_t]
            bs = [roc_auc_score(yt[i], p[i]) for i in boot_idx if 0 < yt[i].sum() < len(i)]
            cls.append({"threshold": thr, "prevalence": yb.mean(), "spec": spec, "model": name,
                        "AUC": v.AUC.mean(), "AUC_sd": v.AUC.std(), "Brier": v.Brier.mean(),
                        "Brier_null": yb.mean() * (1 - yb.mean()),
                        "AUC_temporal": roc_auc_score(yt, p) if 0 < yt.sum() < len(yt) else np.nan,
                        "AUC_temporal_lo": np.quantile(bs, .025), "AUC_temporal_hi": np.quantile(bs, .975)})
    print("threshold", thr, "done", flush=True)
cls = pd.DataFrame(cls)
cls.to_csv(res / "tables/classification_performance.csv", index=False)

# ---------------- SHAP (XGBoost, full sample) ----------------
shap_out = {}
for spec in ("S3", "S4"):
    shap_out[spec] = {}
    for y_name in OUTCOMES:
        X = d[SPECS[spec]].astype(float)
        mdl = models(spec)["XGBoost"].fit(X, d[y_name])
        sv = shap.TreeExplainer(mdl).shap_values(X)
        imp_s = pd.Series(np.abs(sv).mean(0), index=SPECS[spec]).sort_values(ascending=False)
        shap_out[spec][y_name] = imp_s.rename(index=LABELS).round(4).to_dict()
json.dump(shap_out, open(res / "tables/shap_importance.json", "w"), indent=1)

# ---------------- OLS with HC3 robust SEs (associational benchmark) ----------------
OLS_OUT = list(OUTCOMES) + ["log_mol_fills_ratio_t1", "log_mol_cost_per_fill_ratio_t1"]
ols_rows = []
for spec in ("S2", "S3", "S4"):
    for y_name in OLS_OUT:
        X = d[SPECS[spec]].astype(float).copy()
        X["pre_growth_missing"] = X.pre_growth.isna().astype(int)
        X["pre_growth"] = X.pre_growth.fillna(0)
        fit = sm.OLS(d[y_name], sm.add_constant(X)).fit(cov_type="HC3")
        for k in fit.params.index:
            ols_rows.append({"spec": spec, "outcome": y_name, "term": LABELS.get(k, k), "coef": fit.params[k],
                             "se": fit.bse[k], "p": fit.pvalues[k]})
        ols_rows.append({"spec": spec, "outcome": y_name, "term": "N", "coef": fit.nobs})
        ols_rows.append({"spec": spec, "outcome": y_name, "term": "Adj. R2", "coef": fit.rsquared_adj})
pd.DataFrame(ols_rows).to_csv(res / "tables/ols_hc3.csv", index=False)

# ---------------- dose-response: natural cubic spline in log1p(competitors) ----------------
curves = []
for spec, var, raw in (("S3", "log_n_anda_t0", "n_anda_sponsors_t0"), ("S4", "log_n_mfr_t0", "n_mfr_1pct_t0")):
    covs = [c for c in SPECS[spec] if c != var]
    X = d[covs].astype(float).copy()
    X["pre_growth_missing"] = X.pre_growth.isna().astype(int)
    X["pre_growth"] = X.pre_growth.fillna(0)
    basis = patsy.dmatrix("cr(x, df=3) - 1", {"x": d[var]}, return_type="dataframe")
    basis.columns = [f"s{i}" for i in range(basis.shape[1])]
    design = sm.add_constant(pd.concat([X, basis], axis=1))
    for y_name in ("log_gen_price_ratio_t1", "log_mol_spend_ratio_t1", "brand_share_t1"):
        fit = sm.OLS(d[y_name], design).fit(cov_type="HC3")
        hi = int(np.percentile(d[raw], 95))
        grid = np.arange(0, hi + 1)
        bg = patsy.build_design_matrices([basis.design_info if hasattr(basis, "design_info") else
                                          patsy.dmatrix("cr(x, df=3) - 1", {"x": d[var]}).design_info],
                                         {"x": np.log1p(grid)})[0]
        Xg = pd.DataFrame(np.repeat(design.drop(columns=list(basis.columns)).mean().values[None, :], len(grid), 0),
                          columns=design.drop(columns=list(basis.columns)).columns)
        Xg = pd.concat([Xg, pd.DataFrame(np.asarray(bg), columns=basis.columns)], axis=1)[design.columns]
        pr = fit.get_prediction(Xg).summary_frame(alpha=0.05)
        for n, (m, lo, hi_) in zip(grid, pr[["mean", "mean_ci_lower", "mean_ci_upper"]].values):
            curves.append({"spec": spec, "measure": raw, "outcome": y_name, "n": n, "pred": m, "lo": lo, "hi": hi_})
pd.DataFrame(curves).to_csv(res / "tables/dose_response.csv", index=False)

d.to_csv(res / "tables/analytic_sample.csv", index=False)
show = perf.pivot_table(index=["outcome", "spec", "model"], columns="validation", values="R2").round(3)
print(show.to_string())
print(perf[perf.validation == "Temporal"][["outcome", "spec", "model", "R2", "R2_lo", "R2_hi", "dR2_vs_ref", "dR2_lo", "dR2_hi", "LOO_R2_min", "LOO_R2_max", "R2_drop_top5"]].round(3).to_string())
print(cls.round(3).to_string())
