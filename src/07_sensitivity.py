"""Sensitivity analyses: alternative entry thresholds and excluding the 2014 cohort.

Reports, for each variant: N, median outcomes, OLS (HC3) coefficient on log1p(ANDA sponsors), and
Random Forest repeated-CV R2 (spec B) for each outcome.

Usage: python3 src/07_sensitivity.py <results dir> <atc.csv> <events_base.csv> <events_thr0.01.csv> <events_thr0.05.csv> <events_lag2.csv>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

res, atc = Path(sys.argv[1]), pd.read_csv(sys.argv[2])
variants = {"Main (2% threshold)": sys.argv[3], "1% threshold": sys.argv[4], "5% threshold": sys.argv[5],
            "Predictors at t0-2": sys.argv[6]}
ATC = {"C": "cardiovascular", "N": "nervous", "J": "anti-infective", "L": "antineoplastic", "A": "alimentary",
       "D": "dermatological"}
OUT = {"brand_share_t1": "Brand share", "log_gen_price_ratio_t1": "Log price ratio",
       "log_mol_spend_ratio_t1": "Log spending ratio"}


def prep(path):
    d = pd.read_csv(path).merge(atc, on="mol", how="left")
    d["lag"] = d.t0 - d.first_anda_year
    d = d[(d.fda_match != "none") & (d.n_anda_sponsors_t1 >= 1) & (d.lag <= 10) & (d.n_brand_products <= 5)].copy()
    forms, routes = d.nda_forms.fillna("").str.upper(), d.nda_routes.fillna("").str.upper()
    d["oral_solid"] = (forms.str.split("|").map(lambda fs: all(f.startswith(("TABLET", "CAPSULE")) for f in fs if f))
                       & routes.str.split("|").map(lambda rs: all(r == "ORAL" for r in rs if r))).astype(int)
    d["log_pre_spend"] = np.log(d.pre_spend)
    d["log_pre_cost_per_fill"] = np.log(d.pre_cost_per_fill)
    d["log_pre_clms_per_prscrbr"] = np.log(d.pre_clms_per_prscrbr)
    d["log_n_anda_t0"] = np.log1p(d.n_anda_sponsors_t0)
    d["log_gen_price_ratio_t1"] = np.log(d.gen_price_ratio_t1)
    d["log_mol_spend_ratio_t1"] = np.log(d.mol_spend_ratio_t1)
    for k, v in ATC.items():
        d[f"atc_{v}"] = (d.atc1 == k).astype(int)
    return d.reset_index(drop=True)


FEATS = ["log_pre_spend", "log_pre_cost_per_fill", "pre_fills_per_bene", "pre_fills_per_clm",
         "log_pre_clms_per_prscrbr", "pre_ge65_share", "pre_lis_share", "pre_growth", "market_age",
         "n_brand_products", "combination", "oral_solid", "t0"] + [f"atc_{v}" for v in ATC.values()] + ["log_n_anda_t0"]

data = {k: prep(v) for k, v in variants.items()}
data["Excluding 2014 cohort"] = data["Main (2% threshold)"].query("t0 >= 2015").reset_index(drop=True)

rows = []
for name, d in data.items():
    row = {"Variant": name, "N": len(d), "Median brand share t0+1": d.brand_share_t1.median(),
           "Median price ratio t0+1": d.gen_price_ratio_t1.median()}
    for y, lab in OUT.items():
        X = d[FEATS].astype(float).copy()
        X["pg_missing"] = X.pre_growth.isna().astype(int)
        X["pre_growth"] = X.pre_growth.fillna(0)
        fit = sm.OLS(d[y], sm.add_constant(X)).fit(cov_type="HC3")
        row[f"{lab}: OLS coef. on ANDA sponsors (SE)"] = f"{fit.params['log_n_anda_t0']:.3f} ({fit.bse['log_n_anda_t0']:.3f})"
        rf = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                           RandomForestRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.5,
                                                 random_state=42, n_jobs=-1))
        cv = cross_val_score(rf, d[FEATS].astype(float), d[y], scoring="r2",
                             cv=RepeatedKFold(n_splits=10, n_repeats=5, random_state=42))
        row[f"{lab}: RF CV R2"] = cv.mean()
    rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(res / "tables/sensitivity.csv", index=False)
print(out.round(3).T.to_string())

# LaTeX (transposed: rows = statistics, cols = variants)
t = out.set_index("Variant").T
with open(res / "tables/tableS1.tex", "w") as f:
    f.write("\\begin{tabular}{l" + "c" * t.shape[1] + "}\n\\toprule\n & " + " & ".join(c.replace("%", "\\%") for c in t.columns) + "\\\\\n\\midrule\n")
    for idx, r in t.iterrows():
        cells = [f"{v:.2f}" if isinstance(v, float) else str(v) for v in r]
        f.write(idx.replace("R2", "$R^2$") + " & " + " & ".join(cells) + "\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
