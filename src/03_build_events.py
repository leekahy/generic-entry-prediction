"""Build the generic-entry event dataset from the CMS Part D national panel and FDA application data.

A molecule (CMS generic name) has a generic-entry event in year t0 (2014-2023) when generic-named
products first reach GEN_THRESH of molecule claims, having stayed below it in every earlier year.
Predictors are measured in t0-1 (pre-entry); outcomes in t0+1 (first full year after entry).

Usage: python3 src/03_build_events.py <cms_geo dir> <fda_applications.csv> <out csv>
"""
import glob
import re
import sys

import numpy as np
import pandas as pd

cms_dir, fda_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
GEN_THRESH = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02
PRED_LAG = int(sys.argv[5]) if len(sys.argv) > 5 else 1  # predictors measured in t0 - PRED_LAG
MIN_PRE_CLAIMS = 1000

ABBR = {
    "hcl": "hydrochloride", "hbr": "hydrobromide", "na": "sodium", "k": "potassium", "ca": "calcium",
    "mg": "magnesium", "mesylat": "mesylate", "dihydrochlor": "dihydrochloride", "acetat": "acetate",
}


def tokens(s):
    s = re.sub(r"[^a-z0-9/ ]", " ", str(s).lower())
    return [ABBR.get(t, t) for t in s.replace("/", " ").split()]


def ingr_key(s, sep):
    parts = [" ".join(tokens(p)) for p in re.split(sep, str(s)) if p.strip()]
    return "/".join(sorted(parts))


def base_key(s, sep):
    """First word of each component (active moiety without salt)."""
    parts = [tokens(p)[0] for p in re.split(sep, str(s)) if tokens(p)]
    return "/".join(sorted(parts))


# ---------- CMS panel ----------
cms = pd.concat([pd.read_csv(f, low_memory=False) for f in sorted(glob.glob(f"{cms_dir}/*.csv"))])
num = ["Tot_Prscrbrs", "Tot_Clms", "Tot_30day_Fills", "Tot_Drug_Cst", "Tot_Benes", "GE65_Tot_Clms",
       "LIS_Bene_Cst_Shr", "NonLIS_Bene_Cst_Shr"]
for c in num:
    cms[c] = pd.to_numeric(cms[c], errors="coerce")
cms["mol"] = cms["Gnrc_Name"].map(lambda s: ingr_key(s, r"/"))
cms["is_generic"] = [set(tokens(b)) <= set(tokens(g)) for b, g in zip(cms.Brnd_Name, cms.Gnrc_Name)]

yr = cms.groupby(["mol", "year"]).agg(tot_clms=("Tot_Clms", "sum")).reset_index()
gen = cms[cms.is_generic].groupby(["mol", "year"]).Tot_Clms.sum().rename("gen_clms")
yr = yr.merge(gen, on=["mol", "year"], how="left").fillna({"gen_clms": 0})
yr["gen_share"] = yr.gen_clms / yr.tot_clms

events = []
for mol, g in yr.groupby("mol"):
    g = g.set_index("year").sort_index()
    above = g.index[g.gen_share >= GEN_THRESH]
    if len(above) == 0:
        continue
    t0 = int(above.min())
    if t0 < 2014 or t0 > 2023 or (t0 - 1) not in g.index:
        continue
    events.append((mol, t0))
ev = pd.DataFrame(events, columns=["mol", "t0"])
print("candidate events:", len(ev))

recs = []
for mol, t0 in ev.itertuples(index=False):
    m = cms[cms.mol == mol]
    pre = m[m.year == t0 - 1]
    orig_brands = set(pre.loc[~pre.is_generic, "Brnd_Name"])
    if not orig_brands:
        continue
    pre_b = pre[pre.Brnd_Name.isin(orig_brands)]
    if pre_b.Tot_Clms.sum() < MIN_PRE_CLAIMS:
        continue
    rec = {"mol": mol, "generic_name": pre.Gnrc_Name.iloc[0], "t0": t0,
           "orig_brands": "|".join(sorted(orig_brands)), "n_brand_products": len(orig_brands)}
    # pre-entry predictors (t0 - PRED_LAG; default t0-1)
    pred = m[(m.year == t0 - PRED_LAG) & m.Brnd_Name.isin(orig_brands)]
    if pred.empty:
        continue
    s = pred[num].sum()
    rec["denom_cost_per_fill"] = pre_b.Tot_Drug_Cst.sum() / pre_b.Tot_30day_Fills.sum()
    rec.update({
        "pre_spend": s.Tot_Drug_Cst, "pre_clms": s.Tot_Clms, "pre_benes": s.Tot_Benes,
        "pre_prscrbrs": s.Tot_Prscrbrs,
        "pre_cost_per_fill": s.Tot_Drug_Cst / s.Tot_30day_Fills,
        "pre_fills_per_bene": s.Tot_30day_Fills / s.Tot_Benes,
        "pre_fills_per_clm": s.Tot_30day_Fills / s.Tot_Clms,
        "pre_clms_per_prscrbr": s.Tot_Clms / s.Tot_Prscrbrs,
        "pre_ge65_share": s.GE65_Tot_Clms / s.Tot_Clms,
        "pre_lis_share": s.LIS_Bene_Cst_Shr / (s.LIS_Bene_Cst_Shr + s.NonLIS_Bene_Cst_Shr),
        "opioid": int((pre.Opioid_Drug_Flag == "Y").any()),
        "antipsychotic": int((pre.Antpsyct_Drug_Flag == "Y").any()),
        "antibiotic": int((pre.Antbtc_Drug_Flag == "Y").any()),
        "combination": int("/" in mol),
    })
    pre2 = m[(m.year == t0 - PRED_LAG - 1) & m.Brnd_Name.isin(orig_brands)]
    rec["pre_growth"] = (np.log(s.Tot_Clms / pre2.Tot_Clms.sum())
                         if len(pre2) and pre2.Tot_Clms.sum() > 0 else np.nan)
    # outcomes
    for k in (0, 1, 2):
        post = m[m.year == t0 + k]
        if post.empty:
            continue
        tot = post.Tot_Clms.sum()
        b = post[post.Brnd_Name.isin(orig_brands)]
        gnr = post[~post.Brnd_Name.isin(orig_brands)]
        rec[f"brand_share_t{k}"] = b.Tot_Clms.sum() / tot
        rec[f"brand_spend_share_t{k}"] = b.Tot_Drug_Cst.sum() / post.Tot_Drug_Cst.sum()
        if gnr.Tot_30day_Fills.sum() > 0:
            rec[f"gen_price_ratio_t{k}"] = (gnr.Tot_Drug_Cst.sum() / gnr.Tot_30day_Fills.sum()) / rec["denom_cost_per_fill"]
        rec[f"mol_spend_ratio_t{k}"] = post.Tot_Drug_Cst.sum() / pre.Tot_Drug_Cst.sum()
        rec[f"mol_clms_ratio_t{k}"] = tot / pre.Tot_Clms.sum()
        rec[f"mol_fills_ratio_t{k}"] = post.Tot_30day_Fills.sum() / pre.Tot_30day_Fills.sum()
        rec[f"mol_cost_per_fill_ratio_t{k}"] = ((post.Tot_Drug_Cst.sum() / post.Tot_30day_Fills.sum())
                                               / (pre.Tot_Drug_Cst.sum() / pre.Tot_30day_Fills.sum()))
    recs.append(rec)
df = pd.DataFrame(recs)
print("events with >= MIN_PRE_CLAIMS:", len(df))

# ---------- FDA linkage ----------
fda = pd.read_csv(fda_path, parse_dates=["orig_approval_date"])
fda = fda[fda.ingredients.notna()]
fda = fda.assign(ingredients=fda.ingredients.str.split("|")).explode("ingredients")
fda["key"] = fda.ingredients.map(lambda s: ingr_key(s, r";"))
fda["bkey"] = fda.ingredients.map(lambda s: base_key(s, r";"))
df["bkey"] = df.mol.map(lambda s: base_key(s, r"/"))


def fda_feats(row):
    f = fda[fda.key == row.mol]
    how = "exact"
    if f.empty:
        f = fda[fda.bkey == row.bkey]
        how = "base"
    if f.empty:
        return pd.Series({"fda_match": "none"})
    nda = f[f.app_type == "NDA"]
    anda = f[(f.app_type == "ANDA") & f.orig_approval_date.notna()].drop_duplicates("application_number")
    end_tm1 = pd.Timestamp(f"{row.t0 - 1}-12-31")
    end_t0 = pd.Timestamp(f"{row.t0}-12-31")
    end_t1 = pd.Timestamp(f"{row.t0 + 1}-12-31")
    forms = "|".join(nda.dosage_forms.dropna()).upper()
    routes = "|".join(nda.routes.dropna()).upper()
    return pd.Series({
        "fda_match": how,
        "nda_year": nda.orig_approval_date.min().year if nda.orig_approval_date.notna().any() else np.nan,
        "first_anda_year": anda.orig_approval_date.min().year if len(anda) else np.nan,
        "n_anda_sponsors_tm1": anda.loc[anda.orig_approval_date <= end_tm1, "sponsor"].nunique(),
        "n_anda_sponsors_t0": anda.loc[anda.orig_approval_date <= end_t0, "sponsor"].nunique(),
        "n_anda_sponsors_t1": anda.loc[anda.orig_approval_date <= end_t1, "sponsor"].nunique(),
        "nda_forms": forms, "nda_routes": routes,
        "epc": "|".join(sorted(set("|".join(nda.pharm_class_epc.dropna()).split("|")) - {""})),
    })


df = pd.concat([df, df.apply(fda_feats, axis=1)], axis=1)
df["market_age"] = df.t0 - df.nda_year
print(df.fda_match.value_counts().to_string())
df.to_csv(out, index=False)
print(df.t0.value_counts().sort_index().to_string())
