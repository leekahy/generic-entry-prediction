"""Realized generic competition from CMS Part D Spending by Drug manufacturer-level rows (2012-2024).

For each molecule and year, counts manufacturers with Part D claims for non-originator products
(any claims, and at least 1% / 5% of non-originator claims). Each year is taken from the most recent
release that contains it.

Usage: python3 src/08_manufacturers.py <cms_dsd_hist dir> <DSD_DY24.csv> <events.csv> <out csv>
"""
import glob
import re
import sys

import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
hist_dir, dy24, ev_path, out = sys.argv[1:5]

ABBR = {"hcl": "hydrochloride", "hbr": "hydrobromide", "na": "sodium", "k": "potassium", "ca": "calcium",
        "mg": "magnesium", "mesylat": "mesylate", "dihydrochlor": "dihydrochloride", "acetat": "acetate"}


def tokens(s):
    s = re.sub(r"[^a-z0-9/ ]", " ", str(s).lower())
    return [ABBR.get(t, t) for t in s.replace("/", " ").split()]


def ingr_key(s):
    return "/".join(sorted(" ".join(tokens(p)) for p in re.split(r"/", str(s)) if p.strip()))


def read_release(path):
    """Parse the 'Manufacturer Summary' sheet: year blocks laid out side by side."""
    raw = pd.read_excel(path, sheet_name="Manufacturer Summary", header=None)
    hdr_row = raw.index[raw[0].astype(str).str.strip() == "Brand Name"][0]
    yr_row = hdr_row - 1
    years = raw.loc[yr_row].ffill()
    hdr = raw.loc[hdr_row]
    body = raw.loc[hdr_row + 1:]
    keys = body.iloc[:, :3].copy()
    keys.columns = ["Brnd_Name", "Gnrc_Name", "Mftr_Name"]
    frames = []
    for y in sorted({int(re.search(r"\d{4}", str(v)).group()) for v in years.dropna() if re.search(r"\d{4}", str(v))}):
        cols = [c for c in raw.columns if re.sub(r"\.0$", "", str(years[c])).endswith(str(y)) and " ".join(str(hdr[c]).split()) == "Total Claims"]
        if not cols:
            continue
        f = keys.copy()
        f["year"] = y
        f["claims"] = pd.to_numeric(body[cols[0]], errors="coerce")
        frames.append(f)
    return pd.concat(frames)


releases = sorted(glob.glob(f"{hist_dir}/*DYT20*/*.xlsx"))
rows = []
for p in releases:
    rel = int(re.search(r"DYT(\d{2})", p).group(1)) + 2000
    df = read_release(p)
    df["release"] = rel
    rows.append(df)
    print(p.split("/")[-1], sorted(df.year.unique()))

# 2024 release CSV (manufacturer rows; same years as DYT2024 workbook, used for cross-check)
d24 = pd.read_csv(dy24)
long = []
for y in range(2020, 2025):
    f = d24[["Brnd_Name", "Gnrc_Name", "Mftr_Name"]].copy()
    f["year"], f["claims"], f["release"] = y, pd.to_numeric(d24[f"Tot_Clms_{y}"], errors="coerce"), 2025
    long.append(f)
allm = pd.concat(rows + long)
allm = allm[allm.Mftr_Name.notna() & (allm.Mftr_Name.astype(str).str.strip() != "Overall")]
# keep the most recent release for each year
allm = allm[allm.release == allm.groupby("year").release.transform("max")]
allm["mol"] = allm.Gnrc_Name.map(ingr_key)

ev = pd.read_csv(ev_path)
recs = []
for mol, t0, brands in ev[["mol", "t0", "orig_brands"]].itertuples(index=False):
    ob = set(str(brands).split("|"))
    m = allm[(allm.mol == mol) & ~allm.Brnd_Name.isin(ob) & (allm.claims > 0)]
    rec = {"mol": mol}
    for k, yy in (("t0", t0), ("t1", t0 + 1)):
        g = m[m.year == yy].groupby("Mftr_Name").claims.sum()
        tot = g.sum()
        rec[f"n_mfr_any_{k}"] = int((g > 0).sum())
        rec[f"n_mfr_1pct_{k}"] = int((g >= 0.01 * tot).sum()) if tot > 0 else 0
        rec[f"n_mfr_5pct_{k}"] = int((g >= 0.05 * tot).sum()) if tot > 0 else 0
        rec[f"gen_hhi_{k}"] = float(((g / tot) ** 2).sum()) if tot > 0 else float("nan")
    recs.append(rec)
res = pd.DataFrame(recs)
res.to_csv(out, index=False)
print(res.describe().round(2).to_string())
