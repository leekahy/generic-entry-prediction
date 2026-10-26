"""Flatten openFDA drugsfda records into an application-level table.

Usage: python3 src/02_build_fda.py <drugsfda json> <out csv>
"""
import json
import sys

import pandas as pd

src, out = sys.argv[1], sys.argv[2]
d = json.load(open(src))

rows = []
for r in d["results"]:
    app = r["application_number"]
    app_type = "".join(c for c in app if c.isalpha())
    orig = [s for s in r.get("submissions", []) if s.get("submission_type") == "ORIG" and s.get("submission_status") == "AP"]
    orig_date = min((s["submission_status_date"] for s in orig), default=None)
    products = r.get("products", [])
    ingr = sorted({";".join(sorted(a["name"].strip().upper() for a in p.get("active_ingredients", []))) for p in products})
    of = r.get("openfda", {})
    rows.append(
        {
            "application_number": app,
            "app_type": app_type,
            "sponsor": r.get("sponsor_name"),
            "orig_approval_date": orig_date,
            "ingredients": "|".join(ingr),
            "dosage_forms": "|".join(sorted({p.get("dosage_form") or "" for p in products})),
            "routes": "|".join(sorted({p.get("route") or "" for p in products})),
            "brand_names": "|".join(sorted({p.get("brand_name") or "" for p in products})),
            "marketing_status": "|".join(sorted({p.get("marketing_status") or "" for p in products})),
            "n_products": len(products),
            "pharm_class_epc": "|".join(of.get("pharm_class_epc", [])),
            "substance_name": "|".join(of.get("substance_name", [])),
        }
    )

df = pd.DataFrame(rows)
df["orig_approval_date"] = pd.to_datetime(df["orig_approval_date"], format="%Y%m%d", errors="coerce")
df.to_csv(out, index=False)
print(df.app_type.value_counts().to_string())
print("with approval date:", df.orig_approval_date.notna().sum())
