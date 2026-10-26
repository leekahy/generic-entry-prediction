"""Fetch ATC level-1 classes for each event molecule from the NLM RxClass API.

Usage: python3 src/04_fetch_atc.py <events.csv> <out csv>
"""
import json
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

ev, out = pd.read_csv(sys.argv[1]), sys.argv[2]
BASE = "https://rxnav.nlm.nih.gov/REST/rxclass/class/byDrugName.json"


def atc(name):
    url = f"{BASE}?drugName={urllib.parse.quote(name)}&relaSource=ATC"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                d = json.load(r)
            break
        except Exception:  # noqa: BLE001
            time.sleep(2)
    else:
        return []
    items = d.get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", [])
    return sorted({i["rxclassMinConceptItem"]["classId"] for i in items})


rows = []
for mol, gname in ev[["mol", "generic_name"]].drop_duplicates().itertuples(index=False):
    # try full name, then each component's first word (active moiety)
    codes = atc(gname)
    if not codes:
        for comp in mol.split("/"):
            codes += atc(comp.split()[0])
    rows.append({"mol": mol, "atc_codes": "|".join(sorted(set(codes))),
                 "atc1": sorted({c[0] for c in codes})[0] if codes else ""})
    time.sleep(0.1)
df = pd.DataFrame(rows)
df.to_csv(out, index=False)
print((df.atc1 != "").mean(), df.atc1.value_counts().to_string())
