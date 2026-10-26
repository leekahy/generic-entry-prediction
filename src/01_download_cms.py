"""Download national-level rows of CMS Medicare Part D Prescribers by Geography and Drug (2013-2024).

Usage: python3 -I src/01_download_cms.py <catalog data.json> <out_dir>
"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

catalog, out_dir = sys.argv[1], Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

d = json.load(open(catalog))
api_by_year = {}
for ds in d["dataset"]:
    m = re.match(r"Medicare Part D Prescribers - by Geography and Drug : (\d{4})", ds["title"])
    if not m:
        continue
    apis = [x["accessURL"] for x in ds.get("distribution", []) if x.get("format") == "API"]
    if apis:
        api_by_year[int(m.group(1))] = apis[0]


def fetch(url):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            print("retry", attempt, e)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(url)


for year, api in sorted(api_by_year.items()):
    out = out_dir / f"partd_national_{year}.csv"
    if out.exists():
        continue
    rows, offset, size = [], 0, 5000
    while True:
        chunk = fetch(f"{api}?size={size}&offset={offset}&filter[Prscrbr_Geo_Lvl]=National")
        rows += chunk
        if len(chunk) < size:
            break
        offset += size
    df = pd.DataFrame(rows)
    df["year"] = year
    df.to_csv(out, index=False)
    print(year, len(df))
