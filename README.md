# Generic entry in Medicare Part D — reproducible pipeline

Target journal : Therapeutic Innovation & Regulatory Science (Springer/DIA), Research Article.

## Reproduce
```
python3 src/01_download_cms.py data/raw/cms_catalog/data.json data/raw/cms_geo      # CMS Part D national rows 2013-2024
python3 src/02_build_fda.py data/raw/openfda/drug-drugsfda-0001-of-0001.json data/processed/fda_applications.csv
python3 src/03_build_events.py data/raw/cms_geo data/processed/fda_applications.csv data/processed/events.csv [threshold]
python3 src/04_fetch_atc.py data/processed/events.csv data/processed/atc.csv
python3 src/03_build_events.py data/raw/cms_geo data/processed/fda_applications.csv data/processed/events_lag2.csv 0.02 2   # predictors at t0-2
python3 src/08_manufacturers.py data/raw/cms_dsd_hist data/raw/cms_spending/DSD_DY24.csv data/processed/events.csv data/processed/mfr_counts.csv
python3 src/05_analysis.py data/processed/events.csv data/processed/atc.csv data/processed/mfr_counts.csv results
python3 src/06_figures.py results
python3 src/07_sensitivity.py results data/processed/atc.csv data/processed/events.csv data/processed/events_thr0.01.csv data/processed/events_thr0.05.csv data/processed/events_lag2.csv
python3 src/09_round2.py results   # benchmark comparison + classifier operating points
```
Raw inputs: CMS Part D Spending by Drug historical archive (dataset-resources of 7e0b4365-fd63-4a29-8f5e-e0ac9f66a81b, unzip into data/raw/cms_dsd_hist);  CMS catalog `https://data.cms.gov/data.json`; openFDA `https://download.open.fda.gov/drug/drugsfda/drug-drugsfda-0001-of-0001.json.zip`.

## Outputs
- `paper/manuscript.tex` (+ `tables/`, `figures/`) — compiled: `submission/manuscript.pdf`
- `submission/TIRS_manuscript_overleaf.zip` — upload to Overleaf as a new project
- `submission/cover_letter.pdf`

Variable definitions and modeling/tuning protocol: `docs/variables_and_protocol.md`.
