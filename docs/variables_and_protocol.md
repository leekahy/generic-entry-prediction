# Variable definitions and modeling protocol

## Unit of analysis and event definition
- **Molecule**: CMS generic name (active ingredient or fixed-dose combination), normalized (lower case; HCl→hydrochloride, etc.).
- **Generic-named product**: all words of the CMS brand name appear in the generic name (independent and authorized generics marketed under the nonproprietary name).
- **Entry year t0**: first year (2014–2023) in which generic-named products reach ≥2% of molecule Part D claims, with the share below 2% in all earlier years (sensitivity: 1%, 5%).
- **Originator products**: all non-generic-named products of the molecule dispensed in t0−1.
- **Exclusions**: <1,000 originator claims in t0−1; no FDA linkage; no final ANDA approval by end of t0+1; first ANDA approval >10 years before t0; >5 originator products.

## Outcomes (measured in t0+1)
| Variable | Definition |
|---|---|
| `brand_share_t1` | Originator claims / molecule claims |
| `log_gen_price_ratio_t1` | log[(non-originator gross cost per 30-day fill in t0+1) / (originator gross cost per 30-day fill in t0−1)] |
| `log_mol_spend_ratio_t1` | log(molecule gross spending t0+1 / t0−1) = `log_mol_fills_ratio_t1` + `log_mol_cost_per_fill_ratio_t1` |
| Brand retention | `brand_share_t1` ≥ threshold (0.1–0.5; main 0.2 and 0.3) |

## Predictors and information timing
| Variable | Source | Measured | Observable at |
|---|---|---|---|
| Pre-entry market features (spending, cost per fill, fills per beneficiary/claim, claims per prescriber, age 65+ share, LIS share, growth) | CMS Part D Prescribers by Geography and Drug | t0−1 (growth t0−2→t0−1) | End of t0−1 in plan/PBM claims; CMS public release ≈ 16–18 months later (sensitivity uses t0−2) |
| Years since NDA approval, dosage form, combination, no. of originator products, ATC group | Drugs@FDA, RxClass, CMS | Fixed / t0−1 | Before entry |
| `log_n_anda_tm1` | Drugs@FDA final (AP) ANDA approvals, distinct sponsors | By 31 Dec t0−1 | Start of entry year |
| `log_n_anda_t0` | Same | By 31 Dec t0 | End of entry year |
| `log_n_mfr_t0` | CMS Part D Spending by Drug, manufacturer rows; manufacturers with ≥1% of non-originator claims | Calendar year t0 | After entry-year claims accrue (end of t0) |
Tentative approvals (TA) are not counted.

## Feature-timing specifications
- **S1** pre-entry features — prediction date: start of entry year.
- **S2** S1 + approved sponsors by t0−1 — start of entry year.
- **S3** S1 + approved sponsors by t0 — end of entry year.
- **S4** S1 + marketed manufacturers in t0 — end of entry year (realized competition).

## Models and tuning protocol
- Baselines: training mean; competition-only linear regression (S2–S4); OLS with all features.
- Elastic net: penalty and l1-ratio (0.1, 0.5, 0.9, 1) chosen by 5-fold CV **within each training set**.
- Random forest: 500 trees, min leaf 3, max_features 0.5 — fixed a priori, not tuned.
- XGBoost: 400 trees, depth 3, learning rate 0.03, subsample/colsample 0.8, min_child_weight 3 — fixed a priori, not tuned.
- Imputation (median + missingness indicators) and scaling are fitted inside each training fold (scikit-learn pipelines).
- No feature selection beyond the pre-specified sets; no model selection on the temporal test set (all models reported).
- Validation: 10-fold CV × 5 repeats (seed 42); temporal split train 2014–2019 / test 2020–2023.
- Uncertainty: 2,000 bootstrap resamples of the temporal test set (95% percentile CIs), paired bootstrap for differences vs the reference baseline; leave-one-out and top-5-residual influence checks.
