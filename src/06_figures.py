"""Figures and LaTeX tables for the (revised) manuscript.

Usage: python3 src/06_figures.py <results dir>
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

res = Path(sys.argv[1])
fig_dir, tab = res / "figures", res / "tables"
fig_dir.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight", "lines.linewidth": 2,
})

d = pd.read_csv(tab / "analytic_sample.csv")
OUTS = {"brand_share_t1": "Brand claims share", "log_gen_price_ratio_t1": "Generic price ratio (log)",
        "log_mol_spend_ratio_t1": "Molecule spending ratio (log)"}
SPECS = ["S1", "S2", "S3", "S4"]
SPEC_SHORT = {"S1": "S1 pre-entry", "S2": "S2 + approved by t0−1", "S3": "S3 + approved by t0", "S4": "S4 + marketed in t0"}


def save(fig, name):
    fig.savefig(fig_dir / f"{name}.pdf")
    fig.savefig(fig_dir / f"{name}.png")
    plt.close(fig)


# ---------- Figure 1: trajectories by competition tier ----------
d["tier"] = pd.cut(d.n_anda_sponsors_t0, [-1, 3, 9, 1000], labels=["1–3 ANDA sponsors", "4–9 ANDA sponsors", "≥10 ANDA sponsors"])
d.loc[d.n_anda_sponsors_t0 == 0, "tier"] = "1–3 ANDA sponsors"
ks = [-1, 0, 1, 2]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
for (tier, g), color, mk in zip(d.groupby("tier", observed=True), [ORANGE, BLUE, AQUA], ["o", "s", "^"]):
    bs = [np.ones(len(g))] + [g[f"brand_share_t{k}"].dropna().values for k in (0, 1, 2)]
    pr = [np.ones(len(g))] + [g[f"gen_price_ratio_t{k}"].dropna().values for k in (0, 1, 2)]
    for ax, series in zip(axes, [bs, pr]):
        med = [np.median(s) for s in series]
        ax.fill_between(ks, [np.percentile(s, 25) for s in series], [np.percentile(s, 75) for s in series],
                        color=color, alpha=0.12, linewidth=0)
        ax.plot(ks, med, color=color, marker=mk, markersize=5, markeredgecolor="white", markeredgewidth=1,
                label=f"{tier} (n={len(g)})")
        if ax is axes[1]:
            ax.annotate(f"{med[-1]:.2f}", (ks[-1], med[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", color=INK2, fontsize=7.5)
for ax, title in zip(axes, ["A. Originator brand share of claims", "B. Generic price relative to pre-entry brand"]):
    ax.set_xticks(ks, ["t0−1", "t0\n(entry)", "t0+1", "t0+2"])
    ax.set_ylim(0, 1.25 if "price" in title else 1.05)
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    ax.set_xlim(-1.2, 2.5)
axes[0].set_ylabel("Median (IQR band)")
axes[1].axhline(1, color=INK2, lw=0.8, ls=":")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.08))
save(fig, "fig1_trajectories")

# ---------- Figure 2: temporal R2 with bootstrap CIs ----------
perf = pd.read_csv(tab / "model_performance.csv")
SHOW = [("Competition-only linear", ORANGE, "o"), ("OLS (all features)", AQUA, "^"), ("Random Forest", BLUE, "s")]
LO = -0.6
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=True)
for ax, (o, lab) in zip(axes, OUTS.items()):
    for j, (m, color, mk) in enumerate(SHOW):
        sub = perf[(perf.outcome == o) & (perf.model == m) & (perf.validation == "Temporal")].set_index("spec")
        cv = perf[(perf.outcome == o) & (perf.model == m) & (perf.validation == "CV")].set_index("spec")
        for i, s in enumerate(SPECS):
            if s not in sub.index:
                continue
            y = i + (j - 1) * 0.24
            r = sub.loc[s]
            x = max(r.R2, LO)
            ax.errorbar(x, y, xerr=[[x - max(r.R2_lo, LO)], [max(r.R2_hi, LO) - x]], fmt=mk, color=color, ms=5,
                        elinewidth=1.2, capsize=0, mec="white", mew=0.8,
                        label=m if (ax is axes[0] and i == 1) else None)
            ax.scatter(max(cv.loc[s, "R2"], LO), y, marker="|", s=45, color=INK, zorder=4,
                       label="Cross-validated mean" if (ax is axes[0] and i == 1 and j == 0) else None)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlim(LO, 0.85)
    ax.set_title(lab, loc="left", fontsize=9)
    ax.set_xlabel("Temporal test R² (95% CI)")
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(range(4), [SPEC_SHORT[s] for s in SPECS])
axes[0].invert_yaxis()
fig.legend(loc="lower center", ncol=4, fontsize=7.2, bbox_to_anchor=(0.5, -0.12))
save(fig, "fig2_performance")

# ---------- Figure 3: SHAP importance (S3, top 8 per outcome) ----------
imp = json.load(open(tab / "shap_importance.json"))["S3"]
fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0))
for ax, (o, lab) in zip(axes, OUTS.items()):
    s = pd.Series(imp[o]).sort_values(ascending=True).tail(8)
    ax.barh(range(len(s)), s.values, color=BLUE, height=0.6)
    ax.set_yticks(range(len(s)), [t.replace(" (log1p)", "").replace(" (log)", "") for t in s.index], fontsize=7)
    ax.set_title(lab, loc="left", fontsize=8.5)
    ax.set_xlabel("Mean |SHAP|")
    ax.grid(axis="y", visible=False)
fig.tight_layout(w_pad=1.0)
save(fig, "fig3_shap")

# ---------- Figure 4: adjusted dose-response curves ----------
dr = pd.read_csv(tab / "dose_response.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
for ax, (o, lab) in zip(axes, [("log_gen_price_ratio_t1", "A. Generic price ratio, t0+1"),
                               ("log_mol_spend_ratio_t1", "B. Molecule spending ratio, t0+1 vs t0−1")]):
    for meas, color, mk, name in [("n_anda_sponsors_t0", BLUE, "o", "Approved ANDA sponsors by end of t0"),
                                  ("n_mfr_1pct_t0", ORANGE, "s", "Marketed generic manufacturers in t0")]:
        g = dr[(dr.outcome == o) & (dr.measure == meas)]
        ax.fill_between(g.n, np.exp(g.lo), np.exp(g.hi), color=color, alpha=0.15, lw=0)
        ax.plot(g.n, np.exp(g.pred), color=color, marker=mk, ms=3.5, mec="white", mew=0.6, label=name)
    ax.set_xlabel("Number of competitors (0 to 95th percentile)")
    ax.set_xticks(range(0, 15, 2))
    ax.set_title(lab, loc="left", fontsize=9)
    ax.set_ylim(0, None)
axes[0].set_ylabel("Covariate-adjusted prediction (95% CI)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, -0.14))
save(fig, "fig4_dose_response")


# ---------- LaTeX tables ----------
def q(s, fmt="{:.2f}"):
    s = s.dropna()
    return f"{fmt.format(s.median())} [{fmt.format(s.quantile(.25))}, {fmt.format(s.quantile(.75))}]"


def tex_escape(t):
    return t.replace("%", "\\%").replace("_", "\\_").replace("t0-1", "$t_0-1$").replace("by t0", "by $t_0$").replace("in t0", "in $t_0$")


rows = [
    ("Pre-entry brand spending, US\\$ million", q(d.pre_spend / 1e6, "{:.1f}")),
    ("Pre-entry brand claims, thousand", q(d.pre_clms / 1e3, "{:.1f}")),
    ("Pre-entry cost per 30-day fill, US\\$", q(d.pre_cost_per_fill, "{:.0f}")),
    ("30-day fills per beneficiary", q(d.pre_fills_per_bene, "{:.1f}")),
    ("Share of claims, age 65+", q(d.pre_ge65_share)),
    ("LIS share of patient cost", q(d.pre_lis_share)),
    ("Years since NDA approval", q(d.market_age, "{:.0f}")),
    ("Oral solid dosage form, n (\\%)", f"{d.oral_solid.sum()} ({100 * d.oral_solid.mean():.0f})"),
    ("Fixed-dose combination, n (\\%)", f"{d.combination.sum()} ({100 * d.combination.mean():.0f})"),
    ("\\midrule\\textit{Competition measures}", ""),
    ("Approved ANDA sponsors by end of $t_0-1$", q(d.n_anda_sponsors_tm1, "{:.0f}")),
    ("Approved ANDA sponsors by end of $t_0$", q(d.n_anda_sponsors_t0, "{:.0f}")),
    ("Marketed generic manufacturers in $t_0$", q(d.n_mfr_1pct_t0, "{:.0f}")),
    ("\\midrule\\textit{Outcomes}", ""),
    ("Brand claims share, entry year ($t_0$)", q(d.brand_share_t0)),
    ("Brand claims share, $t_0+1$", q(d.brand_share_t1)),
    ("Brand claims share, $t_0+2$", q(d.brand_share_t2)),
    ("Generic/brand price ratio, $t_0+1$", q(d.gen_price_ratio_t1)),
    ("Molecule spending ratio, $t_0+1$ vs $t_0-1$", q(d.mol_spend_ratio_t1)),
    ("\\quad 30-day fills ratio", q(d.mol_fills_ratio_t1)),
    ("\\quad Cost per 30-day fill ratio", q(d.mol_cost_per_fill_ratio_t1)),
]
atc = d.atc_group.value_counts()
with open(tab / "table1.tex", "w") as f:
    f.write("\\begin{tabular}{lr}\n\\toprule\nCharacteristic & Median [IQR] or n (\\%)\\\\\n\\midrule\n")
    for a, b in rows:
        f.write(f"{a} & {b}\\\\\n")
    f.write("\\midrule\\textit{ATC main group}, n (\\%) & \\\\\n")
    for k, v in atc.items():
        f.write(f"\\quad {k} & {v} ({100 * v / len(d):.0f})\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# Table 2: performance by spec
MODELS = ["Mean baseline", "Competition-only linear", "OLS (all features)", "Elastic Net", "Random Forest", "XGBoost"]
with open(tab / "table2.tex", "w") as f:
    f.write("\\begin{tabular}{ll" + "cc" * 4 + "}\n\\toprule\n & & " +
            " & ".join(f"\\multicolumn{{2}}{{c}}{{{s}}}" for s in SPECS) + "\\\\\n" +
            "".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(4)) + "\n" +
            "Outcome & Model & " + " & ".join(["CV & Temporal [95\\% CI]"] * 4) + "\\\\\n\\midrule\n")
    for o, lab in OUTS.items():
        for i, m in enumerate(MODELS):
            cells = []
            for s in SPECS:
                cv = perf[(perf.outcome == o) & (perf.spec == s) & (perf.model == m) & (perf.validation == "CV")]
                tp = perf[(perf.outcome == o) & (perf.spec == s) & (perf.model == m) & (perf.validation == "Temporal")]
                if cv.empty:
                    cells += ["--", "--"]
                else:
                    cv, tp = cv.iloc[0], tp.iloc[0]
                    cells += [f"{cv.R2:.2f}", f"{tp.R2:.2f} [{tp.R2_lo:.2f}, {tp.R2_hi:.2f}]"]
            f.write(f"{lab if i == 0 else ''} & {m} & " + " & ".join(cells) + "\\\\\n")
        f.write("\\midrule\n" if o != list(OUTS)[-1] else "")
    f.write("\\bottomrule\n\\end{tabular}\n")

ols = pd.read_csv(tab / "ols_hc3.csv")


def star(p):
    return "^{***}" if p < .001 else "^{**}" if p < .01 else "^{*}" if p < .05 else ""


OLS_COLS = {"brand_share_t1": "Brand share", "log_gen_price_ratio_t1": "Log price ratio",
            "log_mol_spend_ratio_t1": "Log spending ratio", "log_mol_fills_ratio_t1": "Log fills ratio",
            "log_mol_cost_per_fill_ratio_t1": "Log cost/fill ratio"}
o3 = ols[ols.spec == "S3"]
terms = [t for t in o3.term.unique() if t not in ("const", "N", "Adj. R2")]
with open(tab / "table3.tex", "w") as f:
    f.write("\\begin{tabular}{l" + "c" * len(OLS_COLS) + "}\n\\toprule\n & & & & \\multicolumn{2}{c}{Decomposition of (3)}\\\\\n"
            "\\cmidrule(lr){5-6}\n & " + " & ".join(f"({i + 1}) {v}" for i, v in enumerate(OLS_COLS.values())) + "\\\\\n\\midrule\n")
    for t in terms:
        cells = []
        for o in OLS_COLS:
            r = o3[(o3.outcome == o) & (o3.term == t)].iloc[0]
            cells.append(f"${r.coef:.3f}{star(r.p)}$ ({r.se:.3f})")
        f.write(f"{tex_escape(t)} & " + " & ".join(cells) + "\\\\\n")
    f.write("\\midrule\n")
    for t in ["N", "Adj. R2"]:
        cells = [f"{o3[(o3.outcome == o) & (o3.term == t)].coef.iloc[0]:.{0 if t == 'N' else 2}f}" for o in OLS_COLS]
        tl = t.replace("Adj. R2", "Adjusted $R^2$")
        f.write(f"{tl} & " + " & ".join(cells) + "\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

COMP_TERM = {"S2": "Approved ANDA sponsors by t0-1 (log1p)", "S3": "Approved ANDA sponsors by t0 (log1p)",
             "S4": "Marketed generic manufacturers in t0 (log1p)"}
COMP_ROW = {"S2": "Approved sponsors by end of $t_0-1$ (S2)", "S3": "Approved sponsors by end of $t_0$ (S3)",
            "S4": "Marketed manufacturers in $t_0$ (S4)"}
with open(tab / "tableS2.tex", "w") as f:
    f.write("\\begin{tabular}{l" + "c" * len(OLS_COLS) + "}\n\\toprule\nCompetition measure, $\\log(1+n)$ & " +
            " & ".join(OLS_COLS.values()) + "\\\\\n\\midrule\n")
    for s, t in COMP_TERM.items():
        cells = []
        for o in OLS_COLS:
            r = ols[(ols.spec == s) & (ols.outcome == o) & (ols.term == t)].iloc[0]
            cells.append(f"${r.coef:.3f}{star(r.p)}$ ({r.se:.3f})")
        f.write(f"{COMP_ROW[s]} & " + " & ".join(cells) + "\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

cls = pd.read_csv(tab / "classification_performance.csv")


def cls_table(path, thresholds):
    with open(path, "w") as f:
        f.write("\\begin{tabular}{llccc}\n\\toprule\nThreshold (prevalence) & Model, specification & CV AUC (SD) & "
                "CV Brier (null) & Temporal AUC [95\\% CI]\\\\\n\\midrule\n")
        for thr in thresholds:
            sub = cls[cls.threshold == thr]
            first = True
            for m in sub.model.unique():
                for s in ("S1", "S3", "S4"):
                    r = sub[(sub.model == m) & (sub.spec == s)].iloc[0]
                    lab = f"$\\geq${int(round(thr * 100))}\\% ({r.prevalence:.2f})" if first else ""
                    first = False
                    f.write(f"{lab} & {m}, {s} & {r.AUC:.2f} ({r.AUC_sd:.2f}) & {r.Brier:.3f} ({r.Brier_null:.3f}) & "
                            f"{r.AUC_temporal:.2f} [{r.AUC_temporal_lo:.2f}, {r.AUC_temporal_hi:.2f}]\\\\\n")
            f.write("\\midrule\n" if thr != thresholds[-1] else "")
        f.write("\\bottomrule\n\\end{tabular}\n")


cls_table(tab / "table4.tex", [0.2, 0.3])
cls_table(tab / "tableS3.tex", [0.1, 0.4, 0.5])

with open(tab / "tableS4.tex", "w") as f:
    f.write("\\begin{tabular}{lllcccc}\n\\toprule\nOutcome & Spec. & Model & Temporal $R^2$ [95\\% CI] & "
            "$\\Delta R^2$ vs reference [95\\% CI] & LOO $R^2$ range & $R^2$ excl. top 5\\\\\n\\midrule\n")
    for o, lab in OUTS.items():
        first = True
        for s in SPECS:
            for m in ("OLS (all features)", "Random Forest", "XGBoost"):
                r = perf[(perf.outcome == o) & (perf.spec == s) & (perf.model == m) & (perf.validation == "Temporal")].iloc[0]
                ref = "comp." if r.ref == "Competition-only linear" else "mean"
                f.write(f"{lab if first else ''} & {s} & {m} & {r.R2:.2f} [{r.R2_lo:.2f}, {r.R2_hi:.2f}] & "
                        f"{r.dR2_vs_ref:+.2f} [{r.dR2_lo:+.2f}, {r.dR2_hi:+.2f}] ({ref}) & "
                        f"{r.LOO_R2_min:.2f} to {r.LOO_R2_max:.2f} & {r.R2_drop_top5:.2f}\\\\\n")
                first = False
        f.write("\\midrule\n" if o != list(OUTS)[-1] else "")
    f.write("\\bottomrule\n\\end{tabular}\n")
print("figures and tables written")
