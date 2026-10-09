"""A&E 4-hour performance: is the problem the front door (demand) or the back door (flow)?

Run from the project root:  python src/analysis.py
Writes tables to outputs/ and charts to outputs/figures/.

Unit of analysis: NHS trusts with a major (Type 1) emergency department.
Performance measure for trusts: % of Type 1 attendances admitted, transferred or
discharged within 4 hours. The national headline uses all A&E types, as in the
official statistic.
"""

from __future__ import annotations

import os
import sys

import matplotlib
import matplotlib.dates
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

sys.path.insert(0, os.path.dirname(__file__))
from load import england_monthly, load_all  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "outputs")
FIG = os.path.join(OUT, "figures")

TARGET_2026_27 = 82.0          # NHS England 4-hour target for 2026/27 (% of attendances)
MIN_MONTHLY_T1 = 3000          # a "major" ED: at least this many Type 1 attendances a month on average
WINDOW_MONTHS = 12             # cross-section uses the latest 12 months

# Chart styling (reference palette, light mode)
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, RED, SURFACE = "#2a78d6", "#eb6834", "#e34948", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
})


def trust_month_table(data: pd.DataFrame) -> pd.DataFrame:
    """One row per trust-month for trusts with a major ED, with the derived rates."""
    major = (data.groupby("org_code")["att_t1"].mean() >= MIN_MONTHLY_T1)
    # Children's hospitals treat a different case mix (few frail, elderly admissions),
    # so comparing them with general acute trusts would be unfair to both.
    childrens = data["org_name"].str.upper().str.contains("CHILDREN")
    df = data[data["org_code"].isin(major[major].index) & (data["att_t1"] > 0) & ~childrens].copy()
    df["pct_4h_t1"] = 100 * (1 - df["over4_t1"] / df["att_t1"])
    df["admit_rate"] = 100 * df["adm_t1"] / df["att_t1"]                 # % of Type 1 patients admitted
    df["dta12_per_1000"] = 1000 * df["dta_12h_plus"] / df["att_t1"]       # 12h+ waits for a bed per 1,000 attendances
    return df


def trust_window_table(tm: pd.DataFrame) -> pd.DataFrame:
    """Latest-12-month totals per trust, with rates computed from the summed counts."""
    last = sorted(tm["month"].unique())[-WINDOW_MONTHS:]
    w = tm[tm["month"].isin(last)]
    g = w.groupby(["org_code", "org_name", "region"], as_index=False)[
        ["att_t1", "over4_t1", "adm_t1", "dta_12h_plus"]].sum()
    g = g[w.groupby("org_code")["month"].nunique().reindex(g["org_code"]).values == WINDOW_MONTHS]
    g["monthly_att_t1"] = g["att_t1"] / WINDOW_MONTHS
    g["pct_4h_t1"] = 100 * (1 - g["over4_t1"] / g["att_t1"])
    g["admit_rate"] = 100 * g["adm_t1"] / g["att_t1"]
    g["dta12_per_1000"] = 1000 * g["dta_12h_plus"] / g["att_t1"]
    g["log_volume"] = np.log(g["monthly_att_t1"])
    g["region"] = g["region"].str.replace("NHS ENGLAND ", "").str.title()
    g["org_name"] = g["org_name"].str.title().str.replace("Nhs", "NHS").str.replace("'S ", "'s ")
    return g.reset_index(drop=True), (pd.Timestamp(last[0]), pd.Timestamp(last[-1]))


def cross_section_model(tw: pd.DataFrame):
    """Between-trust model: what explains why some trusts perform better than others?"""
    formula = "pct_4h_t1 ~ dta12_per_1000 + admit_rate + log_volume"
    fit = smf.ols(formula, data=tw).fit(cov_type="HC3")
    X, y = tw[["dta12_per_1000", "admit_rate", "log_volume"]].values, tw["pct_4h_t1"].values
    cv = cross_val_score(LinearRegression(), X, y, cv=KFold(5, shuffle=True, random_state=1), scoring="r2")
    # Which single factor explains most on its own?
    single = {v: smf.ols(f"pct_4h_t1 ~ {v}", data=tw).fit().rsquared
              for v in ["dta12_per_1000", "admit_rate", "log_volume"]}
    tw = tw.assign(expected=fit.fittedvalues, vs_expected=fit.resid)
    return fit, cv, single, tw


def within_trust_model(tm: pd.DataFrame):
    """Within-trust model: in a trust's busier months, what moves its performance?

    Trust fixed effects remove fixed differences between trusts (size, case mix,
    estate); month fixed effects remove national shocks such as winter. What is
    left compares each trust with itself.
    """
    tm = tm.copy()
    tm["att_index"] = 100 * tm["att_t1"] / tm.groupby("org_code")["att_t1"].transform("mean")
    fit = smf.ols("pct_4h_t1 ~ att_index + dta12_per_1000 + C(org_code) + C(month)",
                  data=tm).fit(cov_type="cluster", cov_kwds={"groups": tm["org_code"]})
    sd = {v: tm[v].sub(tm.groupby("org_code")[v].transform("mean")).std()
          for v in ["att_index", "dta12_per_1000"]}
    return fit, sd


def chart_trend(eng: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(eng.index, eng["pct_4h_all"], color=BLUE, lw=2, marker="o", ms=4, label="All A&E types (official measure)")
    ax.plot(eng.index, eng["pct_4h_t1"], color=ORANGE, lw=2, marker="o", ms=4, label="Major EDs (Type 1) only")
    ax.axhline(TARGET_2026_27, color=INK2, lw=1, ls="--")
    ax.text(eng.index[0], TARGET_2026_27 + 0.6, "2026/27 target: 82%", color=INK2, fontsize=9)
    for col, c in (("pct_4h_all", BLUE), ("pct_4h_t1", ORANGE)):
        ax.annotate(f"{eng[col].iloc[-1]:.1f}%", (eng.index[-1], eng[col].iloc[-1]), xytext=(6, -3),
                    textcoords="offset points", color=INK, fontsize=9)
    ax.set_ylim(50, 90)
    ax.set_ylabel("% seen within 4 hours")
    ax.set_title("England is below the 82% target, and major EDs are far below it")
    ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK2)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %y"))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "1_england_trend.png"), dpi=200)
    plt.close(fig)


def chart_flow(tw: pd.DataFrame, fit):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    sizes = 12 + 60 * (tw["monthly_att_t1"] / tw["monthly_att_t1"].max())
    ax.scatter(tw["dta12_per_1000"], tw["pct_4h_t1"], s=sizes, color=BLUE, alpha=0.7,
               edgecolor=SURFACE, linewidth=1)
    simple = smf.ols("pct_4h_t1 ~ dta12_per_1000", data=tw).fit()
    xs = np.linspace(0, tw["dta12_per_1000"].max(), 50)
    ax.plot(xs, simple.params["Intercept"] + simple.params["dta12_per_1000"] * xs, color=INK2, lw=1.5)
    ax.set_xlabel("Patients waiting 12+ hours for a bed after the decision to admit, per 1,000 attendances")
    ax.set_ylabel("% seen within 4 hours (Type 1)")
    ax.set_title("Where patients wait longest for a bed, fewer are seen within 4 hours", pad=20)
    ax.text(0, 1.01, f"{len(tw)} trusts, latest 12 months. Dot size = A&E attendances. Line: R² = {simple.rsquared:.2f}",
            transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "2_flow_vs_performance.png"), dpi=200)
    plt.close(fig)


def chart_expected(tw: pd.DataFrame, n: int = 8):
    top = tw.nlargest(n, "vs_expected")
    bot = tw.nsmallest(n, "vs_expected").iloc[::-1]
    sel = pd.concat([bot, top])
    fig, ax = plt.subplots(figsize=(8, 5.0))
    colors = [BLUE if v > 0 else RED for v in sel["vs_expected"]]
    ax.barh(range(len(sel)), sel["vs_expected"], color=colors, height=0.7)
    ax.set_yticks(range(len(sel)))
    ax.set_yticklabels([n.replace(" NHS Foundation Trust", "").replace(" NHS Trust", "") for n in sel["org_name"]],
                       fontsize=8.5, color=INK2)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_xlabel("Points above (+) or below (−) expected 4-hour performance")
    fig.suptitle("Trusts doing better and worse than their bed waits and admissions predict",
                 x=0.02, ha="left", fontsize=12, fontweight="bold", color=INK)
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(sel["vs_expected"]):
        ax.text(v + (0.3 if v > 0 else -0.3), i, f"{v:+.1f}", va="center",
                ha="left" if v > 0 else "right", fontsize=8, color=INK)
    lim = max(abs(sel["vs_expected"])) + 3
    ax.set_xlim(-lim, lim)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(FIG, "3_vs_expected.png"), dpi=200)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    data, _ = load_all()
    eng = england_monthly(data)
    tm = trust_month_table(data)
    tw, (start, end) = trust_window_table(tm)
    cs_fit, cv, single, tw = cross_section_model(tw)
    wt_fit, sd = within_trust_model(tm)

    eng.round(2).to_csv(os.path.join(OUT, "england_monthly.csv"))
    tw.sort_values("vs_expected").round(2).to_csv(os.path.join(OUT, "trusts_latest_12_months.csv"), index=False)
    chart_trend(eng)
    chart_flow(tw, cs_fit)
    chart_expected(tw)

    last = eng.iloc[-1]
    lines = [
        f"Data: {eng.index[0]:%b %Y} to {eng.index[-1]:%b %Y} ({len(eng)} months). "
        f"Trust analysis: {len(tw)} trusts with a major ED, {start:%b %Y} to {end:%b %Y}.",
        "",
        f"England, {eng.index[-1]:%B %Y}: {last['pct_4h_all']:.1f}% within 4 hours (all types), "
        f"{last['pct_4h_t1']:.1f}% at major EDs; target {TARGET_2026_27:.0f}%.",
        f"Gap to target: {TARGET_2026_27 - last['pct_4h_all']:.1f} points = about "
        f"{(TARGET_2026_27 - last['pct_4h_all']) / 100 * last['attendances']:,.0f} more patients a month "
        "would need to be seen within 4 hours.",
        f"12h+ waits for a bed, England, {eng.index[-1]:%B %Y}: {last['dta_12h_plus']:,.0f}.",
        f"Trusts meeting 82% at Type 1 (12 months): {(tw['pct_4h_t1'] >= TARGET_2026_27).sum()} of {len(tw)}.",
        f"Range across trusts: {tw['pct_4h_t1'].min():.1f}% to {tw['pct_4h_t1'].max():.1f}%.",
        "",
        "BETWEEN TRUSTS (latest 12 months, OLS with robust SEs)",
        cs_fit.summary().tables[1].as_text(),
        f"R² = {cs_fit.rsquared:.2f}; 5-fold cross-validated R² = {cv.mean():.2f} (range {cv.min():.2f} to {cv.max():.2f})",
        "R² of each factor on its own: " + ", ".join(f"{k} {v:.2f}" for k, v in single.items()),
        "",
        "WITHIN TRUSTS (trust and month fixed effects, SEs clustered by trust)",
        f"+10% attendances above the trust's own average: {10 * wt_fit.params['att_index']:+.2f} points "
        f"(95% CI {10 * wt_fit.conf_int().loc['att_index', 0]:+.2f} to {10 * wt_fit.conf_int().loc['att_index', 1]:+.2f})",
        f"+1 SD of 12h bed waits ({sd['dta12_per_1000']:.1f} per 1,000): "
        f"{sd['dta12_per_1000'] * wt_fit.params['dta12_per_1000']:+.2f} points "
        f"(95% CI {sd['dta12_per_1000'] * wt_fit.conf_int().loc['dta12_per_1000', 0]:+.2f} to "
        f"{sd['dta12_per_1000'] * wt_fit.conf_int().loc['dta12_per_1000', 1]:+.2f})",
        f"+1 SD of attendances ({sd['att_index']:.1f}% of own average): "
        f"{sd['att_index'] * wt_fit.params['att_index']:+.2f} points",
        "",
        "BETTER THAN EXPECTED (top 5)",
        tw.nlargest(5, "vs_expected")[["org_name", "region", "pct_4h_t1", "expected", "vs_expected"]].round(1).to_string(index=False),
        "",
        "WORSE THAN EXPECTED (bottom 5)",
        tw.nsmallest(5, "vs_expected")[["org_name", "region", "pct_4h_t1", "expected", "vs_expected"]].round(1).to_string(index=False),
    ]
    text = "\n".join(lines)
    with open(os.path.join(OUT, "results.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
