# -*- coding: utf-8 -*-
"""図の生成。すべて results/ の出力と data/ のスナップショットから作る。"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import FIGURES_DIR, RESULTS_DIR, VIX_LABELS  # noqa: E402
from src.data import forward_excess, load  # noqa: E402
from src.metrics import sign_test  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
LABELS_SHORT = ["<20", "20-30", "30-40", ">=40"]
HZ = ["1M", "3M", "6M", "12M"]

plt.rcParams.update({
    "font.family": ["Yu Gothic", "Meiryo", "MS Gothic", "DejaVu Sans"], "font.size": 10, "axes.unicode_minus": False,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11, "legend.frameon": False, "legend.fontsize": 9,
    "lines.linewidth": 2.0,
})


def save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_risk_vs_return(df: pd.DataFrame) -> None:
    """研究全体の核心: VIXはリターンの水準ではなく、散らばりを説明する。"""
    y = forward_excess(df, 21) * 100
    r = np.log1p(df["rm"])
    rv = np.sqrt((r ** 2).rolling(21).sum().shift(-22) * 252 / 21) * 100
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    ok = y.notna()
    axes[0].scatter(df["VIX"][ok], y[ok], s=4, alpha=0.25, color=BLUE, linewidths=0)
    axes[0].axhline(0, color=MUTED, lw=1)
    axes[0].set_ylabel("その後1ヶ月の超過リターン（%, 対数）")
    axes[0].set_title("リターン: VIXが高いほど上下に散らばる")
    ok = rv.notna()
    axes[1].scatter(df["VIX"][ok], rv[ok], s=4, alpha=0.25, color=BLUE, linewidths=0)
    lim = [9, 85]
    axes[1].plot(lim, lim, color=MUTED, lw=1)
    axes[1].annotate("実線: 実現ボラ = VIX", (9.5, 55), color=MUTED, fontsize=9)
    axes[1].set_ylabel("その後1ヶ月の実現ボラティリティ（年率%）")
    axes[1].set_title("リスク: VIXとともに上がり、実現ボラはVIXを下回りやすい")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks([10, 15, 20, 30, 40, 60, 80])
        ax.set_xticklabels(["10", "15", "20", "30", "40", "60", "80"])
        ax.set_xlabel("当日のVIX（対数目盛）")
    axes[1].set_yscale("log")
    axes[1].set_yticks([5, 10, 20, 40, 80])
    axes[1].set_yticklabels(["5", "10", "20", "40", "80"])
    fig.tight_layout()
    save(fig, "fig4_risk_vs_return.png")


def fig_budget() -> None:
    p = pd.read_csv(RESULTS_DIR / "s1_wealth_paths.csv", index_col=0, parse_dates=True)
    v = pd.read_csv(RESULTS_DIR / "s1_vintage_log_values.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    series = [("DCA(毎月)", "DCA（毎月）", BLUE), ("VIX>=30 待機(年末強制)", "VIX≥30 待機（年末強制）", ORANGE),
              ("VIX>=40 待機(繰越)", "VIX≥40 待機（繰越）", AQUA)]
    for col, lab, c in series:
        d = (p[col] / p["年初一括"] - 1) * 100
        axes[0].plot(d.index, d, color=c, label=lab)
    axes[0].axhline(0, color=MUTED, lw=1)
    axes[0].set_ylabel("年初一括に対する評価額の差（%）")
    axes[0].set_title("待つほど評価額は年初一括を下回りやすい")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, columnspacing=1.0, handlelength=1.5)
    d = (v["VIX>=30 待機(年末強制)"] - v["年初一括"]) * 100
    axes[1].bar(d.index, d, color=ORANGE, width=0.8)
    axes[1].axhline(0, color=MUTED, lw=1)
    axes[1].set_ylabel("年末時点の差（%, 対数）")
    sign = sign_test(d.to_numpy())
    axes[1].set_title(f"VIX≥30 待機 − 年初一括（{sign['wins']}勝 {sign['losses']}敗 {sign['ties']}分）")
    fig.tight_layout()
    save(fig, "fig1_budget.png")


def fig_phase() -> None:
    pt = pd.read_csv(RESULTS_DIR / "s2_phase.csv")
    ev = pd.read_csv(RESULTS_DIR / "s2_events_summary.csv")
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9))
    for ax, hz in zip(axes, HZ):
        s = pt[pt.horizon == hz].set_index("phase").loc[VIX_LABELS]
        x = np.arange(4)
        base = s["all_mean_pct"].iloc[0]
        yerr = [s["diff_vs_all_pct"] - s["diff_lo_pct"], s["diff_hi_pct"] - s["diff_vs_all_pct"]]
        ax.errorbar(x, s["mean_pct"], yerr=yerr,
                    fmt="o", ms=8, color=BLUE, capsize=3, label="該当する全日の平均（縦棒: 全期間平均との差の95%CI）")
        e = ev[ev.horizon == hz].set_index("threshold")
        ax.scatter([2, 3], [e.loc[30, "mean_pct"], e.loc[40, "mean_pct"]], s=70, marker="D", facecolors=SURFACE,
                   edgecolors=ORANGE, linewidths=2, zorder=5, label="初めて閾値に達した日の平均（≥30, ≥40）")
        ax.axhline(base, color=MUTED, lw=1, ls="--", label="全期間の平均")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS_SHORT)
        ax.set_title(f"{hz}後")
        ax.set_xlabel("当日のVIX")
    axes[0].set_ylabel("超過リターン（%, 対数）")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, "fig2_phase.png")


def fig_horizon() -> None:
    g = pd.read_csv(RESULTS_DIR / "s3_horizon_grid.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    axes[0].fill_between(g.h, g.lo, g.hi, color=BLUE, alpha=0.18, linewidth=0)
    axes[0].plot(g.h, g.slope, color=BLUE)
    axes[0].axhline(0, color=MUTED, lw=1)
    axes[0].set_title("連続VIX: log VIXの傾きは0と区別できない")
    axes[0].set_ylabel("傾き（95%CI, Newey-West）")
    axes[1].fill_between(g.h, g.panic_lo_pct, g.panic_hi_pct, color=BLUE, alpha=0.18, linewidth=0)
    axes[1].plot(g.h, g.panic_coef_pct, color=BLUE)
    axes[1].axhline(0, color=MUTED, lw=1)
    axes[1].set_title("離散VIX: VIX≥40の上乗せは期間が長いほど開く")
    axes[1].set_ylabel("VIX<20との差（%, 95%CI）")
    axes[2].plot(g.h, g.oos_r2_cont_pct, color=BLUE, label="連続（線形）")
    axes[2].plot(g.h, g.oos_r2_disc_pct, color=ORANGE, label="離散（4フェーズ）")
    axes[2].axhline(0, color=MUTED, lw=1)
    axes[2].set_title("アウトオブサンプルR²（0が過去平均と同等）")
    axes[2].set_ylabel("OOS R²（%）")
    axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    for ax in axes:
        ax.set_xlabel("投資期間（営業日）")
        ax.set_xticks([21, 63, 126, 189, 252])
    fig.tight_layout()
    save(fig, "fig3_horizon.png")


def fig_conformal() -> None:
    g = pd.read_csv(RESULTS_DIR / "s4_grid.csv")
    bp = pd.read_csv(RESULTS_DIR / "s4_by_phase.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    style = [("Norm", "散らばりに連続VIX（Norm）", BLUE, "o"), ("Mondrian", "散らばりに離散VIX（Mondrian）", ORANGE, "s"),
             ("RegD", "平均に離散VIX（RegD）", AQUA, "^"), ("Reg", "平均に連続VIX（Reg）", YELLOW, "D")]
    for m, lab, c, mk in style:
        s = g[g.method == m]
        axes[0].plot(s.h, s.winkler_skill_pct, color=c, marker=mk, ms=7, label=lab)
    axes[0].axhline(0, color=MUTED, lw=1)
    axes[0].set_xlabel("投資期間（営業日）")
    axes[0].set_xticks([21, 63, 126, 189, 252])
    axes[0].set_ylabel("VIXなしの基準に対する Winkler 改善率（%）")
    axes[0].set_title("散らばりに使うと全期間で改善、平均に使うと短期で悪化")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    s = bp[bp.horizon == "1M"]
    x = np.arange(4)
    width = 0.26
    for i, (m, lab, c) in enumerate([("Base", "VIXなし（Base）", MUTED), ("Mondrian", "離散VIX（Mondrian）", ORANGE),
                                     ("Norm", "連続VIX（Norm）", BLUE)]):
        cov = s[s.method == m].set_index("phase").loc[VIX_LABELS, "coverage"] * 100
        axes[1].bar(x + (i - 1) * width, cov, width - 0.02, color=c, label=lab)
    axes[1].axhline(90, color=INK, lw=1, ls="--")
    axes[1].annotate("破線: 目標90%", (-0.42, 101), color=MUTED, fontsize=9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(LABELS_SHORT)
    axes[1].set_xlabel("区間を出した日のVIX")
    axes[1].set_ylabel("実現カバレッジ（%）")
    axes[1].set_ylim(0, 108)
    axes[1].set_title("1ヶ月先: VIXなしの区間は高VIX局面で大きく外れる")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    fig.tight_layout()
    save(fig, "fig5_conformal.png")


def fig_bands() -> None:
    b = pd.read_csv(RESULTS_DIR / "s4_bands_h21.csv", index_col=0, parse_dates=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    windows = [("2008-06-01", "2009-06-30", "2008年6月〜2009年6月"), ("2019-12-01", "2020-12-31", "2019年12月〜2020年12月")]
    for ax, (a, c, title) in zip(axes, windows):
        s = b.loc[a:c]
        ax.fill_between(s.index, s["Norm_lo"] * 100, s["Norm_hi"] * 100, color=BLUE, alpha=0.2, linewidth=0,
                        label="VIXで幅を変える区間（Norm）")
        ax.plot(s.index, s["Base_lo"] * 100, color=ORANGE, lw=1.5, label="VIXなしの区間（Base）")
        ax.plot(s.index, s["Base_hi"] * 100, color=ORANGE, lw=1.5)
        ax.plot(s.index, s["y"] * 100, color=INK, lw=1.2, label="実現した1ヶ月先リターン")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("超過リターン（%, 対数）")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    save(fig, "fig6_bands.png")


def main() -> None:
    df = load()
    fig_risk_vs_return(df)
    fig_budget()
    fig_phase()
    fig_horizon()
    fig_conformal()
    fig_bands()
    print("図を生成しました:", sorted(p.name for p in FIGURES_DIR.glob("*.png")))


if __name__ == "__main__":
    main()
