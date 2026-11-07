# -*- coding: utf-8 -*-
"""第1段階: VIXシグナル投資を、年初一括・ドルコスト平均法・それらのアンサンブルと同じ期間で比較する。

A. 資金投入型: 毎年1単位が年初に届く。いつ株式に投じるかだけが違う（外部キャッシュフローは全戦略共通）
B. 売買型    : シグナル後N日だけ株式、それ以外はT-bill。常時保有と比較する
C. 検定      : ビンテージ別の対応比較、循環シフトによるプラセボ検定
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src import strategies as S
from src.config import COST_BPS, HORIZONS, N_BOOT, PLACEBO_MIN_SHIFT, PLACEBO_STEP, RESULTS_DIR, SEED, THRESHOLDS
from src.data import load
from src.metrics import block_ci_sharpe_diff, sharpe, sign_test

SUBPERIODS = {"1990-2007": ("1990-01-02", "2007-12-31"), "2008-2025": ("2008-01-01", "2025-12-31")}


RESERVE_SHARES = [0.1, 0.2, 1 / 3, 0.5]


def budget_rules(m: S.Market) -> dict[str, S.Tranches]:
    lump, dca = S.lump_sum(m), S.dca(m)
    rules = {"年初一括": lump, "DCA(毎月)": dca}
    for th in THRESHOLDS:
        rules[f"VIX>={th} 待機(年末強制)"] = S.signal_wait(m, m.vix >= th, "E")
    for th in THRESHOLDS:
        rules[f"VIX>={th} 待機(繰越)"] = S.signal_wait(m, m.vix >= th, "R")
    rules["一括+DCA"] = lump.scaled(0.5) + dca.scaled(0.5)
    for th in THRESHOLDS:
        sig = rules[f"VIX>={th} 待機(年末強制)"]
        rules[f"一括+VIX>={th}"] = lump.scaled(0.5) + sig.scaled(0.5)
        rules[f"DCA+VIX>={th}"] = dca.scaled(0.5) + sig.scaled(0.5)
        rules[f"一括+DCA+VIX>={th}"] = lump.scaled(1 / 3) + dca.scaled(1 / 3) + sig.scaled(1 / 3)
        carry = rules[f"VIX>={th} 待機(繰越)"]
        rules[f"一括+DCA+VIX>={th}(繰越)"] = lump.scaled(1 / 3) + dca.scaled(1 / 3) + carry.scaled(1 / 3)
    return rules


def reserve_mix(m: S.Market, signal: np.ndarray, policy: str, share: float) -> S.Tranches:
    """年間予算のうち share を待機資金にし、残りを年初一括と毎月DCAに半分ずつ割り当てる。"""
    rest = (1.0 - share) / 2.0
    return S.lump_sum(m).scaled(rest) + S.dca(m).scaled(rest) + S.signal_wait(m, signal, policy).scaled(share)


def reserve_sensitivity(m: S.Market) -> pd.DataFrame:
    """待機資金の比率を変えた感度分析。最良の比率を選ぶためではなく、結論が比率に依存するかを見る。"""
    rows = []
    for th in THRESHOLDS:
        for policy, pname in (("E", "年末強制"), ("R", "繰越")):
            for share in RESERVE_SHARES:
                row = S.budget_metrics(m, reserve_mix(m, m.vix >= th, policy, share), f"VIX>={th} {pname}")
                rows.append({"threshold": th, "policy": pname, "reserve_share": share, **row})
    return pd.DataFrame(rows)


def vintage_tests(m: S.Market, rules: dict[str, S.Tranches]) -> pd.DataFrame:
    """年ごとの買付タイミング差（年末時点の対数比）。年は重ならないので独立に近い標本。"""
    base = {k: np.log(S.vintage_values(m, rules[k])) for k in ("年初一括", "DCA(毎月)")}
    years = m.years[m.year_first]
    excl = ~np.isin(years, [2008, 2020])
    rng = np.random.default_rng(SEED)
    rows = []
    for name, tr in rules.items():
        if "繰越" in name or name in ("年初一括",):
            continue
        v = np.log(S.vintage_values(m, tr))
        for bname, b in base.items():
            if name == bname:
                continue
            d = v - b
            boots = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
            sign = sign_test(d)
            rows.append({
                "strategy": name, "versus": bname, "n_years": len(d),
                "mean_diff_pct": d.mean() * 100, "ci_lo_pct": np.quantile(boots, .025) * 100,
                "ci_hi_pct": np.quantile(boots, .975) * 100,
                "t_p": stats.ttest_1samp(d, 0).pvalue,
                "win_years": sign["wins"], "loss_years": sign["losses"], "tie_years": sign["ties"],
                "sign_p": sign["p"],
                "mean_diff_excl_2008_2020_pct": d[excl].mean() * 100,
                "worst_year_pct": d.min() * 100, "best_year_pct": d.max() * 100,
            })
    return pd.DataFrame(rows)


def signal_timing_table(m: S.Market) -> pd.DataFrame:
    """シグナルが何年に発火したか。発火しない年は年末強制投資になる。"""
    rows = []
    for th in THRESHOLDS:
        tr = S.signal_wait(m, m.vix >= th, "E")
        fired = tr.deploy < m.year_last
        rows.append({"threshold": th, "years_fired": int(fired.sum()), "n_years": len(fired),
                     "days_signal": int((m.vix >= th).sum()),
                     "avg_wait_days": float((tr.deploy - tr.arrive).mean())})
    return pd.DataFrame(rows)


def placebo_shifts(n: int) -> np.ndarray:
    return np.arange(PLACEBO_MIN_SHIFT, n - PLACEBO_MIN_SHIFT, PLACEBO_STEP)


def budget_placebo(m: S.Market, share: float = 1.0) -> pd.DataFrame:
    """VIX系列を循環シフトし、発火の塊の形を保ったまま市場との対応だけを壊す。

    share=1 は全額待機のルール。share<1 は残りを年初一括とDCAに割り当てたアンサンブルで、
    族全体の補正は同じ share の6ルールの中で行う。
    """
    shifts = placebo_shifts(m.n)
    specs = [(th, p) for th in THRESHOLDS for p in ("E", "R")]
    real_w, real_s = [], []
    pw, ps = np.empty((len(shifts), len(specs))), np.empty((len(shifts), len(specs)))
    for j, (th, p) in enumerate(specs):
        sim = S.simulate(m, reserve_mix(m, m.vix >= th, p, share))
        real_w.append(sim["wealth"][-1])
        real_s.append(sharpe(sim["twr"][1:], m.rf[1:]))
        base_sig = m.vix >= th
        for i, s in enumerate(shifts):
            sim = S.simulate(m, reserve_mix(m, np.roll(base_sig, s), p, share))
            pw[i, j] = sim["wealth"][-1]
            ps[i, j] = sharpe(sim["twr"][1:], m.rf[1:])
    real_w, real_s = np.array(real_w), np.array(real_s)
    zw = (pw - pw.mean(0)) / pw.std(0)
    zs = (ps - ps.mean(0)) / ps.std(0)
    zw_real = (real_w - pw.mean(0)) / pw.std(0)
    zs_real = (real_s - ps.mean(0)) / ps.std(0)
    rows = []
    for j, (th, p) in enumerate(specs):
        rows.append({
            "rule": f"VIX>={th} {'年末強制' if p == 'E' else '繰越'}",
            "terminal": real_w[j], "placebo_median_terminal": np.median(pw[:, j]),
            "p_terminal": (1 + (pw[:, j] >= real_w[j]).sum()) / (1 + len(shifts)),
            "p_terminal_family": (1 + (zw.max(1) >= zw_real[j]).sum()) / (1 + len(shifts)),
            "sharpe": real_s[j], "placebo_median_sharpe": np.median(ps[:, j]),
            "p_sharpe": (1 + (ps[:, j] >= real_s[j]).sum()) / (1 + len(shifts)),
            "p_sharpe_family": (1 + (zs.max(1) >= zs_real[j]).sum()) / (1 + len(shifts)),
            "n_shifts": len(shifts),
        })
    return pd.DataFrame(rows)


def alpha_beta(rs: np.ndarray, rm: np.ndarray, rf: np.ndarray) -> tuple[float, float]:
    y, x = rs - rf, rm - rf
    beta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    return float((y.mean() - beta * x.mean()) * 252), float(beta)


def tactical_section(m: S.Market) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [S.tactical_metrics(m, m.rm, "常時保有(S&P500 TR)", np.ones(m.n, dtype=bool)),
            S.tactical_metrics(m, m.rf, "T-billのみ", np.zeros(m.n, dtype=bool))]
    shifts = placebo_shifts(m.n)
    specs = [(th, h) for th in THRESHOLDS for h in HORIZONS]
    real_a, real_s = np.empty(len(specs)), np.empty(len(specs))
    pa, ps = np.empty((len(shifts), len(specs))), np.empty((len(shifts), len(specs)))
    rm1, rf1 = m.rm[1:], m.rf[1:]
    for j, (th, h) in enumerate(specs):
        sig = m.vix >= th
        pos = S.tactical_position(sig, h)
        r = S.tactical_returns(m, pos)
        rn = S.tactical_returns(m, pos, cost_bps=COST_BPS)
        row = S.tactical_metrics(m, r, f"VIX>={th} 後{h}日保有", pos)
        row["cagr_net_cost"] = S.tactical_metrics(m, rn, "")["cagr"]
        # 平均エクスポージャーが同じ固定配分（毎日リバランス）: タイミングの価値だけを切り出す比較対象
        w = pos.mean()
        rmix = w * m.rm + (1 - w) * m.rf
        row["cagr_same_exposure_mix"] = S.tactical_metrics(m, rmix, "")["cagr"]
        row["sharpe_same_exposure_mix"] = sharpe(rmix[1:], rf1)
        a, b = alpha_beta(r[1:], rm1, rf1)
        fit = sm.OLS(r[1:] - rf1, sm.add_constant(rm1 - rf1)).fit(cov_type="HAC", cov_kwds={"maxlags": 21})
        row.update(alpha_annual=a, beta=b, alpha_hac_p=float(fit.pvalues[0]))
        ci = block_ci_sharpe_diff(r[1:], rm1, rf1, block=63, n_boot=N_BOOT, seed=SEED + j)
        row.update(sharpe_diff_vs_bh=ci["point"], sharpe_diff_lo=ci["lo"], sharpe_diff_hi=ci["hi"])
        rows.append(row)
        real_a[j], real_s[j] = a, sharpe(r[1:], rf1)
        for i, s in enumerate(shifts):
            rp = S.tactical_returns(m, S.tactical_position(np.roll(sig, s), h))
            pa[i, j], _ = alpha_beta(rp[1:], rm1, rf1)
            ps[i, j] = sharpe(rp[1:], rf1)
    za = (pa - pa.mean(0)) / pa.std(0)
    zreal = (real_a - pa.mean(0)) / pa.std(0)
    pl = []
    for j, (th, h) in enumerate(specs):
        pl.append({"rule": f"VIX>={th} 後{h}日保有", "alpha_annual": real_a[j],
                   "placebo_median_alpha": np.median(pa[:, j]),
                   "p_alpha": (1 + (pa[:, j] >= real_a[j]).sum()) / (1 + len(shifts)),
                   "p_alpha_family": (1 + (za.max(1) >= zreal[j]).sum()) / (1 + len(shifts)),
                   "sharpe": real_s[j], "placebo_median_sharpe": np.median(ps[:, j]),
                   "p_sharpe": (1 + (ps[:, j] >= real_s[j]).sum()) / (1 + len(shifts))})
    return pd.DataFrame(rows), pd.DataFrame(pl)


def subperiods(df: pd.DataFrame) -> pd.DataFrame:
    """全戦略の評価期間を揃えたまま、前後半に分けて同じ計算をする。"""
    rows = []
    for name, (a, b) in SUBPERIODS.items():
        m = S.Market(df.loc[a:b])
        rules = budget_rules(m)
        for k in ["年初一括", "DCA(毎月)", "VIX>=20 待機(年末強制)", "VIX>=30 待機(年末強制)", "VIX>=40 待機(年末強制)",
                  "一括+DCA", "一括+DCA+VIX>=30", "一括+DCA+VIX>=40"]:
            row = S.budget_metrics(m, rules[k], k)
            row["period"] = name
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    df = load()
    m = S.Market(df)
    RESULTS_DIR.mkdir(exist_ok=True)
    rules = budget_rules(m)
    table = pd.DataFrame([S.budget_metrics(m, tr, k) for k, tr in rules.items()])
    table.to_csv(RESULTS_DIR / "s1_budget_metrics.csv", index=False)
    print(table.round(4).to_string())

    vint = vintage_tests(m, rules)
    vint.to_csv(RESULTS_DIR / "s1_vintage_tests.csv", index=False)
    print("\n", vint.round(3).to_string())

    timing = signal_timing_table(m)
    timing.to_csv(RESULTS_DIR / "s1_signal_timing.csv", index=False)
    print("\n", timing.round(2).to_string())

    # 図のための評価額系列とビンテージ別の差
    paths = pd.DataFrame({k: S.simulate(m, rules[k])["wealth"]
                          for k in ["年初一括", "DCA(毎月)", "VIX>=30 待機(年末強制)", "VIX>=40 待機(繰越)", "一括+DCA+VIX>=30"]},
                         index=m.index)
    paths.to_csv(RESULTS_DIR / "s1_wealth_paths.csv")
    yr = m.years[m.year_first]
    vv = pd.DataFrame({k: np.log(S.vintage_values(m, rules[k])) for k in
                       ["年初一括", "DCA(毎月)", "VIX>=20 待機(年末強制)", "VIX>=30 待機(年末強制)", "VIX>=40 待機(年末強制)"]}, index=yr)
    vv.to_csv(RESULTS_DIR / "s1_vintage_log_values.csv")

    print("\nプラセボ（資金投入型）...")
    pb = budget_placebo(m)
    pb.to_csv(RESULTS_DIR / "s1_budget_placebo.csv", index=False)
    print(pb.round(4).to_string())
    print("\nプラセボ（一括+DCA+待機資金、各1/3）...")
    pe = budget_placebo(m, share=1 / 3)
    pe.to_csv(RESULTS_DIR / "s1_ensemble_placebo.csv", index=False)
    print(pe.round(4).to_string())

    sens = reserve_sensitivity(m)
    sens.to_csv(RESULTS_DIR / "s1_reserve_sensitivity.csv", index=False)
    print("\n", sens[["threshold", "policy", "reserve_share", "terminal_multiple", "irr", "twr_cagr", "vol", "sharpe",
                      "mdd", "avg_equity_weight"]].round(4).to_string())

    print("\n売買型...")
    tac, tac_pl = tactical_section(m)
    tac.to_csv(RESULTS_DIR / "s1_tactical_metrics.csv", index=False)
    tac_pl.to_csv(RESULTS_DIR / "s1_tactical_placebo.csv", index=False)
    print(tac.round(4).to_string())
    print(tac_pl.round(4).to_string())

    sub = subperiods(df)
    sub.to_csv(RESULTS_DIR / "s1_subperiods.csv", index=False)
    print("\n", sub[["period", "strategy", "irr", "twr_cagr", "vol", "sharpe", "mdd", "avg_equity_weight"]]
          .round(4).to_string())

    meta = {"start": str(m.index[0].date()), "end": str(m.index[-1].date()), "n_days": m.n,
            "n_years": len(m.year_first), "span_years": m.span_years}
    (RESULTS_DIR / "s1_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
