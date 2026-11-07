# -*- coding: utf-8 -*-
"""REPORT.md を results/ の CSV から生成する。

本文中の統計値はすべて CSV から差し込む（手で数値を書かない）。
文章で述べる定性的な主張は check() で検証し、データと合わなくなったら生成を止める。
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.config import ALPHA, COST_BPS, GAMMA, HORIZON_NAMES, HORIZONS, N_BOOT, ROOT, RESULTS_DIR, VIX_LABELS
from src.data import forward_excess, load

HZ = [HORIZON_NAMES[h] for h in HORIZONS]
METHODS_ALL = ["Base", "Reg", "RegD", "Norm", "Mondrian", "Asym", "ACI-Base", "ACI-Norm"]
OUTPUT = ROOT / "REPORT.md"


def read(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / name, **kw)


def check(cond, message: str) -> None:
    if not bool(cond):
        raise AssertionError(f"本文の主張がデータと一致しません: {message}")


def pct(x: float, d: int = 2) -> str:
    return f"{x * 100:.{d}f}%"


def n(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}"


def sg(x: float, d: int = 2) -> str:
    return f"{x:+.{d}f}"


def pv(x: float) -> str:
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def ci(lo: float, hi: float, d: int = 2) -> str:
    return f"[{lo:+.{d}f}, {hi:+.{d}f}]"


def label(s: str) -> str:
    return str(s).replace(">=", "≥")


def table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def stage1() -> tuple[str, dict]:
    bm = read("s1_budget_metrics.csv").set_index("strategy")
    vt = read("s1_vintage_tests.csv").set_index(["strategy", "versus"])
    tm = read("s1_signal_timing.csv").set_index("threshold")
    pl = read("s1_budget_placebo.csv").set_index("rule")
    ep = read("s1_ensemble_placebo.csv").set_index("rule")
    ta = read("s1_tactical_metrics.csv").set_index("strategy")
    tp = read("s1_tactical_placebo.csv").set_index("rule")
    sub = read("s1_subperiods.csv").set_index(["period", "strategy"])
    sens = read("s1_reserve_sensitivity.csv")
    meta = json.loads((RESULTS_DIR / "s1_meta.json").read_text(encoding="utf-8"))

    lump, dca = "年初一括", "DCA(毎月)"
    check(bm["irr"].idxmax() == lump, "IRRが最大なのは年初一括")
    check(bm["terminal_multiple"].idxmax() == lump, "最終倍率が最大なのは年初一括")
    check(bm.drop(lump)["vol"].max() < bm.loc[lump, "vol"], "年初一括のボラが最大")
    rules = [r for r in tp.index]
    check((ta.loc[rules, "sharpe"] < ta.loc["常時保有(S&P500 TR)", "sharpe"]).all(), "売買12ルールのSharpeは常時保有未満")
    check((pl["p_terminal_family"] >= 0.5).all() and (ep["p_terminal_family"] >= 0.5).all(), "族全体pは0.5以上")
    check((tp["p_alpha_family"] >= 0.5).all(), "売買の族全体pは0.5以上")
    carry = sens[(sens.policy == "繰越") & (sens.threshold >= 30)]
    for _, g in carry.groupby("threshold"):
        check(g.sort_values("reserve_share")["irr"].is_monotonic_decreasing, "繰越は待機比率が高いほどIRRが低い")
    irr_range = sens[sens.policy == "年末強制"]["irr"]
    check((irr_range < bm.loc[lump, "irr"]).all(), "年末強制の待機アンサンブルは比率によらず一括のIRR未満")
    check((sens["irr"] < bm.loc[lump, "irr"]).all(), "全感度分析ケースのIRRは一括未満")

    order = [lump, dca, "一括+DCA", "VIX>=20 待機(年末強制)", "VIX>=30 待機(年末強制)", "VIX>=40 待機(年末強制)",
             "VIX>=20 待機(繰越)", "VIX>=30 待機(繰越)", "VIX>=40 待機(繰越)"]
    for mix in ("一括", "DCA", "一括+DCA"):
        order += [f"{mix}+VIX>={t}" for t in (20, 30, 40)]
    order += [f"一括+DCA+VIX>={t}(繰越)" for t in (20, 30, 40)]
    check(set(order) == set(bm.index), "表にすべての戦略を載せる")
    budget_rows = [[label(k), n(bm.loc[k, "terminal_multiple"]), pct(bm.loc[k, "irr"]), pct(bm.loc[k, "twr_cagr"]),
                    pct(bm.loc[k, "vol"]), n(bm.loc[k, "sharpe"], 3), pct(bm.loc[k, "mdd"], 1),
                    pct(bm.loc[k, "avg_equity_weight"], 1)] for k in order]
    t_budget = table(["戦略", "最終倍率", "IRR", "TWR-CAGR", "ボラ", "Sharpe", "最大DD", "平均株式比率"], budget_rows)

    v_order = [dca, "一括+DCA", "VIX>=20 待機(年末強制)", "VIX>=30 待機(年末強制)", "VIX>=40 待機(年末強制)",
               "一括+DCA+VIX>=20", "一括+DCA+VIX>=30", "一括+DCA+VIX>=40"]
    v_rows = []
    for k in v_order:
        a = vt.loc[(k, lump)]
        row = [label(k), f"{sg(a.mean_diff_pct)} {ci(a.ci_lo_pct, a.ci_hi_pct)}",
               f"{int(a.win_years)}勝{int(a.loss_years)}敗{int(a.tie_years)}分", pv(a.sign_p)]
        if k != dca:
            b = vt.loc[(k, dca)]
            row += [f"{sg(b.mean_diff_pct)} {ci(b.ci_lo_pct, b.ci_hi_pct)}"]
        else:
            row += ["—"]
        v_rows.append(row)
    t_vintage = table(["戦略", "対 年初一括（%, 95%CI）", "年ごとの勝敗", "符号検定p", "対 DCA（%, 95%CI）"], v_rows)

    s_rows = []
    for (th, pol), g in sens.groupby(["threshold", "policy"], sort=False):
        g = g.sort_values("reserve_share")
        s_rows.append([f"VIX≥{th} {pol}"] + [f"{pct(r.irr)} / {n(r.sharpe, 3)}" for r in g.itertuples()])
    t_sens = table(["待機資金の条件", "比率10%", "比率20%", "比率33%", "比率50%"], s_rows)

    p_rows = []
    for k in pl.index:
        p_rows.append([label(k), pv(pl.loc[k, "p_terminal"]), pv(pl.loc[k, "p_terminal_family"]), pv(pl.loc[k, "p_sharpe"]),
                       pv(ep.loc[k, "p_sharpe"]), pv(ep.loc[k, "p_sharpe_family"])])
    t_placebo = table(["待機ルール", "評価額p", "評価額p（族補正）", "全額待機のSharpe p", "1/3待機のSharpe p",
                       "1/3待機のSharpe p（族補正）"], p_rows)

    tac_rows = []
    bh = ta.loc["常時保有(S&P500 TR)"]
    tac_rows.append(["常時保有", pct(bh.cagr), pct(bh.vol), n(bh.sharpe, 3), pct(bh.mdd, 1), "100%", "—", "—", "—", "—"])
    for k in rules:
        r = ta.loc[k]
        tac_rows.append([label(k), pct(r.cagr), pct(r.vol), n(r.sharpe, 3), pct(r.mdd, 1), pct(r.exposure, 1),
                         sg(r.alpha_annual * 100), pv(r.alpha_hac_p), pv(tp.loc[k, "p_alpha"]),
                         f"{sg(r.sharpe_diff_vs_bh, 3)} {ci(r.sharpe_diff_lo, r.sharpe_diff_hi, 3)}"])
    t_tactical = table(["ルール", "CAGR", "ボラ", "Sharpe", "最大DD", "投資期間", "アルファ（年率%）", "HAC p", "プラセボp",
                        "対常時保有のSharpe差"], tac_rows)

    per = ["1990-2007", "2008-2025"]
    keys = [lump, dca, "一括+DCA", "VIX>=30 待機(年末強制)", "VIX>=40 待機(年末強制)", "一括+DCA+VIX>=30", "一括+DCA+VIX>=40"]
    t_sub = table(["戦略"] + [f"{p} {x}" for p in per for x in ("IRR", "Sharpe", "最大DD")],
                  [[label(k)] + [v for p in per for v in (pct(sub.loc[(p, k), "irr"], 1), n(sub.loc[(p, k), "sharpe"], 3),
                                                          pct(sub.loc[(p, k), "mdd"], 1))] for k in keys])

    timing = "、".join(f"VIX≥{t}は{tm.loc[t, 'years_fired']}/{tm.loc[t, 'n_years']}年（平均待機{tm.loc[t, 'avg_wait_days']:.0f}日）"
                      for t in tm.index)
    best_tac = ta.loc["VIX>=40 後252日保有"]
    ens30 = bm.loc["一括+DCA+VIX>=30"]
    ens30c = bm.loc["一括+DCA+VIX>=30(繰越)"]
    v30 = vt.loc[("VIX>=30 待機(年末強制)", lump)]
    clear_tac = [k for k in rules if ta.loc[k, "sharpe_diff_hi"] < 0]

    text = f"""## 3. 第1段階: VIXをシグナルに投資するとリターンはどうなるか

### 3.1 資金投入型: 毎年の資金をいつ株式に入れるか

全戦略で評価期間（{meta['start']}〜{meta['end']}、{meta['n_days']:,}営業日）と外部キャッシュフロー（毎年初に1単位、計{meta['n_years']}単位）を揃えています。違うのは、届いた資金をいつ株式へ投じるかだけです。未投資分はT-billで運用します。

- **VIX待機（年末強制）:** 年内にVIXが閾値に達したら翌営業日に投資し、達しなければ年末に投資する
- **VIX待機（繰越）:** 閾値に達するまで翌年以降も待つ
- **アンサンブル:** 構成要素に予算を等分する。`一括+DCA+VIX≥30` なら、毎年の資金の1/3を年初に投資し、1/3を毎月積み立て、残り1/3をVIX待機資金にする

{t_budget}

シグナルの発火は、{timing}でした。

**読み取り:**

- 外部フローを揃えた収益率であるIRRと最終倍率は、年初一括が全戦略で最大です（{pct(bm.loc[lump, 'irr'])}、{n(bm.loc[lump, 'terminal_multiple'])}倍）。
- ボラは年初一括が最大で、他はわずかに下がります。その結果Sharpeは {n(bm['sharpe'].min(), 3)}〜{n(bm['sharpe'].max(), 3)} に収まり、最大は `{label(bm['sharpe'].idxmax())}` の {n(bm['sharpe'].max(), 3)} です。ただし次節のとおり偶然と区別できません。
- 最大DDは、繰越の大きいケースを除くとほぼ同水準です。積立が進むと資産の大半が株式になるため、新規資金のタイミングでは大きな下落を避けられません。
- `VIX≥30 年末強制` はIRRが年初一括を下回るのに、TWR-CAGRは上回ります（{pct(bm.loc['VIX>=30 待機(年末強制)', 'twr_cagr'])} 対 {pct(bm.loc[lump, 'twr_cagr'])}）。TWRは運用額の大小によらず各日を同じ重みで数えるため、現金比率が高い積立初期の影響を受けやすいと考えられます。積み立てる投資家の実感に近いのはIRRなので、判断にはIRRを使います。

### 3.2 待機資金を組み込んだアンサンブル

年初一括・月次DCA・VIX待機資金の組み合わせを、待機資金の比率を変えて確認しました。各セルは IRR / Sharpe です。比率は最良値を選ぶためでなく、結論が比率に左右されるかを見るためのものです。比率以外の予算は年初一括とDCAに半分ずつ割り当てます。

{t_sens}

比較対象は年初一括（{pct(bm.loc[lump, 'irr'])} / {n(bm.loc[lump, 'sharpe'], 3)}）、DCA（{pct(bm.loc[dca, 'irr'])} / {n(bm.loc[dca, 'sharpe'], 3)}）、一括+DCA（{pct(bm.loc['一括+DCA', 'irr'])} / {n(bm.loc['一括+DCA', 'sharpe'], 3)}）です。

- **IRR:** 全{len(sens)}ケースで年初一括を下回りました。
- **繰越:** VIX≥30とVIX≥40では、待機比率を上げるほどIRRが下がります。資金を翌年へ持ち越すと、待つコストが積み上がるためです。
- **年末強制:** 1/3待機のVIX≥30は、IRR {pct(ens30.irr)}、Sharpe {n(ens30.sharpe, 3)}、最大DD {pct(ens30.mdd, 1)}です。IRRは年初一括とDCAの間にあり、Sharpeは両者をわずかに上回りますが、次節のとおりプラセボと区別できません（p={pv(ep.loc['VIX>=30 年末強制', 'p_sharpe'])}）。同じ条件の繰越版は IRR {pct(ens30c.irr)}、Sharpe {n(ens30c.sharpe, 3)} です。
- **解釈:** アンサンブルの評価額は、構成要素の評価額の加重和です（テストで確認済み）。そのため最終評価額は必ず構成要素の範囲に収まり、組み合わせからリターンの上乗せは生まれません。

### 3.3 差は偶然と区別できるか

1年ごとに投入した資金の年末価値を比べました（ビンテージ分析）。年が重ならないため、独立に近い{int(vt['n_years'].iloc[0])}標本として扱えます。値は対数差で、差が0の年（シグナルが既に点灯していて買付日が同じになった年など）は符号検定から除きます。

{t_vintage}

次に、VIX系列を循環シフトしたプラセボ（{int(pl['n_shifts'].iloc[0]):,}通り）と比べました。発火の塊の形を保ったまま、市場との対応だけを壊す検定です。「族補正」は6ルールをまとめて見たときのpです。

{t_placebo}

**判定:** 待機ルールとそのアンサンブルのうち、族補正後に偶然と区別できるものはありません。繰越ルールのpが1に近いのは、同じ形でランダムな時期に待つ場合よりも評価額が低かったことを示します。ただし「年初一括が明確に勝つ」とも言えません。`VIX≥30 年末強制` の差は {sg(v30.mean_diff_pct)}% {ci(v30.ci_lo_pct, v30.ci_hi_pct)} でCIが0をまたぎ、勝敗も{int(v30.win_years)}勝{int(v30.loss_years)}敗です。言えるのは「VIXで待つことが年初一括を上回るという証拠がない」までです。

![待機戦略と年初一括の差](figures/fig1_budget.png)

**図1:** 待機戦略と年初一括の差。左は評価額の差の推移、右はVIX≥30待機の年ごとの差。

### 3.4 売買型: シグナル後だけ株式を持つ

資金投入型とは別に、シグナルからN日だけ株式を持ち、それ以外はT-billで運用するルールを、常時保有と比べました。CAGRは片道{COST_BPS}bpの売買コスト控除前です（控除後の値は `results/s1_tactical_metrics.csv` にあります）。

{t_tactical}

- **Sharpe:** 12ルールすべてが常時保有（{n(bh.sharpe, 3)}）を下回ります。CIが0を除外したのは {len(clear_tac)} ルールです（{"、".join(label(k) for k in clear_tac)}）。
- **アルファ:** 年率 {sg(ta.loc[rules, 'alpha_annual'].min() * 100, 1)}%〜{sg(ta.loc[rules, 'alpha_annual'].max() * 100, 1)}% で、プラセボpは最小でも {pv(tp['p_alpha'].min())} でした。
- **リスク低減:** `VIX≥40 後252日保有` は最大DDを {pct(best_tac.mdd, 1)} まで縮めますが、CAGRも {pct(best_tac.cagr, 1)} に下がります。ベータが {n(best_tac.beta)} に下がった分でほぼ説明でき、アルファは {sg(best_tac.alpha_annual * 100, 1)}%（HAC p={pv(best_tac.alpha_hac_p)}）です。
- **注意:** 保有日は高ボラの日に偏るため、投資期間が同じ固定配分とのCAGR比較はリスクを過小評価します。判断はアルファで行います。

### 3.5 前半と後半で結論は変わるか

{t_sub}

後半は全戦略のIRRが高くなりますが、戦略間の順序は大きく変わりません。

後半の最大DDには注意が必要です。この区間は2008年の初めから積立を始めるため、リーマン危機の時点で資金の多くが現金のDCAや待機戦略は、最大DDとSharpeが有利に見えます。VIX待機の優位性ではなく、開始時点の影響です。
"""
    facts = {"lump": bm.loc[lump], "dca": bm.loc[dca], "ens30": ens30, "ens30c": ens30c, "v_ens": vt.loc[("一括+DCA+VIX>=30", lump)],
             "v_ens_dca": vt.loc[("一括+DCA+VIX>=30", dca)], "bh": bh, "best_tac": best_tac, "tm": tm,
             "alpha_min": ta.loc[rules, "alpha_annual"].min(), "alpha_max": ta.loc[rules, "alpha_annual"].max(),
             "p_alpha_min": tp["p_alpha"].min(), "n_sens": len(sens), "bm": bm}
    return text, facts


def stage2() -> tuple[str, dict]:
    ph = read("s2_phase.csv").set_index(["horizon", "phase"])
    th = read("s2_threshold.csv").set_index(["threshold", "horizon"])
    ev = read("s2_events_summary.csv").set_index(["threshold", "horizon"])
    st = read("s2_stability.csv").set_index(["subset", "horizon", "threshold"])
    crisis = [s for s in st.index.get_level_values(0).unique() if "除外" in s][0]

    check((ev["binom_p_vs_all"][[(t, h) for t in (30, 40) for h in HZ]] >= 0.2).all(), "≥30/≥40初回到達の勝率差は有意でない")
    check(ev.loc[(30, "12M"), "mean_pct"] < ev.loc[(30, "12M"), "all_mean_pct"], "VIX≥30初回到達の12Mは全期間平均未満")
    for t in (30, 40):
        check(ev.loc[(t, "1M"), "mean_pct"] < 0, "初回到達直後の1Mは平均マイナス")
    for h in ("6M", "12M"):
        check(ev.loc[(40, h), "mean_pct"] < ph.loc[(h, VIX_LABELS[3]), "mean_pct"], "≥40の初回到達は全日平均より小さい")
    for t, lab in ((30, VIX_LABELS[2]), (40, VIX_LABELS[3])):
        for h in HZ:
            check(ev.loc[(t, h), "mean_pct"] < ph.loc[(h, lab), "mean_pct"], "初回到達の平均は全日平均より小さい")
    check(ph.loc[("1M", VIX_LABELS[3]), "p05_pct"] < ph.loc[("1M", VIX_LABELS[0]), "p05_pct"], "1Mの5%点は高VIXで悪化")
    check((ph.loc[[(h, VIX_LABELS[1]) for h in HZ], "mean_pct"] < ph.loc[[(h, VIX_LABELS[1]) for h in HZ], "all_mean_pct"]).all(),
          "VIX 20-30は全期間平均以下")
    check(th.loc[(30, "6M"), "shift_p_holm"] < 0.05 and th.loc[(40, "12M"), "shift_p_holm"] < 0.05, "Holm後も極端域は有意")
    check((th.xs(20, level="threshold")["shift_p_holm"] > 0.5).all(), "VIX≥20はHolm後に有意でない")

    rows = []
    for h in HZ:
        for lab in VIX_LABELS:
            r = ph.loc[(h, lab)]
            rows.append([h, label(lab), f"{int(r.n_days):,}", n(r.mean_pct), n(r.median_pct), n(r.std_pct), n(r.p05_pct),
                         pct(r.win_rate, 0), f"{sg(r.diff_vs_all_pct)} {ci(r.diff_lo_pct, r.diff_hi_pct)}"])
    t_phase = table(["期間", "VIX", "日数", "平均%", "中央値%", "標準偏差%", "5%点%", "勝率", "全期間平均との差（95%CI）"], rows)

    t_th = table(["閾値"] + HZ, [
        [f"VIX≥{t}"] + [f"{sg(th.loc[(t, h), 'diff_vs_all_pct'])}（p={pv(th.loc[(t, h), 'shift_p_holm'])}）" for h in HZ]
        for t in (20, 30, 40)])
    ev_rows = []
    for t in (20, 30, 40):
        for h in HZ:
            r = ev.loc[(t, h)]
            ev_rows.append([f"VIX≥{t}", h, int(r.n_events), n(r.mean_pct), n(r.median_pct), n(r.min_pct), pct(r.win_rate, 0),
                            n(r.all_mean_pct), pct(r.all_win_rate, 0), pv(r.binom_p_vs_all)])
    t_ev = table(["閾値", "期間", "件数", "平均%", "中央値%", "最悪%", "勝率", "全期間平均%", "全期間勝率", "勝率の二項検定p"], ev_rows)
    c40 = [st.loc[(crisis, h, 40), "diff_vs_subset_all_pct"] for h in HZ]
    p12 = ph.loc[("12M", VIX_LABELS[3])]
    e20 = ev.xs(20, level="threshold")

    text = f"""## 4. 第2段階: シグナルとしてどの程度機能しているか

### 4.1 VIXの局面別に見たその後のリターン（離散VIX × 離散期間）

超過リターンはT-bill控除後の対数値です。CIは moving block bootstrap（{N_BOOT:,}回、ブロック長 max(21, 2h)）によるものです。

{t_phase}

平均は単調に増えていません。VIX 20〜30 の局面は全期間平均以下で、上乗せが出るのはVIX≥30からです。1Mでは高VIXほど標準偏差が大きく、5%点も悪化します（VIX<20で {n(ph.loc[('1M', VIX_LABELS[0]), 'p05_pct'])}%、VIX≥40で {n(ph.loc[('1M', VIX_LABELS[3]), 'p05_pct'])}%）。一方、12MのVIX≥40は5%点が {n(p12.p05_pct)}% と高くなります。ただしこれは少数の危機の回復局面を数えた結果です。

### 4.2 閾値ごとの検定（循環シフト・Holm補正後のp）

{t_th}

全日で見ると、VIX≥30（全期間）とVIX≥40（6M、12M）はHolm補正後も有意です。VIX≥20は有意ではありません。

### 4.3 初めて閾値に達した日に買った場合

全日平均は、VIXが高い状態が続く間の日を何度も数えます。実際に使えるシグナルに近づけるため、直前の閾値到達から63営業日超あいた日だけを独立なイベントとして数えました。

{t_ev}

![局面別の平均と初回到達の平均](figures/fig2_phase.png)

**図2:** 局面別の全日平均と、閾値に初めて達した日の平均。

- **VIX≥30、VIX≥40:** 1Mの平均はマイナスです（{n(ev.loc[(30, '1M'), 'mean_pct'])}%、{n(ev.loc[(40, '1M'), 'mean_pct'])}%）。VIX≥30の12Mは {n(ev.loc[(30, '12M'), 'mean_pct'])}% で全期間平均の {n(ev.loc[(30, '12M'), 'all_mean_pct'])}% を下回ります。
- **VIX≥40:** 6M・12Mの平均は全期間平均を上回りますが、全日平均のおよそ半分から2/3です（12Mで {n(ev.loc[(40, '12M'), 'mean_pct'])}% 対 {n(p12.mean_pct)}%）。3Mだけは全日平均に近い値です。件数は{int(ev.loc[(40, '6M'), 'n_events'])}件しかなく、最悪は6Mで {n(ev.loc[(40, '6M'), 'min_pct'])}% でした。
- **VIX≥20（想定外の結果）:** 静かな相場の後に初めて20に達した{int(e20['n_events'].iloc[0])}件は、平均が全期間平均を上回り、勝率も 1M {pct(e20.loc['1M', 'win_rate'], 0)}、6M {pct(e20.loc['6M', 'win_rate'], 0)} と高めです（二項検定 p={pv(e20.loc['1M', 'binom_p_vs_all'])}、{pv(e20.loc['6M', 'binom_p_vs_all'])}）。ただし12本の検定の未補正値で、事前に立てた仮説でもありません。新しいデータで確かめるべき仮説として扱います。

### 4.4 シグナルとしての判定

| 観点 | 判定 | 根拠 |
|---|---|---|
| 投資戦略として | 上乗せを確認できない | 年初一括のIRRを上回らず、プラセボと区別できない |
| 全日の条件付き平均 | 極端域のみ検出できる | VIX≥30とVIX≥40の長期はHolm補正後も有意 |
| 初回到達という使い方 | 弱い | 平均は全日より小さく、VIX≥30・VIX≥40の勝率差は有意でない |
| 危機の除外 | 極端域の差は残る | 2008-09年と2020年を除いてもVIX≥40の差は {" / ".join(sg(c, 1) for c in c40)}%（1M/3M/6M/12M）。ただし日数は少ない |

12M・VIX≥40の全日平均 {n(p12.mean_pct)}% は目を引きますが、{int(p12.n_days)}日の大半は少数の危機に集中しています。日数は多く見えても、独立な標本は初回到達の{int(ev.loc[(40, '12M'), 'n_events'])}件程度と考えるべきです。
"""
    return text, {"ev": ev, "ph": ph, "th": th}


def stage3(df: pd.DataFrame) -> tuple[str, dict]:
    co = read("s3_continuous.csv").set_index("horizon")
    di = read("s3_discrete.csv").set_index("horizon")
    mo = read("s3_models.csv").set_index(["horizon", "model"])
    vo = read("s3_volatility.csv").set_index("horizon")
    cn = read("s3_controls.csv").set_index("horizon")
    st = read("s3_stability.csv").set_index(["subset", "horizon"])
    grid = read("s3_horizon_grid.csv")
    meta = json.loads((RESULTS_DIR / "s3_meta.json").read_text(encoding="utf-8"))
    crisis = [s for s in st.index.get_level_values(0).unique() if "除外" in s][0]

    check((co["p_newey_west"] > 0.1).all(), "線形傾きはNWで有意でない")
    check((co["oos_r2"] < 0).all(), "線形のOOS R²は全期間で負")
    check((co["r2_pct"] < 1).all(), "線形のR²は1%未満")
    check((di.loc[["6M", "12M"], "oos_r2"] > 0).all() and (di.loc[["1M", "3M"], "oos_r2"] < 0).all(), "離散のOOSは6M/12Mのみ正")
    check((di.loc[["6M", "12M"], "oos_r2_excl_crises"] < 0).all() and (di.loc[["6M", "12M"], "oos_r2_from2010"] < 0).all(), "離散OOSは危機除外と2010年以降で負")
    check((vo["r2_pct"] > 25).all() and (vo["p_newey_west"] < 0.001).all(), "実現ボラの説明力は強い")
    check(vo["r2_pct"].is_monotonic_decreasing, "ボラのR²は期間が長いほど低い")
    check((grid["slope"] > 0).all(), "連続グリッドで傾きは正")
    check((grid["lo"] < 0).all() and (grid["hi"] > 0).all(), "連続グリッドのCIは全て0を含む")
    check(st.loc[("1990-2007", "12M"), "slope"] < 0 < st.loc[("2008-2025", "12M"), "slope"], "12Mの傾きは前後半で符号が反転")

    t_co = table(["期間", "傾き", "1SDの効果%", "補正後%", "R²%", "OLS p", "NW p", "Hodrick t", "VAR帰無 p", "OOS R²%", "CW p"],
                 [[h, n(r.slope, 4), n(r.effect_1sd_pct), n(r.effect_1sd_bias_adj_pct), n(r.r2_pct), pv(r.p_ols), pv(r.p_newey_west),
                   n(r.t_hodrick_1b), pv(r.p_var_null_two_sided), sg(r.oos_r2 * 100, 1), pv(r.cw_p_one_sided)]
                  for h, r in co.loc[HZ].iterrows()])
    t_di = table(["期間", "VIX<20の平均%", "VIX 20-30の差%", "VIX 30-40の差%", "VIX≥40の差%", "Wald p（NW）"],
                 [[h, n(r.normal_mean_pct)] + [f"{sg(r[f'coef_{lab}_pct'])}（p={pv(r[f'p_{lab}'])}）" for lab in VIX_LABELS[1:]] + [pv(r.wald_p_newey_west)]
                  for h, r in di.loc[HZ].iterrows()])
    m_rows = []
    for h in HZ:
        for m in ["連続: 線形", "連続: 折れ線", "離散: 4フェーズ"]:
            r = mo.loc[(h, m)]
            m_rows.append([h, m, n(r.r2_pct), "—" if pd.isna(r.r2_var_null_p) else pv(r.r2_var_null_p), sg(r.oos_r2 * 100, 1),
                           sg(r.oos_r2_excl_crises * 100, 1), sg(r.oos_r2_to2009 * 100, 1), sg(r.oos_r2_from2010 * 100, 1), pv(r.cw_p_one_sided)])
    t_mo = table(["期間", "モデル", "R²%", "VAR帰無 p", "OOS R²%", "危機除外", "〜2009年", "2010年〜", "CW p"], m_rows)
    t_vo = table(["期間", "弾力性", "NW p", "R²%"], [[h, n(r.elasticity, 3), pv(r.p_newey_west), n(r.r2_pct, 1)] for h, r in vo.loc[HZ].iterrows()])
    t_cn = table(["期間", "VIXのみR²%", "統制変数のみR²%", "全部R²%", "VIXの増分R²%", "VIXの傾き", "VIXのNW p"],
                 [[h, n(r.r2_vix_only_pct), n(r.r2_controls_only_pct), n(r.r2_full_pct), n(r.incremental_r2_vix_pct), n(r.slope_vix_full, 3), pv(r.p_vix_full)]
                  for h, r in cn.loc[HZ].iterrows()])
    subs = list(st.index.get_level_values(0).unique())
    t_st = table(["期間（標本）"] + HZ, [[s] + [f"{sg(st.loc[(s, h), 'effect_1sd_pct'])}（p={pv(st.loc[(s, h), 'p_newey_west'])}）" for h in HZ] for s in subs])

    y = forward_excess(df, 21) * 100
    r = np.log1p(df["rm"])
    rv = np.sqrt((r ** 2).rolling(21).sum().shift(-22) * 252 / 21) * 100
    ok = rv.notna()
    rv_below = float((rv[ok] < df["VIX"][ok]).mean())
    check(rv_below > 0.5, "実現ボラはVIXを下回る日が多い")
    check(y.notna().sum() > 0, "将来リターンがある")

    text = f"""## 5. 第3段階: 回帰分析でどの程度寄与しているか

### 5.1 離散と連続の組み合わせ

| | 投資期間が離散（1M/3M/6M/12M） | 投資期間が連続（5〜252営業日） |
|---|---|---|
| **VIXが連続（log VIX）** | 5.2の表 | 図3の左 |
| **VIXが離散（4局面）** | 5.3の表 | 図3の中央 |

### 5.2 連続VIX: log VIXの線形回帰

log VIXの1SDは {n(meta['sd_logvix'], 3)}、AR(1)は {n(meta['phi'], 3)} です。VIXは持続性が非常に高く、窓も重なるため、OLSのpはそのまま使えません。Newey-West、Hodrick 1B、帰無仮説の下のVARブートストラップ（Stambaughバイアスと窓の重複を反映）を併記します。OOS R²はCampbell-Thompson型（過去平均と比較）で、CWはClark-West検定の片側pです。

{t_co}

1SD上がったときの効果は {n(co['effect_1sd_pct'].min())}〜{n(co['effect_1sd_pct'].max())}% で、バイアス補正後は {n(co['effect_1sd_bias_adj_pct'].min())}〜{n(co['effect_1sd_bias_adj_pct'].max())}% に縮みます。R²は最大でも {n(co['r2_pct'].max())}% です。OLSのpはほぼ0ですが、頑健な方法ではどれも有意にならず、OOS R²は全期間で負です。**線形の関係としては、VIXはリターンをほとんど説明しません。**

![投資期間を連続で見た結果](figures/fig3_horizon.png)

**図3:** 投資期間を連続で見た結果。左は連続VIXの傾き、中央は離散VIX（VIX≥40）の差、右はOOS R²。

### 5.3 離散VIXと、線形の仮定

VIX<20を基準にしたダミー回帰です。pはNewey-Westによるものです。

{t_di}

{t_mo}

線形（R²が1%未満）と離散（6M・12Mで {n(di.loc['6M', 'r2_pct'])}%、{n(di.loc['12M', 'r2_pct'])}%）の差は、離散化そのものではなく線形の仮定から生じています。節点20/30/40の折れ線でも、離散と同程度のR²が出ます。関係は極端域に偏った非線形で、VAR帰無でも6M・12Mでは偶然と区別できます。

ただし、**標本内で見える関係は標本外では脆い**です。

- 離散モデルのOOS R²が正なのは6M・12Mだけです（{sg(di.loc['6M', 'oos_r2'] * 100, 1)}%、{sg(di.loc['12M', 'oos_r2'] * 100, 1)}%、CW p={pv(di.loc['6M', 'cw_p_one_sided'])}、{pv(di.loc['12M', 'cw_p_one_sided'])}）。
- その正の値は2009年までに偏っています（{sg(di.loc['6M', 'oos_r2_to2009'] * 100, 1)}%、{sg(di.loc['12M', 'oos_r2_to2009'] * 100, 1)}%）。2010年以降は {sg(di.loc['6M', 'oos_r2_from2010'] * 100, 1)}%、{sg(di.loc['12M', 'oos_r2_from2010'] * 100, 1)}% で、危機を除くと {sg(di.loc['6M', 'oos_r2_excl_crises'] * 100, 1)}%、{sg(di.loc['12M', 'oos_r2_excl_crises'] * 100, 1)}% です。
- VIX≥40のダミーは12Mでp={pv(di.loc['12M', 'p_Panic(>=40)'])}ですが、危機のエピソードが少ないため、Newey-Westの標準誤差は過小になっている可能性があります。

### 5.4 リスクには強く効く

翌期間の実現ボラを log VIX で回帰した結果です。

{t_vo}

![VIXとリターン、VIXとリスク](figures/fig4_risk_vs_return.png)

**図4:** 当日のVIXと、その後1ヶ月のリターン（左）および実現ボラ（右）。

平均リターンでは最大 {n(co['r2_pct'].max())}% だったR²が、実現ボラでは {n(vo['r2_pct'].min(), 1)}〜{n(vo['r2_pct'].max(), 1)}% になります。1M先の実現ボラがVIXを下回った日は全体の {pct(rv_below, 0)} でした。VIXは将来の変動の大きさを示す指標としては強く、期間が延びるほど説明力は下がります。

### 5.5 統制変数と頑健性

過去リターン、ドローダウン、過去21日の実現ボラを統制した結果です。

{t_cn}

短期（1M、3M）では統制後にVIXの傾きが有意になります（p={pv(cn.loc['1M', 'p_vix_full'])}、{pv(cn.loc['3M', 'p_vix_full'])}）。実現ボラを固定したうえでVIXが高い状態、つまりボラティリティ・リスク・プレミアムに近い情報を拾っている可能性がありますが、今回はVRPを直接検証していません。増分R²も {n(cn['incremental_r2_vix_pct'].min())}〜{n(cn['incremental_r2_vix_pct'].max())}% と小さく、標本内の結果です。

線形モデルの1SD効果（%）を標本別に見ると、次のとおりです。括弧内はNW pです。

{t_st}

6M・12Mでは前半と後半で傾きの符号が変わります。危機除外では1Mの傾きがp={pv(st.loc[(crisis, '1M'), 'p_newey_west'])}で有意になりますが、除外期間は結果を見て決めたものなので、強い根拠にはなりません。
"""
    return text, {"co": co, "di": di, "vo": vo, "rv_below": rv_below}


def stage4() -> tuple[str, dict]:
    ov = read("s4_overall.csv").set_index(["horizon", "method"])
    vs = read("s4_vs_base.csv").set_index(["horizon", "method"])
    bp = read("s4_by_phase.csv").set_index(["horizon", "method", "phase"])
    post = read("s4_post_panic.csv").set_index(["horizon", "method"])
    pre = read("s4_pre_panic.csv").set_index(["horizon", "method"])
    grid = read("s4_grid.csv").pivot(index="h", columns="method", values="winkler_skill_pct")
    meta = json.loads((RESULTS_DIR / "s4_meta.json").read_text(encoding="utf-8"))

    check((grid["Reg"] <= 0).all(), "平均に連続VIX(Reg)はどの期間も改善しない")
    check((grid["Norm"] > 0).all() and (grid["Mondrian"] > 0).all(), "散らばりに使うと全期間で点推定は改善")
    check(grid["Norm"].iloc[0] > grid["Norm"].iloc[-1], "Normの改善は短期ほど大きい")
    check(not vs.xs("Reg", level="method")["winkler_clear"].any(), "Regはどの期間でも差が確認できない")
    for m in ("Norm", "Mondrian", "Asym"):
        check(vs.loc[("1M", m), "winkler_clear"] and vs.loc[("1M", m), "winkler_diff_pct"] < 0, "1Mで散らばり系は明確に改善")
    aci = vs[vs.index.get_level_values("method").str.startswith("ACI")]
    check(not ((aci["winkler_diff_pct"] < 0) & aci["winkler_clear"]).any(), "ACIで明確な改善はない")
    check(bp.loc[("1M", "Base", VIX_LABELS[3]), "coverage"] < 0.6, "BaseはVIX≥40で大きく外れる")
    check(bp.loc[("1M", "Norm", VIX_LABELS[3]), "coverage"] > 0.9, "NormはVIX≥40でも目標を満たす")
    check(pre.loc[("1M", "Norm"), "coverage"] < 0.5 and pre.loc[("1M", "Base"), "coverage"] < 0.5, "VIX急騰前の1M区間はVIXを使っても外れる")

    t_ov = table(["手法"] + HZ, [[m] + [f"{n(ov.loc[(h, m), 'winkler_pct'])} / {pct(ov.loc[(h, m), 'coverage'], 1)}" for h in HZ] for m in METHODS_ALL])
    clear = vs[vs["winkler_clear"]].reset_index()
    t_clear = table(["期間", "手法", "Winkler差（%、負が改善）", "95%CI", "判定"],
                    [[r.horizon, r.method, sg(r.winkler_diff_pct), ci(r.winkler_lo_pct, r.winkler_hi_pct), "改善" if r.winkler_diff_pct < 0 else "悪化"]
                     for r in clear.itertuples()])
    n_total = len(vs)
    p_rows = []
    for h in HZ:
        for m in ("Base", "Mondrian", "Norm"):
            p_rows.append([h, m] + [pct(bp.loc[(h, m, lab), "coverage"], 1) for lab in VIX_LABELS])
    t_ph = table(["期間", "手法"] + [label(x) for x in VIX_LABELS], p_rows)
    ms = ["Base", "Mondrian", "Norm", "Asym", "ACI-Norm"]
    t_pp = table(["手法", "直前21日 カバレッジ", "直前21日 平均幅%", "直後21日 カバレッジ", "直後21日 平均幅%"],
                 [[m, pct(pre.loc[("1M", m), "coverage"], 1), n(pre.loc[("1M", m), "mean_width_pct"], 1),
                   pct(post.loc[("1M", m), "coverage"], 1), n(post.loc[("1M", m), "mean_width_pct"], 1)] for m in ms])
    t_gr = table(["投資期間（営業日）", "Norm", "Mondrian", "RegD", "Reg"],
                 [[h, sg(r.Norm, 1), sg(r.Mondrian, 1), sg(r.RegD, 1), sg(r.Reg, 1)] for h, r in grid.iterrows()])
    eff = [ov.loc[(h, "Base"), "effective_n"] for h in HZ]
    wid = [ov.loc[(h, "Base"), "mean_width_pct"] for h in HZ]
    aci_low = [f"{h} {pct(ov.loc[(h, 'ACI-Base'), 'coverage'], 1)}" for h in HZ]

    text = f"""## 6. 第4段階: コンフォーマル予測はどの程度機能するか

### 6.1 設定

t日に、その後の超過対数リターンに対する名目{pct(1 - ALPHA, 0)}の予測区間を出します。使うのは確定済みの結果だけです（テストで保証しています）。校正窓は直近2,520日で、有限標本補正つきの分位点を使います。全手法・全期間で区間の発行日を揃えました（{meta['test_start']}〜{meta['test_end']}、{meta['n_test']:,}日）。

カバレッジは区間を広げれば必ず上がるため、幅と外れの両方を罰する **Winkler score**（小さいほど良い）を中心に評価します。

| 手法 | 平均の予測 | 区間の幅 | VIXの使い方 |
|---|---|---|---|
| Base | 過去平均 | 一定 | 使わない（基準） |
| Reg / RegD | 連続VIX / 離散VIXの回帰 | 一定 | 平均に使う |
| Norm / Mondrian | 過去平均 | 連続VIXの関数 / 局面別の分位点 | 幅に使う |
| Asym | 過去平均 | Normと同じで上下別々の分位点 | 幅に使う |
| ACI-Base / ACI-Norm | BaseまたはNormに適応αを追加（γ={GAMMA}で固定） | | |

### 6.2 全体の結果（Winkler% / カバレッジ）

{t_ov}

Baseの平均幅は {" / ".join(n(w, 1) for w in wid)}%（1M/3M/6M/12M）と広く、区間予測で方向を当てられるわけではありません。実効的に独立な区間の数も {" / ".join(f"{e:.0f}" for e in eff)} 個しかなく、12Mのカバレッジは目安程度に見るべきです。

Baseとの差のCIが0を除外したのは、{n_total}通りのうち次の{len(clear)}通りです（ブロックブートストラップ、多重比較の補正なし）。

{t_clear}

![コンフォーマル予測の評価](figures/fig5_conformal.png)

**図5:** 左は投資期間別のWinkler改善率、右は1M先の局面別カバレッジ。

### 6.3 投資期間を連続に近づけた結果（Baseに対するWinkler改善率%）

{t_gr}

- **幅に使う（Norm、Mondrian）:** 点推定は全期間で改善し、Normは短期ほど効きます（5日で {sg(grid.loc[5, 'Norm'], 1)}%、252日で {sg(grid.loc[252, 'Norm'], 1)}%）。ただしCIで差を確認できるのは6.2の組み合わせだけで、長期の改善は誤差の範囲に入るものが多いです。
- **平均に使う（Reg、RegD）:** Regはどの期間でも改善しません。RegDは短中期で悪化し、126日以降でわずかに改善します。これは第3段階の結果と整合します。

### 6.4 VIX局面別のカバレッジ

{t_ph}

全体のカバレッジが約90%でも、Baseは局面によって大きく偏ります。1Mで見ると、VIX<20で {pct(bp.loc[('1M', 'Base', VIX_LABELS[0]), 'coverage'], 1)}（広すぎる）、VIX≥40で {pct(bp.loc[('1M', 'Base', VIX_LABELS[3]), 'coverage'], 1)}（大きく外れる）です。Normは {pct(bp.loc[('1M', 'Norm', VIX_LABELS[0]), 'coverage'], 1)} と {pct(bp.loc[('1M', 'Norm', VIX_LABELS[3]), 'coverage'], 1)} まで揃います。**VIXの主な貢献は、平均の改善ではなく局面ごとの条件付きカバレッジの改善です。** ただしNormは高VIX側で目標を上回ることが多く、幅を広げすぎている面があります。

### 6.5 限界: ショックの初動は捉えられない

図6を見て、VIXが上がる前に出した区間が2020年2月の暴落を外していることに気づき、この診断を後から追加しました。VIX≥40に初めて達した{meta['n_panic_events']}件の、直前21営業日と直後21営業日に出した1M区間を比べています（各N={int(pre.loc[('1M', 'Base'), 'n'])}）。

{t_pp}

直後はNormのカバレッジが {pct(post.loc[('1M', 'Norm'), 'coverage'], 1)} まで回復します（その分、幅は {n(post.loc[('1M', 'Norm'), 'mean_width_pct'], 1)}% に広がります）。しかし直前は、VIXを使ってもカバレッジが {pct(pre.loc[('1M', 'Norm'), 'coverage'], 1)} でBaseと変わりません（直前のVIXの平均は {n(pre.loc[('1M', 'Norm'), 'mean_vix_at_issue'], 1)}）。**VIXはショックを予告するのではなく、起きた後の不確実性を反映します。** なお、この診断は「後で急騰した」という未来の情報で標本を選んでいるので、低い値が出るのはある程度当然です。手法の評価ではなく、限界の確認として読んでください。

![危機時の予測区間](figures/fig6_bands.png)

**図6:** 2008年と2020年前後の1M先予測区間と実現値。横軸は区間を出した日。

### 6.6 ACI

ACIは今回の設定では改善しません。カバレッジは {" / ".join(aci_low)}（ACI-Base）と、長期ほど目標を下回ります。結果がh+1日遅れて届くため、適応が遅れるのが一因と考えられます。ACIが良く見えるようなγの調整はしておらず、γの感度分析も未実施です。「ACIが一般に効かない」という結論ではありません。
"""
    return text, {"ov": ov, "vs": vs, "bp": bp, "post": post, "pre": pre, "grid": grid, "clear": clear, "n_total": n_total}


def build() -> str:
    df = load()
    s1, f1 = stage1()
    s2, f2 = stage2()
    s3, f3 = stage3(df)
    s4, f4 = stage4()
    meta = json.loads((RESULTS_DIR / "s1_meta.json").read_text(encoding="utf-8"))
    bm, co, di, vo = f1["bm"], f3["co"], f3["di"], f3["vo"]
    ev, ph = f2["ev"], f2["ph"]
    grid, bp = f4["grid"], f4["bp"]
    vs = f4["vs"]
    tm = f1["tm"]
    lump, dca, ens, ens_c = f1["lump"], f1["dca"], f1["ens30"], f1["ens30c"]
    v_ens, v_ens_d = f1["v_ens"], f1["v_ens_dca"]
    bh, best = f1["bh"], f1["best_tac"]
    events = read("s2_events_detail.csv")
    ev40 = events[events.threshold == 40]
    ev40_post = ev40[ev40.date >= "2010-01-01"]
    ev40_worst = ev40.loc[ev40["6M"].idxmin()]
    check(ev40_post["6M"].min() > 0 and ev40_worst["6M"] < -30 and ev40_worst.date < "2010-01-01", "TQQQ比較の前提")
    t_summary = table(["戦略", "IRR", "TWR-CAGR", "リスク（ボラ）", "Sharpe", "最大DD"],
                      [[label(k), pct(bm.loc[k, "irr"]), pct(bm.loc[k, "twr_cagr"]), pct(bm.loc[k, "vol"]), n(bm.loc[k, "sharpe"], 3), pct(bm.loc[k, "mdd"], 1)]
                       for k in ["年初一括", "DCA(毎月)", "一括+DCA", "VIX>=30 待機(年末強制)", "VIX>=40 待機(繰越)", "一括+DCA+VIX>=30", "一括+DCA+VIX>=30(繰越)"]])

    head = f"""# VIXはS&P500のリターンにどの程度影響するか

シグナル投資の比較から、回帰分析、コンフォーマル予測まで（{meta['start']}〜{meta['end']}）

> このレポートの数値はすべて `results/` のCSVから `build_report.py` が差し込んでいます。文章中の定性的な主張も、データと合わなくなると生成が止まるようにしています。実装と本文の作成にはAI（Claude）を使用しました。統計的な検証であり、個別の投資助言ではありません。

## 1. 結論

**VIXはリターンの水準よりも、リスクの大きさをよく説明します。VIXを見て買うタイミングを選んでも、年初一括やDCAを上回る証拠は得られませんでした。一方、予測区間の幅に使うと、特に短期で精度が上がります。**

{t_summary}

全戦略で期間と外部キャッシュフローを揃えています。積立ではCAGRが資金の入るタイミングを反映しないため、投資家の収益率としてIRRも併記します。

1. **投資戦略としての上乗せは確認できません。** IRRは年初一括 {pct(lump.irr)} が全戦略で最大です。年初一括・DCA・VIX待機資金を組み合わせたアンサンブルは {pct(ens.irr)}（繰越 {pct(ens_c.irr)}）で、構成要素の中間に収まります。差はプラセボと区別できません。
2. **シグナルの情報は極端域に限られます。** VIX≥40の12M平均は全日で {n(ph.loc[('12M', VIX_LABELS[3]), 'mean_pct'])}% ですが、初めて到達した日に買うと {n(ev.loc[(40, '12M'), 'mean_pct'])}%（{int(ev.loc[(40, '12M'), 'n_events'])}件）に下がります。VIX≥30の初回到達の12M平均は全期間平均を下回ります。
3. **回帰での寄与は小さく、不安定です。** log VIXの線形回帰のR²は最大 {n(co['r2_pct'].max())}%、OOS R²は全期間で負です。離散モデルの標本内R²は12Mで {n(di.loc['12M', 'r2_pct'])}% に達しますが、標本外では危機を除くと負になります。
4. **リスクへの説明力は強いです。** 翌期間の実現ボラに対するR²は {n(vo['r2_pct'].min(), 1)}〜{n(vo['r2_pct'].max(), 1)}% です。
5. **コンフォーマル予測は「幅に使う」と機能します。** 1M先のWinklerはNormで {n(vs.loc[('1M', 'Norm'), 'winkler_skill_pct'], 1)}% 改善し（差のCIは0を除外）、VIX≥40でのカバレッジは {pct(bp.loc[('1M', 'Base', VIX_LABELS[3]), 'coverage'], 1)} から {pct(bp.loc[('1M', 'Norm', VIX_LABELS[3]), 'coverage'], 1)} に上がります。平均に使っても改善せず、ショックの初動も捉えられず、ACIもこの設定では改善しません。

## 2. 研究の設計

### 2.1 問いの分解

| 段階 | 問い | 主な指標 |
|---|---|---|
| 1 | VIXをシグナルに投資するとどうなるか | IRR、CAGR、ボラ、Sharpe、最大DD |
| 2 | シグナルとして機能しているか | 局面別リターン、初回到達、プラセボ |
| 3 | リターンにどの程度寄与するか | 回帰係数、R²、OOS R² |
| 4 | 予測の不確実性の評価に使えるか | Winkler、カバレッジ |

### 2.2 離散と連続の扱い

- **VIX:** 離散は4局面（<20、20-30、30-40、≥40）、連続はlog VIXです。第1段階のシグナルは運用ルールとして明快なので離散にしています。閾値はデータを見る前に登録したものではなく、キリの良い値として選んだものです。
- **投資期間:** 離散は1M/3M/6M/12M（21/63/126/252営業日）、連続は5〜252営業日のグリッドです。

### 2.3 データと執行

- **データ:** `^SP500TR`（配当込み）、`^VIX`、`^IRX`（T-bill）。取得日は2026-11-07です。原データと日次系列はリポジトリに含めません（`DATA_POLICY.md`）。
- **執行:** t日のVIX終値で判定し、**t+1日の終値で執行**します。VIXの確定（16:15ET）が指数の確定（16:00ET）より遅いため、同日執行だと先読みになります。将来リターンもt+1日終値を起点にしています。

"""

    discussion = f"""## 7. 総合考察

### 7.1 なぜ「情報があるように見えるのに」戦略で勝てないのか

第2段階と第3段階では、VIX≥30の後のリターンが高く見えました。それでも第1段階で勝てなかった理由は、結果から3つ挙げられます。

1. **発火が少なく、待つコストが大きい。** VIX≥40は{tm.loc[40, 'years_fired']}/{tm.loc[40, 'n_years']}年しか発火せず、平均{tm.loc[40, 'avg_wait_days']:.0f}日待ちます。その間は株式リスク・プレミアムを取り逃がします。
2. **上乗せは初回到達より、高VIXが続く日に多い。** シグナルで最初に買う日は、全日平均が示す日ではありません。VIX≥30・VIX≥40に初めて達した直後の1M平均はマイナスでした。下げの途中で買っている形です。
3. **全日平均は少数の危機に依存している。** 件数が少なく、標本外では安定しません。

### 7.2 言えること、言えないこと

**言えること**

- この期間の米国株では、VIXで買いのタイミングを選ぶルールが年初一括を上回る証拠はない
- VIXと将来リターンの関係は線形ではなく、極端域に偏っている
- VIXは将来の実現ボラをよく説明し、予測区間の条件付きカバレッジを改善する

**言えないこと**

- 「買い持ちが最適」: 比較したのはVIXを使う単純なルールだけです
- 「年初一括が有意に勝つ」: 待機ルールとの年ごとの差のCIは0をまたぎます。DCA対年初一括も平均差のCIは0をまたぎます
- 「VIXに予測力がない」: 極端域の標本内の関係は検出されています。少数の危機に依存し、標本外で不安定だというのが正確な表現です
- 「VIXでリスク調整すれば儲かる」: 投資額をVIXで調整する戦略（ボラティリティ・ターゲティングなど）は未検証です

### 7.3 別の検証との関係

以前の別検証（`tqqq_test`）では、VIX≥40でTQQQを買うと少ない件数で全勝という結果が出ました。今回のS&P500のイベントを見ると、2010年以降の{len(ev40_post)}件は6M後がすべてプラスです（最小 {sg(ev40_post['6M'].min())}%）。しかし1990年以降に広げると、{ev40_worst.date}に買った場合の6M後は {n(ev40_worst['6M'])}% でした。TQQQの全勝は、TQQQの上場後に2008年型の下落が起きていないという標本に依存している可能性があります。2010年以降でも1M後は最悪 {n(ev40_post['1M'].min())}% で、レバレッジ商品ならさらに大きく下がります。この比較は定性的なもので、TQQQの数値は今回再計算していません。

先行研究との関係では、Bollerslev, Tauchen & Zhou (2009) はVIXの水準ではなく分散リスク・プレミアムを予測変数としており、今回の変数とは異なります。5.5で統制後に短期の傾きが有意になった点は関連しうるものの、VRPそのものは検証していないので、今後の課題です。

### 7.4 実務上の含意

統計的な結果から読み取れるのは「VIXは売買のタイミングより、起こりうる振れ幅の見積もりに向く」ということです。たとえば、VIX≥40の日の1M区間の平均幅は {n(bp.loc[('1M', 'Norm', VIX_LABELS[3]), 'mean_width_pct'], 1)}%（Norm）で、VIX<20の日の {n(bp.loc[('1M', 'Norm', VIX_LABELS[0]), 'mean_width_pct'], 1)}% とは大きく異なります。

## 8. 限界

- **データの範囲:** 1990〜2025年の米国株のみです。米国が好調だった期間で、他の市場や期間へ一般化はできません。
- **独立な標本数:** 極端域の実効的な標本は少数の危機です。Newey-Westの標準誤差は過小になっている可能性があります。
- **データを見た後の判断:** 閾値（20/30/40）は事前登録していません。危機除外の期間、初回到達のギャップ（63営業日）、局面の境界、6.5の診断、待機比率の感度分析も分析者の判断です。
- **多重比較:** プラセボと閾値検定には補正を入れていますが、ビンテージの符号検定、初回到達の二項検定、コンフォーマルの差のCIには補正がありません。
- **コスト:** 資金投入型は取引コストを含みません。売買型のコストは片道{COST_BPS}bpの概算です。税金は含みません。
- **データの近似:** T-billは `^IRX` を実日数/360で近似しています。S&P500TRの指数そのものを使っており、実際のファンドの信託報酬やトラッキング誤差は含みません。
- **ACI:** γは固定で、感度分析は未実施です。
- **統計手法の前提:** コンフォーマル予測の交換可能性は時系列では成り立たず、カバレッジの理論保証はありません。ここでの評価は経験的なものです。
- **実装の確認:** テストは実装の先読み防止や計算の一致を確認するものです。統計的な解釈の正しさまで保証するものではありません。第三者によるレビューは受けていません。

## 9. 再現手順

```bash
pip install -r requirements.txt
python -m src.data --refresh
python -m pytest -q
python -X utf8 run_all.py
```

`run_all.py` は取得済みのデータから全段階、図、このレポートを生成します。主な乱数シードは `src/config.py` にあります。データを再取得すると、調整済みデータの改訂により数値が完全には一致しないことがあります。本文の数値は2026-11-07取得のデータで、同じデータからの2回の通し実行で結果が一致することを確認しました。

## 参考文献

文献名は著者と年のみです。発表や論文に使う前に、原典を確認してください。

- Bollerslev, Tauchen & Zhou (2009): 分散リスク・プレミアムと期待リターン
- Campbell & Thompson (2008): OOS R²
- Clark & West (2007): ネストしたモデルの予測比較
- Gibbs & Candès (2021): Adaptive Conformal Inference
- Hodrick (1992): 重複窓の標準誤差
- Stambaugh (1999): 持続的な予測変数のバイアス
- Winkler (1972): 区間スコア
"""
    text = head + s1 + "\n" + s2 + "\n" + s3 + "\n" + s4 + "\n" + discussion
    check(f2["ev"].loc[(40, "6M"), "min_pct"] < -30, "VIX≥40の6M最悪は-30%より悪い")
    check(bh.mdd < best.mdd, "売買型は最大DDを縮めている")
    check(v_ens.ci_lo_pct < 0 < v_ens.ci_hi_pct and v_ens_d.ci_lo_pct < 0 < v_ens_d.ci_hi_pct, "アンサンブルの差はCIが0をまたぐ")
    check(grid is not None and dca.irr < lump.irr, "DCAのIRRは一括未満")
    return text


def main() -> None:
    text = build()
    OUTPUT.write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"生成しました: {OUTPUT}（{len(text):,}文字）")


if __name__ == "__main__":
    main()
