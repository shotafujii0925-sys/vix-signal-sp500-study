# -*- coding: utf-8 -*-
"""第2段階: VIXはシグナルとしてどの程度機能しているか。

戦略の成績ではなく、シグナル自体の情報量を見る。
- VIXフェーズ別の将来超過リターンと、全期間平均との差（ブロックブートストラップCI・循環シフト検定）
- 独立イベント（しばらくぶりに閾値を超えた日）だけに絞った集計
- 前後半と危機除外での安定性
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.config import (HORIZON_NAMES, HORIZONS, N_BOOT, PLACEBO_MIN_SHIFT, PLACEBO_STEP, RESULTS_DIR, SEED,
                        THRESHOLDS, VIX_LABELS)
from src.data import forward_excess, load
from src.metrics import block_ci_mean_diff

EVENT_GAP = 63  # この営業日数以上シグナルが途切れたら別イベントとする
CRISES = {"2008-09〜2009-03": ("2008-09-01", "2009-03-31"), "2020-02〜2020-06": ("2020-02-01", "2020-06-30")}


def first_cross_events(vix: np.ndarray, th: float, gap: int = EVENT_GAP) -> np.ndarray:
    """閾値以上の日のうち、直前の閾値以上の日から gap 営業日超離れている日。"""
    above = np.flatnonzero(vix >= th)
    if len(above) == 0:
        return above
    keep = np.r_[True, np.diff(above) > gap]
    return above[keep]


def shift_pvalue(y: np.ndarray, mask: np.ndarray) -> float:
    """シグナル系列を循環シフトした場合に、同じ平均差以上が出る割合（片側）。"""
    ok = np.isfinite(y)
    base = y[ok].mean()
    real = y[ok & mask].mean() - base
    n = len(y)
    shifts = np.arange(PLACEBO_MIN_SHIFT, n - PLACEBO_MIN_SHIFT, PLACEBO_STEP)
    dist = np.array([y[ok & np.roll(mask, s)].mean() - base for s in shifts])
    return float((1 + (dist >= real).sum()) / (1 + len(shifts)))


def phase_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        y = forward_excess(df, h).to_numpy()
        ok = np.isfinite(y)
        block = max(21, 2 * h)
        base = y[ok].mean()
        for lab in VIX_LABELS:
            mask = (df["phase"] == lab).to_numpy()
            s = y[mask & ok]
            ci = block_ci_mean_diff(np.where(mask & ok, y - base, 0.0)[ok] * ok.sum() / len(s), block,
                                    N_BOOT, SEED + h)
            rows.append({"horizon": HORIZON_NAMES[h], "h": h, "phase": lab, "n_days": len(s),
                         "mean_pct": s.mean() * 100, "median_pct": np.median(s) * 100,
                         "std_pct": s.std(ddof=1) * 100, "p05_pct": np.quantile(s, .05) * 100,
                         "min_pct": s.min() * 100, "win_rate": (s > 0).mean(),
                         "diff_vs_all_pct": (s.mean() - base) * 100,
                         "diff_lo_pct": ci["lo"] * 100, "diff_hi_pct": ci["hi"] * 100,
                         "diff_excludes_zero": ci["excludes_zero"],
                         "shift_p": shift_pvalue(y, mask), "all_mean_pct": base * 100,
                         "all_win_rate": (y[ok] > 0).mean(), "all_std_pct": y[ok].std(ddof=1) * 100})
    return pd.DataFrame(rows)


def threshold_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        y = forward_excess(df, h).to_numpy()
        for th in THRESHOLDS:
            mask = (df["VIX"] >= th).to_numpy()
            ok = np.isfinite(y)
            rows.append({"threshold": th, "horizon": HORIZON_NAMES[h], "n_days": int((mask & ok).sum()),
                         "diff_vs_all_pct": (y[mask & ok].mean() - y[ok].mean()) * 100,
                         "shift_p": shift_pvalue(y, mask)})
    out = pd.DataFrame(rows)
    # 12本の検定の多重性: Holm法で補正した値を併記する
    order = out["shift_p"].sort_values().index
    m = len(out)
    adj, running = {}, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * out.loc[i, "shift_p"]))
        adj[i] = running
    out["shift_p_holm"] = pd.Series(adj)
    return out


def event_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    vix = df["VIX"].to_numpy()
    summary, detail = [], []
    for th in THRESHOLDS:
        ev = first_cross_events(vix, th)
        for h in HORIZONS:
            y = forward_excess(df, h).to_numpy()
            ok = np.isfinite(y)
            v = y[ev]
            v = v[np.isfinite(v)]
            base_win = (y[ok] > 0).mean()
            wins = int((v > 0).sum())
            summary.append({"threshold": th, "horizon": HORIZON_NAMES[h], "n_events": len(v),
                            "mean_pct": v.mean() * 100, "median_pct": np.median(v) * 100,
                            "min_pct": v.min() * 100, "max_pct": v.max() * 100,
                            "win_rate": wins / len(v), "all_mean_pct": y[ok].mean() * 100,
                            "all_win_rate": base_win,
                            "binom_p_vs_all": stats.binomtest(wins, len(v), base_win).pvalue})
        for i in ev:
            row = {"threshold": th, "date": str(df.index[i].date()), "vix": vix[i]}
            for h in HORIZONS:
                row[HORIZON_NAMES[h]] = forward_excess(df, h).iloc[i] * 100
            detail.append(row)
    return pd.DataFrame(summary), pd.DataFrame(detail)


def stability_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subsets = {"全期間": np.ones(len(df), dtype=bool),
               "1990-2007": (df.index <= "2007-12-31"), "2008-2025": (df.index >= "2008-01-01")}
    excl = np.ones(len(df), dtype=bool)
    for _, (a, b) in CRISES.items():
        excl &= ~((df.index >= a) & (df.index <= b))
    subsets["2008-09〜2009-03と2020-02〜06を除外"] = excl
    for name, sub in subsets.items():
        for h in HORIZONS:
            y = forward_excess(df, h).to_numpy()
            ok = np.isfinite(y) & sub
            for th in THRESHOLDS:
                mask = (df["VIX"] >= th).to_numpy() & ok
                rows.append({"subset": name, "horizon": HORIZON_NAMES[h], "threshold": th, "n_days": int(mask.sum()),
                             "diff_vs_subset_all_pct": (y[mask].mean() - y[ok].mean()) * 100 if mask.any() else np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    df = load()
    RESULTS_DIR.mkdir(exist_ok=True)
    pt = phase_table(df)
    pt.to_csv(RESULTS_DIR / "s2_phase.csv", index=False)
    print(pt.drop(columns=["h", "all_win_rate"]).round(2).to_string())
    tt = threshold_table(df)
    tt.to_csv(RESULTS_DIR / "s2_threshold.csv", index=False)
    print("\n", tt.round(3).to_string())
    es, ed = event_table(df)
    es.to_csv(RESULTS_DIR / "s2_events_summary.csv", index=False)
    ed.to_csv(RESULTS_DIR / "s2_events_detail.csv", index=False)
    print("\n", es.round(3).to_string())
    print("\n", ed[ed.threshold == 40].round(1).to_string())
    st = stability_table(df)
    st.to_csv(RESULTS_DIR / "s2_stability.csv", index=False)
    print("\n", st.pivot_table(index=["subset", "threshold"], columns="horizon", values="diff_vs_subset_all_pct",
                               sort=False)[["1M", "3M", "6M", "12M"]].round(1).to_string())


if __name__ == "__main__":
    main()
