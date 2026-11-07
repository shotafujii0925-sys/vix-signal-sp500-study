# -*- coding: utf-8 -*-
"""第4段階: VIXを使うと将来リターンの予測区間はどの程度良くなるか。

全手法・全ホライズンで、区間の発行期間を揃えて評価する。
評価の中心はカバレッジではなく Winkler score（幅と外れの両方を罰する）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import conformal as C
from src.config import ALPHA, HORIZON_NAMES, HORIZONS, N_BOOT, RESULTS_DIR, SEED, TEST_START, VIX_LABELS
from src.data import forward_excess, load
from src.metrics import block_ci_mean_diff
from run_02_signal import first_cross_events

GRID = [5, 10, 21, 42, 63, 126, 189, 252]
GRID_METHODS = ["Base", "Reg", "RegD", "Norm", "Mondrian"]
MAX_H = max(HORIZONS)


def evaluate(y, lo, hi, idx) -> dict:
    yy, ll, hh = y[idx], lo[idx], hi[idx]
    cov = (yy >= ll) & (yy <= hh)
    return {"n": int(len(idx)), "coverage": float(cov.mean()), "miss_low": float((yy < ll).mean()),
            "miss_high": float((yy > hh).mean()), "mean_width_pct": float((hh - ll).mean() * 100),
            "median_width_pct": float(np.median(hh - ll) * 100),
            "winkler_pct": float(C.winkler(yy, ll, hh).mean() * 100)}


def main() -> None:
    df = load()
    RESULTS_DIR.mkdir(exist_ok=True)
    x = np.log(df["VIX"].to_numpy())
    phase = df["phase"].cat.codes.to_numpy()
    n = len(df)
    t_start = int(df.index.searchsorted(pd.Timestamp(TEST_START)))
    t_end = n - 1 - MAX_H   # 12Mの結果まで確定している最後の発行日（全ホライズン共通）
    idx = np.arange(t_start, t_end)
    dates = df.index
    print(f"評価期間（区間の発行日）: {dates[t_start].date()} 〜 {dates[t_end - 1].date()}  N={len(idx)}")
    ev40 = first_cross_events(df["VIX"].to_numpy(), 40)
    ev40 = ev40[(ev40 >= t_start) & (ev40 < t_end)]

    overall, by_phase, diffs, yearly, post, pre, bands = [], [], [], [], [], [], {}
    for h in HORIZONS:
        name = HORIZON_NAMES[h]
        y = forward_excess(df, h).to_numpy()
        fc = C.walk_forward(y, x, phase, h)
        block = max(21, 2 * h)
        results = {}
        for method in C.METHODS:
            res = C.intervals(method, y, phase, fc, h, t_start, t_end)
            results[method] = res
            row = {"horizon": name, "method": method, **evaluate(y, res["lo"], res["hi"], idx)}
            cov_series = ((y[idx] >= res["lo"][idx]) & (y[idx] <= res["hi"][idx])).astype(float)
            ci = block_ci_mean_diff(cov_series - (1 - ALPHA), block, N_BOOT, SEED + h)
            row.update(coverage_gap_lo=ci["lo"], coverage_gap_hi=ci["hi"], effective_n=len(idx) / h)
            yr = pd.Series(cov_series, index=dates[idx]).groupby(dates[idx].year).mean()
            row.update(worst_year_cov=float(yr.min()), worst_year=int(yr.idxmin()),
                       share_years_80_100=float(((yr >= 0.8)).mean()))
            overall.append(row)
            yearly.append(yr.rename(f"{name}/{method}"))
            for g, lab in enumerate(VIX_LABELS):
                sub = idx[phase[idx] == g]
                by_phase.append({"horizon": name, "method": method, "phase": lab,
                                 **evaluate(y, res["lo"], res["hi"], sub)})
            post_idx = np.unique(np.concatenate([np.arange(e, min(e + 21, t_end)) for e in ev40]))
            post.append({"horizon": name, "method": method, "n_events": len(ev40),
                         **evaluate(y, res["lo"], res["hi"], post_idx)})
            # 図6を見た後に追加した診断: VIXが跳ねる前の21営業日に出した区間（ショックの初動を捉えられるか）
            pre_idx = np.unique(np.concatenate([np.arange(max(e - 21, t_start), e) for e in ev40]))
            pre.append({"horizon": name, "method": method, "n_events": len(ev40),
                        "mean_vix_at_issue": float(df["VIX"].to_numpy()[pre_idx].mean()),
                        **evaluate(y, res["lo"], res["hi"], pre_idx)})
        base_w = C.winkler(y[idx], results["Base"]["lo"][idx], results["Base"]["hi"][idx])
        base_cov = (y[idx] >= results["Base"]["lo"][idx]) & (y[idx] <= results["Base"]["hi"][idx])
        for method in C.METHODS:
            if method == "Base":
                continue
            res = results[method]
            w = C.winkler(y[idx], res["lo"][idx], res["hi"][idx])
            cov = (y[idx] >= res["lo"][idx]) & (y[idx] <= res["hi"][idx])
            dw = block_ci_mean_diff(w - base_w, block, N_BOOT, SEED + h)
            dc = block_ci_mean_diff(cov.astype(float) - base_cov.astype(float), block, N_BOOT, SEED + h)
            diffs.append({"horizon": name, "method": method, "winkler_diff_pct": dw["point"] * 100,
                          "winkler_lo_pct": dw["lo"] * 100, "winkler_hi_pct": dw["hi"] * 100,
                          "winkler_skill_pct": (1 - w.mean() / base_w.mean()) * 100,
                          "winkler_clear": dw["excludes_zero"], "coverage_diff_pt": dc["point"] * 100,
                          "coverage_lo_pt": dc["lo"] * 100, "coverage_hi_pt": dc["hi"] * 100,
                          "coverage_clear": dc["excludes_zero"], "block": block, "n_boot": N_BOOT})
        if h in (21, 63):
            bands[h] = pd.DataFrame({"y": y, "vix": df["VIX"].to_numpy(),
                                     **{f"{m}_lo": results[m]["lo"] for m in ("Base", "Norm", "ACI-Norm")},
                                     **{f"{m}_hi": results[m]["hi"] for m in ("Base", "Norm", "ACI-Norm")}},
                                    index=dates).iloc[idx]
        print(f"h={h} 完了")

    ov = pd.DataFrame(overall)
    ov.to_csv(RESULTS_DIR / "s4_overall.csv", index=False)
    print(ov.round(3).to_string())
    bp = pd.DataFrame(by_phase)
    bp.to_csv(RESULTS_DIR / "s4_by_phase.csv", index=False)
    dd = pd.DataFrame(diffs)
    dd.to_csv(RESULTS_DIR / "s4_vs_base.csv", index=False)
    print(dd.round(3).to_string())
    pd.DataFrame(post).to_csv(RESULTS_DIR / "s4_post_panic.csv", index=False)
    pd.DataFrame(pre).to_csv(RESULTS_DIR / "s4_pre_panic.csv", index=False)
    pd.concat(yearly, axis=1).to_csv(RESULTS_DIR / "s4_yearly_coverage.csv")
    for h, b in bands.items():
        b.to_csv(RESULTS_DIR / f"s4_bands_h{h}.csv")

    # 投資期間を連続に近いグリッドで見る
    rows = []
    for h in GRID:
        y = forward_excess(df, h).to_numpy()
        fc = C.walk_forward(y, x, phase, h)
        base = None
        for method in GRID_METHODS:
            res = C.intervals(method, y, phase, fc, h, t_start, t_end)
            e = evaluate(y, res["lo"], res["hi"], idx)
            base = e["winkler_pct"] if method == "Base" else base
            rows.append({"h": h, "method": method, **e, "winkler_skill_pct": (1 - e["winkler_pct"] / base) * 100})
    gr = pd.DataFrame(rows)
    gr.to_csv(RESULTS_DIR / "s4_grid.csv", index=False)
    print(gr.pivot(index="h", columns="method", values="winkler_skill_pct").round(2).to_string())
    pd.Series({"test_start": str(dates[t_start].date()), "test_end": str(dates[t_end - 1].date()),
               "n_test": len(idx), "n_panic_events": len(ev40)}).to_json(RESULTS_DIR / "s4_meta.json")


if __name__ == "__main__":
    main()
