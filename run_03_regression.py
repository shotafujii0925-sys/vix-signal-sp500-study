# -*- coding: utf-8 -*-
"""第3段階: VIXが将来リターンにどの程度寄与しているかの回帰分析。

設計は VIXの扱い × 投資期間の扱い の 2×2。
    VIX    : 離散（4フェーズのダミー） / 連続（log VIX）
    投資期間: 離散（1M,3M,6M,12M）    / 連続（5〜252営業日のグリッド）

重複した将来リターンと、持続性の高い説明変数を扱うため、推論は1つに頼らず並べる。
    OLS(比較用、過大評価されることを示す) / Newey-West / Hodrick 1B /
    帰無仮説の下でのVARブートストラップ（重複とStambaughバイアスの両方を反映）/ 循環シフト
実用上の寄与は、アウトオブサンプルR²（Campbell-Thompson）とClark-West検定で評価する。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.signal import lfilter

from src.config import (HORIZON_GRID, HORIZON_NAMES, HORIZONS, MIN_TRAIN, N_BOOT, PLACEBO_MIN_SHIFT, PLACEBO_STEP,
                        REFIT_EVERY, RESULTS_DIR, SEED, VIX_LABELS)
from src.data import forward_excess, load
from src.metrics import moving_block_indices


KNOTS = (20, 30, 40)
CRISIS_WINDOWS = (("2008-09-01", "2009-03-31"), ("2020-02-01", "2020-06-30"))


def design_continuous(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"logVIX": np.log(df["VIX"])}, index=df.index)


def design_discrete(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({lab: (df["phase"] == lab).astype(float) for lab in VIX_LABELS[1:]}, index=df.index)


def design_piecewise(df: pd.DataFrame) -> pd.DataFrame:
    """連続のまま非線形を許す折れ線（節点はフェーズ境界の20/30/40に固定）。"""
    x = np.log(df["VIX"])
    out = {"logVIX": x}
    for k in KNOTS:
        out[f"hinge{k}"] = np.maximum(x - np.log(k), 0.0)
    return pd.DataFrame(out, index=df.index)


def fit_nw(y: pd.Series, X: pd.DataFrame, lags: int):
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    return sm.OLS(d["y"], sm.add_constant(d.drop(columns="y"))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def hodrick_1b_t(df: pd.DataFrame, x: pd.Series, h: int) -> float:
    """Hodrick (1992) の 1B: 1期先リターンを、説明変数の過去 h 期合計に回帰したときの t 値（帰無仮説の下で頑健）。"""
    r1 = df["lx"].diff().shift(-2)  # t+2 日の超過リターン（t+1 日引け執行に合わせる）
    xs = x.rolling(h).sum()
    d = pd.concat([r1.rename("r"), xs.rename("x")], axis=1).dropna()
    fit = sm.OLS(d["r"], sm.add_constant(d["x"])).fit(cov_type="HC0")
    return float(fit.tvalues["x"])


def var_null_bootstrap(df: pd.DataFrame, horizons: list[int], n_boot: int, seed: int, block: int = 63) -> dict:
    """「リターンはVIXで予測できない」という帰無仮説の下で傾きの分布を作る。

    log VIX は AR(1)、リターンは定数＋残差とし、(リターン残差, VIX残差) を対にしてブロックで復元抽出する。
    両残差の負の相関と VIX の持続性から生じる上方バイアス（Stambaugh）と、窓の重複が同時に反映される。
    """
    x = np.log(df["VIX"].to_numpy())
    r = df["lx"].diff().to_numpy()[1:]             # r[t] は t→t+1 のリターン
    x0, x1 = x[:-1], x[1:]
    b = np.polyfit(x0, x1, 1)
    u = x1 - np.polyval(b, x0)
    e = r - r.mean()
    n = len(e)
    rng = np.random.default_rng(seed)
    out = {h: [] for h in horizons}
    r2_disc = {h: [] for h in horizons}
    r2_piece = {h: [] for h in horizons}
    log_edges = np.log([20.0, 30.0, 40.0])
    for start in range(0, n_boot, 100):
        k = min(100, n_boot - start)
        idx = moving_block_indices(n, block, k, rng)
        es, us = e[idx], u[idx]
        # x*[t+1] = c + phi x*[t] + u*[t+1]
        zi = np.full((k, 1), b[0] * x[0])
        xs = lfilter([1.0], [1.0, -b[0]], us + b[1], axis=1, zi=zi)[0]
        xs = np.c_[np.full(k, x[0]), xs]
        cum = np.c_[np.zeros(k), np.cumsum(es, axis=1)]
        for h in horizons:
            # y[t] = cum[t+1+h] - cum[t+1], 説明変数は x[t]
            y = cum[:, 1 + h:] - cum[:, 1:-h]
            xx = xs[:, :y.shape[1]]
            xc = xx - xx.mean(axis=1, keepdims=True)
            yc = y - y.mean(axis=1, keepdims=True)
            out[h].append((xc * yc).sum(axis=1) / (xc ** 2).sum(axis=1))
            sst = (yc ** 2).sum(axis=1)
            # フェーズダミーのR2（群間平方和/全平方和）
            ph = np.digitize(xx, log_edges)
            ssb = np.zeros(k)
            for g in range(4):
                mg = ph == g
                cnt = mg.sum(axis=1)
                sg = (yc * mg).sum(axis=1)
                ssb += np.divide(sg ** 2, cnt, out=np.zeros(k), where=cnt > 0)
            r2_disc[h].append(ssb / sst)
            # 折れ線のR2（バッチ正規方程式）
            Z = np.stack([xx] + [np.maximum(xx - e, 0.0) for e in log_edges], axis=2)
            Z = Z - Z.mean(axis=1, keepdims=True)
            ZtZ = np.einsum("kti,ktj->kij", Z, Z) + 1e-9 * np.eye(4)
            Zty = np.einsum("kti,kt->ki", Z, yc)
            coef = np.linalg.solve(ZtZ, Zty[..., None])[..., 0]
            r2_piece[h].append((coef * Zty).sum(axis=1) / sst)
    res = {h: np.concatenate(v) for h, v in out.items()}
    res["r2_disc"] = {h: np.concatenate(v) for h, v in r2_disc.items()}
    res["r2_piece"] = {h: np.concatenate(v) for h, v in r2_piece.items()}
    res["phi"] = float(b[0])
    return res


def r2_shift_pvalue(y: np.ndarray, X: np.ndarray) -> tuple[float, float]:
    """説明変数を循環シフトしたときに、同じ決定係数以上が出る割合。"""
    n = len(y)

    def r2(xm: np.ndarray) -> float:
        ok = np.isfinite(y)
        A = np.c_[np.ones(ok.sum()), xm[ok]]
        beta, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
        res = y[ok] - A @ beta
        return 1 - res.var() / y[ok].var()

    real = r2(X)
    shifts = np.arange(PLACEBO_MIN_SHIFT, n - PLACEBO_MIN_SHIFT, PLACEBO_STEP * 4)
    dist = np.array([r2(np.roll(X, s, axis=0)) for s in shifts])
    return float(real), float((1 + (dist >= real).sum()) / (1 + len(shifts)))


def oos_r2(df: pd.DataFrame, X: pd.DataFrame, h: int) -> dict:
    """拡大窓で係数を推定し、歴史的平均と予測誤差を比べる。

    t 時点の学習に使えるのは y[i] が確定済みの i（i+1+h <= t）だけ。
    """
    y = forward_excess(df, h).to_numpy()
    Xv = X.to_numpy()
    n = len(y)
    pred, base = np.full(n, np.nan), np.full(n, np.nan)
    beta, mu = None, None
    for t in range(MIN_TRAIN, n - h - 1):
        if beta is None or (t - MIN_TRAIN) % REFIT_EVERY == 0:
            end = t - h  # 確定済みは i <= t-1-h → スライス上限 t-h
            A = np.c_[np.ones(end), Xv[:end]]
            beta, *_ = np.linalg.lstsq(A, y[:end], rcond=None)
            mu = y[:end].mean()
        pred[t] = np.r_[1.0, Xv[t]] @ beta
        base[t] = mu
    ok = np.isfinite(pred) & np.isfinite(y)
    e1, e0 = y[ok] - pred[ok], y[ok] - base[ok]
    r2 = 1 - (e1 ** 2).sum() / (e0 ** 2).sum()
    dates = df.index[ok]
    keep = np.ones(ok.sum(), dtype=bool)
    for a, b in CRISIS_WINDOWS:
        keep &= ~((dates >= a) & (dates <= b))
    r2_excl = 1 - (e1[keep] ** 2).sum() / (e0[keep] ** 2).sum()
    early = dates <= "2009-12-31"
    r2_early = 1 - (e1[early] ** 2).sum() / (e0[early] ** 2).sum()
    r2_late = 1 - (e1[~early] ** 2).sum() / (e0[~early] ** 2).sum()
    # Clark-West (2007): 入れ子モデルの比較。片側検定
    f = e0 ** 2 - (e1 ** 2 - (base[ok] - pred[ok]) ** 2)
    fit = sm.OLS(f, np.ones(len(f))).fit(cov_type="HAC", cov_kwds={"maxlags": h})
    return {"oos_r2": float(r2), "oos_r2_excl_crises": float(r2_excl), "oos_r2_to2009": float(r2_early),
            "oos_r2_from2010": float(r2_late), "cw_t": float(fit.tvalues[0]), "cw_p_one_sided": float(fit.pvalues[0] / 2
            if fit.tvalues[0] > 0 else 1 - fit.pvalues[0] / 2), "n_oos": int(ok.sum())}


def realized_vol_forward(df: pd.DataFrame, h: int) -> pd.Series:
    r = np.log1p(df["rm"])
    rv = (r ** 2).rolling(h).sum().shift(-(h + 1))
    return np.log(np.sqrt(rv * 252 / h))


def controls(df: pd.DataFrame) -> pd.DataFrame:
    r = np.log1p(df["rm"])
    out = pd.DataFrame(index=df.index)
    out["past21"] = r.rolling(21).sum()
    out["past63"] = r.rolling(63).sum()
    out["drawdown252"] = df["TR"] / df["TR"].rolling(252, min_periods=21).max() - 1
    out["logRV21"] = np.log(np.sqrt((r ** 2).rolling(21).sum() * 252 / 21))
    return out


def main() -> None:
    df = load()
    RESULTS_DIR.mkdir(exist_ok=True)
    Xc, Xd = design_continuous(df), design_discrete(df)
    sd_logvix = float(Xc["logVIX"].std())

    print("帰無仮説ブートストラップ...")
    null = var_null_bootstrap(df, sorted(set(HORIZONS)), N_BOOT, SEED)

    # ---- 投資期間=離散, VIX=連続 ----
    rows = []
    for h in HORIZONS:
        y = forward_excess(df, h)
        d = pd.concat([y.rename("y"), Xc], axis=1).dropna()
        ols = sm.OLS(d["y"], sm.add_constant(d["logVIX"])).fit()
        nw = fit_nw(y, Xc, h)
        slope = float(nw.params["logVIX"])
        dist = null[h]
        bias = float(dist.mean())
        p_null = float((np.abs(dist - bias) >= abs(slope - bias)).mean())
        p_one = float((dist >= slope).mean())
        r2, shift_p = r2_shift_pvalue(y.to_numpy(), Xc.to_numpy())
        rows.append({
            "horizon": HORIZON_NAMES[h], "h": h, "n": int(nw.nobs), "slope": slope,
            "effect_1sd_pct": slope * sd_logvix * 100, "r2_pct": r2 * 100,
            "p_ols": float(ols.pvalues["logVIX"]), "p_newey_west": float(nw.pvalues["logVIX"]),
            "t_hodrick_1b": hodrick_1b_t(df, Xc["logVIX"], h),
            "stambaugh_bias": bias, "slope_bias_adj": slope - bias,
            "effect_1sd_bias_adj_pct": (slope - bias) * sd_logvix * 100,
            "p_var_null_two_sided": p_null, "p_var_null_one_sided": p_one, "r2_shift_p": shift_p,
            **oos_r2(df, Xc, h),
        })
    cont = pd.DataFrame(rows)
    cont.to_csv(RESULTS_DIR / "s3_continuous.csv", index=False)
    print(cont.round(4).to_string())
    print("log VIX の1SD =", round(sd_logvix, 3), " AR(1)係数 =", round(null["phi"], 4))

    # ---- 投資期間=離散, VIX=離散 ----
    rows = []
    for h in HORIZONS:
        y = forward_excess(df, h)
        nw = fit_nw(y, Xd, h)
        wald = nw.wald_test(np.eye(4)[1:], scalar=True)
        r2, shift_p = r2_shift_pvalue(y.to_numpy(), Xd.to_numpy())
        row = {"horizon": HORIZON_NAMES[h], "h": h, "n": int(nw.nobs),
               "normal_mean_pct": float(nw.params["const"]) * 100,
               "r2_pct": r2 * 100, "wald_p_newey_west": float(wald.pvalue), "r2_shift_p": shift_p,
               **oos_r2(df, Xd, h)}
        for lab in VIX_LABELS[1:]:
            row[f"coef_{lab}_pct"] = float(nw.params[lab]) * 100
            row[f"p_{lab}"] = float(nw.pvalues[lab])
        rows.append(row)
    disc = pd.DataFrame(rows)
    disc.to_csv(RESULTS_DIR / "s3_discrete.csv", index=False)
    print(disc.round(4).to_string())

    # ---- 関数形の比較: 線形 / 折れ線 / フェーズダミー ----
    Xp = design_piecewise(df)
    rows = []
    for h in HORIZONS:
        y = forward_excess(df, h)
        for name, X, dist in [("連続: 線形", Xc, None), ("連続: 折れ線", Xp, null["r2_piece"][h]),
                              ("離散: 4フェーズ", Xd, null["r2_disc"][h])]:
            r2, shift_p = r2_shift_pvalue(y.to_numpy(), X.to_numpy())
            row = {"horizon": HORIZON_NAMES[h], "model": name, "n_params": X.shape[1], "r2_pct": r2 * 100,
                   "r2_shift_p": shift_p, **oos_r2(df, X, h)}
            if dist is None:
                row["null_r2_median_pct"] = np.nan
                row["r2_var_null_p"] = np.nan
            else:
                row["null_r2_median_pct"] = float(np.median(dist)) * 100
                row["r2_var_null_p"] = float((dist >= r2).mean())
            rows.append(row)
    models = pd.DataFrame(rows)
    models.to_csv(RESULTS_DIR / "s3_models.csv", index=False)
    print(models.round(4).to_string())

    # ---- 投資期間=連続 ----
    rows = []
    for h in HORIZON_GRID:
        y = forward_excess(df, h)
        nw_c = fit_nw(y, Xc, h)
        nw_d = fit_nw(y, Xd, h)
        slope, se = float(nw_c.params["logVIX"]), float(nw_c.bse["logVIX"])
        panic = float(nw_d.params[VIX_LABELS[3]])
        panic_se = float(nw_d.bse[VIX_LABELS[3]])
        o_c, o_d = oos_r2(df, Xc, h), oos_r2(df, Xd, h)
        rows.append({"h": h, "slope": slope, "lo": slope - 1.96 * se, "hi": slope + 1.96 * se,
                     "slope_per_year": slope * 252 / h, "r2_cont_pct": float(nw_c.rsquared) * 100,
                     "r2_disc_pct": float(nw_d.rsquared) * 100,
                     "panic_coef_pct": panic * 100, "panic_lo_pct": (panic - 1.96 * panic_se) * 100,
                     "panic_hi_pct": (panic + 1.96 * panic_se) * 100,
                     "oos_r2_cont_pct": o_c["oos_r2"] * 100, "oos_r2_disc_pct": o_d["oos_r2"] * 100,
                     "p_newey_west": float(nw_c.pvalues["logVIX"])})
    grid = pd.DataFrame(rows).drop_duplicates("h")
    grid.to_csv(RESULTS_DIR / "s3_horizon_grid.csv", index=False)
    print(grid.iloc[::5].round(3).to_string())

    # ---- 分散（散らばり）への寄与 ----
    rows = []
    for h in HORIZONS:
        fit = fit_nw(realized_vol_forward(df, h), Xc, h)
        rows.append({"horizon": HORIZON_NAMES[h], "elasticity": float(fit.params["logVIX"]),
                     "p_newey_west": float(fit.pvalues["logVIX"]), "r2_pct": float(fit.rsquared) * 100})
    vol = pd.DataFrame(rows)
    vol.to_csv(RESULTS_DIR / "s3_volatility.csv", index=False)
    print(vol.round(4).to_string())

    # ---- 統制変数を入れたときのVIXの寄与 ----
    C = controls(df)
    rows = []
    for h in HORIZONS:
        y = forward_excess(df, h)
        all_x = pd.concat([Xc, C], axis=1)
        m_full = fit_nw(y, all_x, h)
        d = pd.concat([y.rename("y"), all_x], axis=1).dropna()
        m_ctrl = sm.OLS(d["y"], sm.add_constant(d[C.columns])).fit()
        m_vix = sm.OLS(d["y"], sm.add_constant(d[["logVIX"]])).fit()
        rows.append({"horizon": HORIZON_NAMES[h], "r2_vix_only_pct": float(m_vix.rsquared) * 100,
                     "r2_controls_only_pct": float(m_ctrl.rsquared) * 100, "r2_full_pct": float(m_full.rsquared) * 100,
                     "incremental_r2_vix_pct": float(m_full.rsquared - m_ctrl.rsquared) * 100,
                     "slope_vix_full": float(m_full.params["logVIX"]), "p_vix_full": float(m_full.pvalues["logVIX"]),
                     "slope_logRV21": float(m_full.params["logRV21"]), "p_logRV21": float(m_full.pvalues["logRV21"])})
    ctrl = pd.DataFrame(rows)
    ctrl.to_csv(RESULTS_DIR / "s3_controls.csv", index=False)
    print(ctrl.round(4).to_string())

    # ---- 安定性 ----
    rows = []
    excl = ~(((df.index >= "2008-09-01") & (df.index <= "2009-03-31"))
             | ((df.index >= "2020-02-01") & (df.index <= "2020-06-30")))
    subs = {"全期間": np.ones(len(df), dtype=bool), "1990-2007": df.index <= "2007-12-31",
            "2008-2025": df.index >= "2008-01-01",
            "2008-09〜2009-03と2020-02〜06を除外": excl}
    for name, sub in subs.items():
        for h in HORIZONS:
            fit = fit_nw(forward_excess(df, h)[sub], Xc[sub], h)
            rows.append({"subset": name, "horizon": HORIZON_NAMES[h], "slope": float(fit.params["logVIX"]),
                         "effect_1sd_pct": float(fit.params["logVIX"]) * sd_logvix * 100,
                         "p_newey_west": float(fit.pvalues["logVIX"]), "r2_pct": float(fit.rsquared) * 100,
                         "n": int(fit.nobs)})
    stab = pd.DataFrame(rows)
    stab.to_csv(RESULTS_DIR / "s3_stability.csv", index=False)
    print(stab.round(4).to_string())
    pd.Series({"sd_logvix": sd_logvix, "phi": null["phi"]}).to_json(RESULTS_DIR / "s3_meta.json")


if __name__ == "__main__":
    main()
