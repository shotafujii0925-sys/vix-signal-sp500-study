# -*- coding: utf-8 -*-
"""評価指標。すべて同じ期間の日次系列に対して計算する。"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import brentq

TRADING_DAYS = 252


def cagr_from_returns(r: np.ndarray, years: float) -> float:
    return float(np.prod(1.0 + r) ** (1.0 / years) - 1.0)


def annual_vol(r: np.ndarray) -> float:
    return float(np.std(r, ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe(r: np.ndarray, rf: np.ndarray) -> float:
    """日次超過リターンの算術平均/標準偏差を年率化。リスクフリーは各日のT-bill。"""
    x = r - rf
    s = np.std(x, ddof=1)
    return float(np.mean(x) / s * np.sqrt(TRADING_DAYS)) if s > 0 else float("nan")


def max_drawdown(r: np.ndarray) -> float:
    idx = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(np.concatenate(([1.0], idx)))[1:]
    return float((idx / peak - 1.0).min())


def irr(flow_years: np.ndarray, flows: np.ndarray, terminal_year: float, terminal: float) -> float:
    """資金加重収益率(XIRR)。flows は投入額(正)、terminal は期末評価額。"""
    def f(rate: float) -> float:
        return terminal - float(np.sum(flows * (1.0 + rate) ** (terminal_year - flow_years)))
    return float(brentq(f, -0.9, 1.0))


def moving_block_indices(n: int, block: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """moving block bootstrap の添字行列 (n_boot, n)。"""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = starts[:, :, None] + np.arange(block)[None, None, :]
    return idx.reshape(n_boot, -1)[:, :n]


def block_ci_mean_diff(d: np.ndarray, block: int, n_boot: int, seed: int) -> dict:
    """平均差の95%信頼区間（moving block bootstrap）。"""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for start in range(0, n_boot, 200):
        k = min(200, n_boot - start)
        means[start:start + k] = d[moving_block_indices(len(d), block, k, rng)].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"point": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "block": int(block), "n_boot": int(n_boot), "n": int(len(d)),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def block_ci_sharpe_diff(ra: np.ndarray, rb: np.ndarray, rf: np.ndarray, block: int,
                         n_boot: int, seed: int) -> dict:
    """Sharpe差 (a-b) の95%信頼区間。"""
    xa, xb = ra - rf, rb - rf
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for start in range(0, n_boot, 200):
        k = min(200, n_boot - start)
        idx = moving_block_indices(len(xa), block, k, rng)
        sa = xa[idx].mean(axis=1) / xa[idx].std(axis=1, ddof=1)
        sb = xb[idx].mean(axis=1) / xb[idx].std(axis=1, ddof=1)
        out[start:start + k] = (sa - sb) * np.sqrt(TRADING_DAYS)
    lo, hi = np.quantile(out, [0.025, 0.975])
    point = (xa.mean() / xa.std(ddof=1) - xb.mean() / xb.std(ddof=1)) * np.sqrt(TRADING_DAYS)
    return {"point": float(point), "lo": float(lo), "hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0), "block": int(block), "n_boot": int(n_boot)}


def sign_test(d: np.ndarray, tol: float = 1e-12) -> dict:
    """符号検定。差が0の標本（同値）は勝ちにも負けにも数えず、検定から除く。"""
    d = np.asarray(d, dtype=float)
    wins = int((d > tol).sum())
    losses = int((d < -tol).sum())
    decided = wins + losses
    p = float(stats.binomtest(wins, decided, 0.5).pvalue) if decided else float("nan")
    return {"wins": wins, "losses": losses, "ties": int(len(d) - decided), "p": p}
