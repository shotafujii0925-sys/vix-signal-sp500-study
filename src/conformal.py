# -*- coding: utf-8 -*-
"""将来リターンに対するコンフォーマル予測。

VIXをどこに使うかで手法を分ける（離散/連続 × 平均/散らばり）。

    Base    : 平均=過去平均          散らばり=一定          VIXを使わない基準
    Reg     : 平均=log VIXの回帰      散らばり=一定          VIX(連続)を平均に使う
    RegD    : 平均=フェーズ別の回帰   散らばり=一定          VIX(離散)を平均に使う
    Norm    : 平均=過去平均           散らばり=VIXの関数     VIX(連続)を散らばりに使う
    Mondrian: 平均=過去平均           散らばり=フェーズ別分位  VIX(離散)を散らばりに使う
    Asym    : Normと同じだが上下で別の分位を使う（負の歪みへの対応）
    ACI-Base / ACI-Norm : 同じ土台に適応αを載せる

情報の扱い（先読み防止）
    y[i] は i+1+h 日の引けに確定する。t 日に区間を出すとき使えるのは i <= t-1-h の y だけ。
    回帰の再推定にも校正にも ACI の更新にも、この同じ制約をかける。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ALPHA, CAL_WINDOW, GAMMA, MIN_CAL, MIN_TRAIN, MONDRIAN_MIN, REFIT_EVERY

ALPHA_MIN, ALPHA_MAX = 0.001, 0.5
METHODS = ["Base", "Reg", "RegD", "Norm", "Mondrian", "Asym", "ACI-Base", "ACI-Norm"]


def winkler(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    return (hi - lo) + (2 / alpha) * np.clip(lo - y, 0, None) + (2 / alpha) * np.clip(y - hi, 0, None)


def upper_quantile(scores: np.ndarray, alpha: float) -> float:
    """有限標本補正つき上側分位 ceil((n+1)(1-alpha))/n。"""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float(scores.max())
    return float(np.partition(scores, k - 1)[k - 1])


def lower_quantile(scores: np.ndarray, alpha: float) -> float:
    """有限標本補正つき下側分位 floor((n+1)alpha)/n。"""
    n = len(scores)
    k = int(np.floor((n + 1) * alpha))
    if k < 1:
        return float(scores.min())
    return float(np.partition(scores, k - 1)[k - 1])


@dataclass
class Forecasts:
    m_base: np.ndarray
    m_reg: np.ndarray
    m_regd: np.ndarray
    s_vix: np.ndarray


def walk_forward(y: np.ndarray, x: np.ndarray, phase: np.ndarray, h: int, start: int = MIN_TRAIN,
                 refit: int = REFIT_EVERY) -> Forecasts:
    """各 t で確定済みの y だけを使って平均と散らばりの予測を作る。"""
    n = len(y)
    m_base, m_reg, m_regd, s_vix = (np.full(n, np.nan) for _ in range(4))
    last = n - 1
    for t in range(start, last + 1):
        if (t - start) % refit == 0:
            end = t - h   # 確定済み: i <= t-1-h
            yt, xt, pt = y[:end], x[:end], phase[:end]
            ok = np.isfinite(yt)
            yt, xt, pt = yt[ok], xt[ok], pt[ok]
            mu = yt.mean()
            b = np.polyfit(xt, yt, 1)
            phase_mean = np.array([yt[pt == g].mean() if (pt == g).any() else mu for g in range(4)])
            c = np.polyfit(xt, np.log(np.abs(yt - mu) + 1e-4), 1)
        m_base[t] = mu
        m_reg[t] = np.polyval(b, x[t])
        m_regd[t] = phase_mean[phase[t]]
        s_vix[t] = np.exp(np.polyval(c, x[t]))
    return Forecasts(m_base, m_reg, m_regd, s_vix)


def _window(t: int, h: int, first: int) -> tuple[int, int]:
    """校正に使える予測の範囲 [a, b)。b は確定済みの上限。"""
    b = t - h
    a = max(first, b - CAL_WINDOW)
    return a, b


def intervals(method: str, y: np.ndarray, phase: np.ndarray, fc: Forecasts, h: int, t_start: int,
              t_end: int, alpha: float = ALPHA, gamma: float = GAMMA, first: int = MIN_TRAIN) -> dict:
    """t_start..t_end-1 の各日に出す区間を返す。"""
    n = len(y)
    lo, hi = np.full(n, np.nan), np.full(n, np.nan)
    if method in ("Base", "ACI-Base", "Mondrian"):
        m, s = fc.m_base, np.ones(n)
    elif method == "Reg":
        m, s = fc.m_reg, np.ones(n)
    elif method == "RegD":
        m, s = fc.m_regd, np.ones(n)
    elif method in ("Norm", "Asym", "ACI-Norm"):
        m, s = fc.m_base, fc.s_vix
    else:
        raise ValueError(method)
    signed = (y - m) / s
    scores = np.abs(signed)
    adaptive = method.startswith("ACI")
    alpha_t = alpha
    alpha_path = np.full(n, np.nan)
    for t in range(t_start, t_end):
        if adaptive:
            i = t - 1 - h  # t の引けで結果が確定する、h+1 日前に発行した区間
            if i >= t_start:
                err = 0.0 if lo[i] <= y[i] <= hi[i] else 1.0
                alpha_t = float(np.clip(alpha_t + gamma * (alpha - err), ALPHA_MIN, ALPHA_MAX))
            alpha_path[t] = alpha_t
        a, b = _window(t, h, first)
        if b - a < MIN_CAL:
            raise ValueError(f"校正データ不足: t={t}, n={b - a}")
        if method == "Asym":
            w = signed[a:b]
            lo[t] = m[t] + lower_quantile(w, alpha / 2) * s[t]
            hi[t] = m[t] + upper_quantile(w, alpha / 2) * s[t]
            continue
        if method == "Mondrian":
            w = _mondrian_scores(scores[a:b], phase[a:b], phase[t])
            q = upper_quantile(w, alpha)
        else:
            q = upper_quantile(scores[a:b], alpha_t if adaptive else alpha)
        lo[t], hi[t] = m[t] - q * s[t], m[t] + q * s[t]
    return {"lo": lo, "hi": hi, "alpha_t": alpha_path}


def _mondrian_scores(scores: np.ndarray, phases: np.ndarray, current: int) -> np.ndarray:
    """現在のフェーズのスコアを使う。件数不足なら隣接する低いフェーズを順に足す。"""
    g = current
    mask = phases == g
    while mask.sum() < MONDRIAN_MIN and g > 0:
        g -= 1
        mask |= phases == g
    return scores[mask]
