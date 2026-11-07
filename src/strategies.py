# -*- coding: utf-8 -*-
"""投資戦略のシミュレーション。

資金投入型（積立の問題）
    毎年、各年最初の営業日に同額(1)の資金が入る。全戦略で外部キャッシュフローが同一なので、
    期末評価額・IRRをそのまま比較できる。未投資の資金はT-billで運用する。
    各「トランシェ」は (到着日, 投資日, 金額) で表す。

売買型（タイミングの問題）
    初期資金1を全額T-billまたは株式に置く。シグナル後N日保有。常時保有と比較する。

執行規則（先読み防止）
    シグナルは t 日のVIX終値で判定し、t+1 日の終値で執行する。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import annual_vol, cagr_from_returns, irr, max_drawdown, sharpe


@dataclass
class Tranches:
    arrive: np.ndarray   # 到着インデックス
    deploy: np.ndarray   # 投資インデックス（n 以上なら期末まで未投資）
    amount: np.ndarray

    def __add__(self, other: "Tranches") -> "Tranches":
        return Tranches(np.concatenate([self.arrive, other.arrive]),
                        np.concatenate([self.deploy, other.deploy]),
                        np.concatenate([self.amount, other.amount]))

    def scaled(self, w: float) -> "Tranches":
        return Tranches(self.arrive, self.deploy, self.amount * w)


class Market:
    """シミュレーションに使う配列の入れ物。"""

    def __init__(self, df: pd.DataFrame):
        self.index = df.index
        self.n = len(df)
        self.tr = df["TR"].to_numpy(float)
        self.c = df["C"].to_numpy(float)
        self.rm = df["rm"].to_numpy(float)
        self.rf = df["rf"].to_numpy(float)
        self.vix = df["VIX"].to_numpy(float)
        years = df.index.year.to_numpy()
        months = df.index.to_period("M").asi8
        self.year_first = np.flatnonzero(np.r_[True, years[1:] != years[:-1]])
        self.year_last = np.r_[self.year_first[1:] - 1, self.n - 1]
        self.month_first = np.flatnonzero(np.r_[True, months[1:] != months[:-1]])
        self.years = years
        self.t_years = np.asarray((df.index - df.index[0]).days, dtype=float) / 365.25
        self.span_years = float(self.t_years[-1])


def lump_sum(m: Market) -> Tranches:
    a = m.year_first
    return Tranches(a.copy(), a.copy(), np.ones(len(a)))


def dca(m: Market) -> Tranches:
    """年初に届く資金を12等分し、各月最初の営業日に投資する。"""
    d = m.month_first
    ypos = np.searchsorted(m.year_first, d, side="right") - 1
    a = m.year_first[ypos]
    amounts = 1.0 / np.bincount(ypos)[ypos]
    return Tranches(a, d, amounts)


def next_execution(signal: np.ndarray) -> np.ndarray:
    """各 i について、シグナル翌日の執行日のうち i 以上で最小のもの。なければ n。"""
    n = len(signal)
    exec_days = np.flatnonzero(signal[:-1]) + 1
    pos = np.searchsorted(exec_days, np.arange(n), side="left")
    out = np.full(n, n, dtype=int)
    valid = pos < len(exec_days)
    out[valid] = exec_days[pos[valid]]
    return out


def signal_wait(m: Market, signal: np.ndarray, policy: str) -> Tranches:
    """シグナルまで資金を待機させて全額投資する。

    policy="E": その年の最終営業日までに出なければ強制投資
    policy="R": 出るまで繰り越す（期末まで出なければ現金のまま）
    """
    if policy not in ("E", "R"):
        raise ValueError(policy)
    nxt = next_execution(signal)
    a = m.year_first
    d = nxt[a].copy()
    if policy == "E":
        d = np.minimum(d, m.year_last)
    return Tranches(a.copy(), d, np.ones(len(a)))


def simulate(m: Market, tr: Tranches) -> dict:
    """トランシェから日次の保有・評価額・時間加重リターンを計算する（取引は引け）。"""
    n = m.n
    d_units = np.zeros(n)
    d_shares = np.zeros(n)
    np.add.at(d_units, tr.arrive, tr.amount / m.c[tr.arrive])
    done = tr.deploy < n
    dep = tr.deploy[done]
    arr = tr.arrive[done]
    amt = tr.amount[done]
    np.add.at(d_units, dep, -amt / m.c[arr])
    np.add.at(d_shares, dep, amt * (m.c[dep] / m.c[arr]) / m.tr[dep])
    units = np.cumsum(d_units)
    shares = np.cumsum(d_shares)
    equity = shares * m.tr
    cash = units * m.c
    wealth = equity + cash
    # 当日のフロー前評価額 ÷ 前日のフロー後評価額 − 1 = 時間加重の日次リターン
    pre = np.r_[wealth[0], shares[:-1] * m.tr[1:] + units[:-1] * m.c[1:]]
    prev = np.r_[wealth[0], wealth[:-1]]
    twr = np.divide(pre, prev, out=np.ones(n), where=prev > 0) - 1.0
    return {"wealth": wealth, "equity": equity, "cash": cash, "twr": twr,
            "weight": np.divide(equity, wealth, out=np.zeros(n), where=wealth > 0)}


def budget_metrics(m: Market, tr: Tranches, label: str) -> dict:
    sim = simulate(m, tr)
    twr = sim["twr"][1:]
    rf = m.rf[1:]
    arrivals = pd.Series(tr.amount).groupby(tr.arrive).sum()
    flow_t = m.t_years[arrivals.index.to_numpy()]
    flows = arrivals.to_numpy()
    w_end = sim["wealth"][-1]
    wealth_dd = (sim["wealth"] / np.maximum.accumulate(sim["wealth"]) - 1.0).min()
    return {
        "strategy": label,
        "terminal_multiple": float(w_end / flows.sum()),
        "irr": irr(flow_t, flows, m.span_years, w_end),
        "twr_cagr": cagr_from_returns(twr, m.span_years),
        "vol": annual_vol(twr),
        "sharpe": sharpe(twr, rf),
        "mdd": max_drawdown(twr),
        "wealth_mdd": float(wealth_dd),
        "avg_equity_weight": float(sim["weight"].mean()),
        "final_cash_share": float(sim["cash"][-1] / w_end),
    }


def vintage_values(m: Market, tr: Tranches) -> np.ndarray:
    """各年の資金1単位が、その年の最終営業日にいくらになったか（年内投資が前提）。

    年末時点で全戦略が全額投資済みなら、これ以降の成長率は共通なので、
    年末時点の対数差がそのまま「買付タイミングの差」になる。
    """
    ypos = np.searchsorted(m.year_first, tr.arrive, side="right") - 1
    end = m.year_last[ypos]
    if (tr.deploy > end).any():
        raise ValueError("年内に投資されないトランシェがあります（policy=E のみ対応）")
    value = tr.amount * (m.c[tr.deploy] / m.c[tr.arrive]) * (m.tr[end] / m.tr[tr.deploy])
    return np.bincount(ypos, weights=value, minlength=len(m.year_first))


def tactical_position(signal: np.ndarray, hold: int) -> np.ndarray:
    """各日のリターンに対する保有フラグ。

    t 日引けのシグナル → t+1 日引けで買付 → t+2 日のリターンから hold 日分を受け取る。
    保有中の再シグナルは保有を延長する。
    """
    sig = pd.Series(signal.astype(float))
    return (sig.shift(2).rolling(hold, min_periods=1).max().fillna(0.0).to_numpy()) > 0.5


def tactical_returns(m: Market, pos: np.ndarray, cost_bps: float = 0.0) -> np.ndarray:
    r = np.where(pos, m.rm, m.rf)
    if cost_bps:
        change = np.abs(np.diff(pos.astype(float), prepend=0.0))
        r = r - change * cost_bps / 1e4
    return r


def tactical_metrics(m: Market, r: np.ndarray, label: str, pos: np.ndarray | None = None) -> dict:
    r1, rf = r[1:], m.rf[1:]
    out = {"strategy": label, "cagr": cagr_from_returns(r1, m.span_years), "vol": annual_vol(r1),
           "sharpe": sharpe(r1, rf), "mdd": max_drawdown(r1)}
    if pos is not None:
        out["exposure"] = float(pos.mean())
        out["round_trips_per_year"] = float((np.diff(pos.astype(int)) == 1).sum() / m.span_years)
    return out
