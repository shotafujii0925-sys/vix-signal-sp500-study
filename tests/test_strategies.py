# -*- coding: utf-8 -*-
"""戦略エンジンの検証。合成データのみで、外部接続なし。"""
import numpy as np
import pandas as pd
import pytest

from src import metrics as M
from src import strategies as S


def make_market(n_years=3, seed=0):
    idx = pd.bdate_range("2000-01-03", periods=260 * n_years)
    rng = np.random.default_rng(seed)
    rm = rng.normal(0.0004, 0.01, len(idx))
    rm[0] = 0.0
    rf = np.full(len(idx), 0.0001)
    rf[0] = 0.0
    df = pd.DataFrame({"TR": 100 * np.cumprod(1 + rm), "C": np.cumprod(1 + rf), "rm": rm, "rf": rf,
                       "VIX": 15 + 10 * rng.random(len(idx))}, index=idx)
    return S.Market(df)


def brute_force_wealth(m, tr):
    """日次ループでの素朴な実装。ベクトル化版の答え合わせに使う。"""
    shares, cash, out = 0.0, 0.0, []
    tr_list = list(zip(tr.arrive, tr.deploy, tr.amount))
    pending = []
    for t in range(m.n):
        if t > 0:
            cash *= m.c[t] / m.c[t - 1]
        for a, d, x in tr_list:
            if a == t:
                cash += x
                pending.append((d, x, t))
        keep = []
        for d, x, a in pending:
            if d == t:
                v = x * m.c[t] / m.c[a]
                cash -= v
                shares += v / m.tr[t]
            else:
                keep.append((d, x, a))
        pending = keep
        out.append(shares * m.tr[t] + cash)
    return np.array(out)


@pytest.mark.parametrize("policy", ["E", "R"])
def test_simulate_matches_brute_force(policy):
    m = make_market()
    sig = m.vix > 24
    for tr in [S.lump_sum(m), S.dca(m), S.signal_wait(m, sig, policy),
               S.lump_sum(m).scaled(0.5) + S.signal_wait(m, sig, policy).scaled(0.5)]:
        assert np.allclose(S.simulate(m, tr)["wealth"], brute_force_wealth(m, tr))


def test_all_strategies_have_identical_external_flows():
    m = make_market()
    sig = m.vix > 24
    for tr in [S.lump_sum(m), S.dca(m), S.signal_wait(m, sig, "E")]:
        by_day = pd.Series(tr.amount).groupby(tr.arrive).sum()
        assert list(by_day.index) == list(m.year_first)
        assert np.allclose(by_day.values, 1.0)


def test_signal_executes_day_after_signal():
    sig = np.zeros(10, dtype=bool)
    sig[3] = True
    nxt = S.next_execution(sig)
    assert nxt[0] == 4 and nxt[4] == 4 and nxt[5] == 10


def test_signal_on_last_day_is_not_executable():
    sig = np.zeros(5, dtype=bool)
    sig[4] = True
    assert (S.next_execution(sig) == 5).all()


def test_future_signal_changes_nothing_before_it():
    """将来のシグナルを変えても、それ以前の評価額は変わらない（先読みなし）。"""
    m = make_market()
    sig = m.vix > 24
    sig2 = sig.copy()
    cut = 500
    sig2[cut:] = ~sig2[cut:]
    a = S.simulate(m, S.signal_wait(m, sig, "R"))["wealth"]
    b = S.simulate(m, S.signal_wait(m, sig2, "R"))["wealth"]
    assert np.allclose(a[:cut + 1], b[:cut + 1])


def test_policy_E_deploys_within_year():
    m = make_market()
    tr = S.signal_wait(m, np.zeros(m.n, dtype=bool), "E")
    assert (tr.deploy == m.year_last).all()


def test_policy_R_keeps_cash_without_signal():
    m = make_market()
    tr = S.signal_wait(m, np.zeros(m.n, dtype=bool), "R")
    sim = S.simulate(m, tr)
    assert sim["equity"][-1] == 0.0
    assert np.isclose(sim["cash"][-1], sum(m.c[-1] / m.c[a] for a in m.year_first))


def test_tactical_position_timing():
    sig = np.zeros(12, dtype=bool)
    sig[2] = True
    pos = S.tactical_position(sig, hold=3)
    # 2日引けシグナル→3日引け買付→4,5,6日のリターンを受け取る
    assert list(np.flatnonzero(pos)) == [4, 5, 6]


def test_tactical_returns_cost():
    m = make_market()
    pos = np.zeros(m.n, dtype=bool)
    pos[10:20] = True
    gross = S.tactical_returns(m, pos)
    net = S.tactical_returns(m, pos, cost_bps=10)
    assert np.isclose((gross - net).sum(), 2 * 10 / 1e4)


def test_vintage_equals_simulated_wealth_for_single_year():
    m = make_market(n_years=1)
    tr = S.dca(m)
    assert np.isclose(S.vintage_values(m, tr)[0], S.simulate(m, tr)["wealth"][-1])


def test_irr_known_answer():
    # 1を2回(0年,1年)投入し、2年後に1.1^2+1.1=2.31 → IRR=10%
    r = M.irr(np.array([0.0, 1.0]), np.array([1.0, 1.0]), 2.0, 2.31)
    assert r == pytest.approx(0.10, abs=1e-8)


def test_max_drawdown_known_answer():
    assert M.max_drawdown(np.array([0.1, -0.5, 0.2])) == pytest.approx(-0.5)


def test_block_indices_within_range():
    idx = M.moving_block_indices(100, 10, 5, np.random.default_rng(0))
    assert idx.shape == (5, 100) and idx.min() >= 0 and idx.max() < 100


def test_sign_test_excludes_ties():
    from src.metrics import sign_test
    r = sign_test(np.array([1.0, 2.0, -1.0, 0.0, 0.0, 1e-15]))
    assert (r["wins"], r["losses"], r["ties"]) == (2, 1, 3)
    assert r["p"] == pytest.approx(1.0)
    assert np.isnan(sign_test(np.zeros(3))["p"])


def test_reserve_mix_is_weighted_sum_of_components():
    """アンサンブルの評価額は、構成要素の評価額の加重和に一致する。"""
    from run_01_strategies import reserve_mix
    m = make_market(n_years=4, seed=2)
    signal = m.vix >= 22
    mix = S.simulate(m, reserve_mix(m, signal, "R", 1 / 3))["wealth"]
    parts = [S.lump_sum(m), S.dca(m), S.signal_wait(m, signal, "R")]
    expected = sum(S.simulate(m, p)["wealth"] for p in parts) / 3
    assert mix == pytest.approx(expected)
    only = S.simulate(m, reserve_mix(m, signal, "E", 1.0))["wealth"]
    assert only == pytest.approx(S.simulate(m, S.signal_wait(m, signal, "E"))["wealth"])
