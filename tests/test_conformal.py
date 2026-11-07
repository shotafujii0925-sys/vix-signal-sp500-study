# -*- coding: utf-8 -*-
"""コンフォーマル予測の検証。合成データのみ。"""
import numpy as np
import pytest

from src import conformal as C


def toy(n=2400, seed=0):
    rng = np.random.default_rng(seed)
    x = np.log(15 + 10 * rng.random(n))
    y = rng.normal(0.01, 0.05, n)
    phase = rng.integers(0, 4, n)
    return y, x, phase


H = 21
FIRST, MINCAL = 300, 300


@pytest.fixture(autouse=True)
def small_windows(monkeypatch):
    monkeypatch.setattr(C, "MIN_CAL", MINCAL)
    monkeypatch.setattr(C, "CAL_WINDOW", 600)
    monkeypatch.setattr(C, "MONDRIAN_MIN", 30)


def run(method, y, x, phase, t_start=1000, t_end=2300, **kw):
    fc = C.walk_forward(y, x, phase, H, start=FIRST, refit=21)
    return C.intervals(method, y, phase, fc, H, t_start, t_end, first=FIRST, **kw), fc


@pytest.mark.parametrize("method", C.METHODS)
def test_future_outcomes_do_not_change_interval(method):
    """t 日の区間は、まだ確定していない y[i] (i > t-1-h) を変えても変わらない。"""
    y, x, phase = toy()
    t = 1500
    a, _ = run(method, y, x, phase, t_end=t + 1)
    y2 = y.copy()
    y2[t - H:] += 5.0   # 未確定の結果を大きく変える
    b, _ = run(method, y2, x, phase, t_end=t + 1)
    assert a["lo"][t] == pytest.approx(b["lo"][t])
    assert a["hi"][t] == pytest.approx(b["hi"][t])


@pytest.mark.parametrize("method", ["Base", "Norm"])
def test_latest_realized_outcome_is_usable(method):
    """確定直後の y[t-1-h] は校正に入る（制約が厳しすぎないことの確認）。"""
    y, x, phase = toy()
    t = 1500
    a, _ = run(method, y, x, phase, t_end=t + 1)
    y2 = y.copy()
    y2[t - 1 - H] += 1000.0
    y2[t - 2 - H] -= 1000.0
    # 上位に2件の極端値が入っても90%分位は小さく動くだけだが、完全一致はしない
    b, _ = run(method, y2, x, phase, t_end=t + 1)
    assert a["hi"][t] != b["hi"][t]


def test_aci_alpha_is_fixed_until_first_feedback():
    y, x, phase = toy()
    res, _ = run("ACI-Base", y, x, phase)
    path = res["alpha_t"]
    t0 = 1000
    # 最初の結果が確定するのは t0+1+h 日
    assert np.allclose(path[t0:t0 + 1 + H], 0.10)
    assert path[t0 + 2 + H] != 0.10


def test_aci_uses_issued_interval_at_that_time():
    y, x, phase = toy()
    res, _ = run("ACI-Base", y, x, phase)
    lo, hi, path = res["lo"], res["hi"], res["alpha_t"]
    t0 = 1000
    for t in range(t0 + 1 + H, 1300):
        i = t - 1 - H
        err = 0.0 if lo[i] <= y[i] <= hi[i] else 1.0
        expected = np.clip(path[t - 1] + C.GAMMA * (0.10 - err), C.ALPHA_MIN, C.ALPHA_MAX)
        assert path[t] == pytest.approx(expected)


@pytest.mark.parametrize("method", ["Base", "Norm", "Mondrian", "Asym"])
def test_coverage_near_target_on_iid_data(method):
    y, x, phase = toy(n=4000, seed=3)
    res, _ = run(method, y, x, phase, t_start=1000, t_end=3900)
    t = np.arange(1000, 3900)
    cov = ((y[t] >= res["lo"][t]) & (y[t] <= res["hi"][t])).mean()
    assert 0.86 < cov < 0.94


def test_normalized_interval_tracks_vix_when_scale_depends_on_it():
    rng = np.random.default_rng(1)
    x = np.log(10 + 40 * rng.random(3000))
    y = rng.normal(0, 1, 3000) * np.exp(x) / 100
    phase = np.digitize(np.exp(x), [20, 30, 40])
    fc = C.walk_forward(y, x, phase, H, start=FIRST, refit=21)
    res = C.intervals("Norm", y, phase, fc, H, 1000, 2900, first=FIRST)
    width = res["hi"] - res["lo"]
    t = np.arange(1000, 2900)
    assert np.corrcoef(width[t], x[t])[0, 1] > 0.9


def test_winkler_known_values():
    assert C.winkler(np.array([5.0]), np.array([0.0]), np.array([10.0]), .1)[0] == pytest.approx(10.0)
    assert C.winkler(np.array([12.0]), np.array([0.0]), np.array([10.0]), .1)[0] == pytest.approx(50.0)
    assert C.winkler(np.array([-1.0]), np.array([0.0]), np.array([10.0]), .1)[0] == pytest.approx(30.0)


def test_quantile_finite_sample_rule():
    s = np.arange(1.0, 20.0)  # n=19, ceil(20*0.9)=18 番目
    assert C.upper_quantile(s, 0.1) == 18.0
    assert C.upper_quantile(np.arange(1.0, 6.0), 0.01) == 5.0
    assert C.lower_quantile(np.arange(1.0, 20.0), 0.1) == 2.0
