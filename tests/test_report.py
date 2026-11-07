# -*- coding: utf-8 -*-
"""REPORT.md が結果と一致し、未解決の値や存在しない図を含まないことを確認する。"""
import re

import pandas as pd
import pytest

from src.config import ROOT, RESULTS_DIR, SNAPSHOT

build_report = pytest.importorskip("build_report")
pytestmark = pytest.mark.skipif(not SNAPSHOT.exists(), reason="市場データは同梱していない（DATA_POLICY.md）")


@pytest.fixture(scope="module")
def text():
    return build_report.build()


def test_no_unresolved_values(text):
    assert not re.search(r"\bnan\b|\bNaN\b|\binf\b", text)
    assert "{" not in text and "}" not in text
    assert not re.search(r"\d\.0(勝|敗|分|年|件|日)", text), "件数が小数で表示されている"


def test_figures_exist(text):
    figures = re.findall(r"\]\((figures/[^)]+)\)", text)
    assert len(figures) == 6
    for f in figures:
        assert (ROOT / f).exists()


def test_values_match_results(text):
    bm = pd.read_csv(RESULTS_DIR / "s1_budget_metrics.csv").set_index("strategy")
    assert f"{bm.loc['年初一括', 'irr'] * 100:.2f}%" in text
    assert f"{bm.loc['一括+DCA+VIX>=30', 'sharpe']:.3f}" in text
    ov = pd.read_csv(RESULTS_DIR / "s4_overall.csv").set_index(["horizon", "method"])
    assert f"{ov.loc[('1M', 'Norm'), 'winkler_pct']:.2f} / {ov.loc[('1M', 'Norm'), 'coverage'] * 100:.1f}%" in text
    co = pd.read_csv(RESULTS_DIR / "s3_continuous.csv").set_index("horizon")
    assert f"{co.loc['12M', 'slope']:.4f}" in text


def test_saved_report_is_current(text):
    path = ROOT / "REPORT.md"
    if path.exists():
        assert path.read_text(encoding="utf-8") == text
