# -*- coding: utf-8 -*-
"""研究全体で共有する定数。分析を見る前に固定する値はここだけに置く。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
SNAPSHOT = DATA_DIR / "market_snapshot.csv"

# 全段階で共通の分析期間（暦年で完結させ、全戦略の評価期間を揃える）
SAMPLE_START = "1990-01-02"   # ^VIX の開始日
SAMPLE_END = "2025-12-31"

# VIXの区切り。既存研究で使った慣行値をそのまま使い、結果を見て動かさない
VIX_BINS = [0, 20, 30, 40, float("inf")]
VIX_LABELS = ["Normal(<20)", "Alert(20-30)", "Fear(30-40)", "Panic(>=40)"]
THRESHOLDS = [20, 30, 40]

# 投資期間（営業日）: 離散4点と、連続扱いのグリッド
HORIZONS = [21, 63, 126, 252]
HORIZON_NAMES = {21: "1M", 63: "3M", 126: "6M", 252: "12M"}
HORIZON_GRID = list(range(5, 253, 5)) + [252]

# 売買型戦略の取引コスト感度（片道）
COST_BPS = 10

# ブートストラップ・プラセボ
N_BOOT = 2000
SEED = 20261107
PLACEBO_STEP = 5        # 循環シフトの刻み（営業日）
PLACEBO_MIN_SHIFT = 252  # 元の系列との重なりを避ける最小シフト

# コンフォーマル予測
ALPHA = 0.10
GAMMA = 0.01            # ACI学習率（前回研究と同じ。調整しない）
MIN_TRAIN = 1260        # 回帰の最小学習日数（約5年）
CAL_WINDOW = 2520       # 校正窓（約10年）
MIN_CAL = 1260          # 校正に必要な最小件数
REFIT_EVERY = 21        # 回帰の再推定間隔
TEST_START = "2001-01-02"  # 全ホライズン・全手法で共通の評価開始日
MONDRIAN_MIN = 100      # フェーズ別校正の最小件数（下回れば隣接フェーズと統合）
