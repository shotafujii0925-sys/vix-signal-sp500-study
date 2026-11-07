# VIXはS&P500のリターンにどの程度影響するか

VIXを投資シグナルとして、年初一括・ドルコスト平均法・それらのアンサンブルと比較し、その後の回帰分析とコンフォーマル予測までを一貫して評価する研究です。

**結果は [REPORT.md](REPORT.md) にまとめています。**

## 構成

| 段階 | スクリプト | 内容 |
|---|---|---|
| 1 | `run_01_strategies.py` | 資金投入型・売買型の比較、ビンテージ分析、プラセボ検定 |
| 2 | `run_02_signal.py` | VIX局面別のリターン、閾値検定、初回到達イベント |
| 3 | `run_03_regression.py` | 連続・離散VIXの回帰、OOS R²、実現ボラ、統制変数 |
| 4 | `run_04_conformal.py` | 8手法のコンフォーマル予測（Winkler、局面別カバレッジ） |
| — | `run_05_figures.py` / `build_report.py` | 図とレポートの生成 |

共通の処理は `src/`、テストは `tests/`、出力は `results/` と `figures/` にあります。

## 再現

```bash
pip install -r requirements.txt
python -m pytest -q
python -X utf8 run_all.py
```

市場データは同梱していません（[DATA_POLICY.md](DATA_POLICY.md)）。上の手順の前に、利用条件を確認したうえで取得してください。

```bash
python -m src.data --refresh
```

データは再取得になるため、`REPORT.md` の数値と完全には一致しないことがあります。データがない環境では、データに依存するテストをスキップします。

## 注意

- 実装と本文の作成にはAI（Claude）を使用しています。
- 統計的な検証であり、個別の投資助言ではありません。
