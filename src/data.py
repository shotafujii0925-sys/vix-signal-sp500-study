# -*- coding: utf-8 -*-
"""市場データの取得・検証。既定は保存済みスナップショットの読み込み。

    python -m src.data --refresh   # yfinance から再取得してスナップショットを更新
"""
from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import pandas as pd

from .config import SAMPLE_END, SAMPLE_START, SNAPSHOT, VIX_BINS, VIX_LABELS

TICKERS = {"TR": "^SP500TR", "PX": "^GSPC", "VIX": "^VIX", "IRX": "^IRX"}


def fetch_online() -> pd.DataFrame:
    import yfinance as yf

    out = {}
    for name, tk in TICKERS.items():
        raw = yf.download(tk, start="1989-01-01", auto_adjust=False, progress=False)
        s = raw["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s.index = pd.DatetimeIndex(s.index).tz_localize(None)
        out[name] = s
    return pd.concat(out, axis=1).sort_index()


def validate(df: pd.DataFrame) -> dict:
    if not df.index.is_monotonic_increasing:
        raise ValueError("日付が昇順ではありません")
    if df.index.duplicated().any():
        raise ValueError("重複した日付があります")
    if not (df["TR"] > 0).all():
        raise ValueError("総収益指数に非正値があります")
    if not df["VIX"].between(5, 100).all():
        raise ValueError("VIXの値域が想定外です")
    if not df["IRX"].between(-1, 12).all():
        raise ValueError("T-bill利回りの値域が想定外です（%表記を想定）")
    return {"n": int(len(df)), "start": str(df.index[0].date()),
            "end": str(df.index[-1].date())}


def load(refresh: bool = False) -> pd.DataFrame:
    """分析用データを返す。

    TR : S&P500 総収益指数（配当込み。ファンド信託報酬・税は含まない）
    VIX: 当日終値。VIXは16:15ET、指数は16:00ET確定のため、シグナルの執行は翌日終値とする
    rf : ^IRX(13週T-bill割引利回り, %) を前営業日の値で固定した日次リターン（実日数/360の近似）
    """
    if refresh:
        raw = fetch_online()
        SNAPSHOT.parent.mkdir(exist_ok=True)
        raw.to_csv(SNAPSHOT)
        print(f"[snapshot更新] {SNAPSHOT} 取得日={date.today()}")
    if not SNAPSHOT.exists():
        raise FileNotFoundError(f"{SNAPSHOT} がありません。--refresh で取得してください")
    raw = pd.read_csv(SNAPSHOT, index_col=0, parse_dates=True)
    # TRとVIXが揃う日のみ使う。T-billだけ欠損の日は直前値を使う（先読みなし）
    df = raw.dropna(subset=["TR", "VIX"]).copy()
    df["IRX"] = df["IRX"].ffill()
    df = df.loc[SAMPLE_START:SAMPLE_END].copy()
    validate(df.dropna(subset=["IRX"]))
    days = df.index.to_series().diff().dt.days.fillna(1).to_numpy()
    df["rf"] = df["IRX"].shift(1).bfill().to_numpy() / 100.0 * days / 360.0
    df["rm"] = df["TR"].pct_change().fillna(0.0)
    df["C"] = (1.0 + df["rf"]).cumprod()   # T-bill積立指数
    df["phase"] = pd.cut(df["VIX"], bins=VIX_BINS, labels=VIX_LABELS, right=False)
    # 超過対数リターン（累積）: 将来リターンの差分計算に使う
    df["lx"] = (np.log1p(df["rm"]) - np.log1p(df["rf"])).cumsum()
    return df


def forward_excess(df: pd.DataFrame, h: int) -> pd.Series:
    """t 日のVIXを見て t+1 日引けで買い、h 営業日保有したときの超過対数リターン。

    売買シミュレーションと同じ執行規則に揃える。値が確定するのは t+1+h 日。
    """
    return df["lx"].shift(-(h + 1)) - df["lx"].shift(-1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args()
    d = load(refresh=a.refresh)
    print(d[["TR", "VIX", "IRX"]].describe().round(3))
    print(d["phase"].value_counts().sort_index())
    print(d.index[0].date(), d.index[-1].date(), len(d))
