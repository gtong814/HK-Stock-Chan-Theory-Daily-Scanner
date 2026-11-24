import time
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime, timezone, timedelta

import akshare as ak
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ==========================================
# 0. 檢查今日是否為港股實際交易日 (過濾香港公眾假期)
# ==========================================
def is_hk_trading_day_today() -> bool:
    hkt = timezone(timedelta(hours=8))
    today_str = datetime.now(hkt).strftime("%Y-%m-%d")
    try:
        hsi = yf.download("^HSI", period="5d", interval="1d", progress=False)
        if hsi.empty:
            return False
        latest_market_date = str(hsi.index[-1].date())
        print(f"系統日期 (HKT): {today_str} | 恒指最新交易日: {latest_market_date}")
        return today_str == latest_market_date
    except Exception as e:
        print(f"檢查交易日異常，預設繼續執行: {e}")
        return True

# ==========================================
# 1. 纏論數據結構與核心演算法
# ==========================================
@dataclass
class MergedBar:
    h: float
    l: float
    dir: int

@dataclass
class Fractal:
    type: int
    price: float
    mIdx: int
    macdArea: float

def get_hk_connect_stocks() -> pd.DataFrame:
    try:
        df_ggt = ak.stock_hk_ggt_components_em()
        df_ggt['yf_ticker'] = df_ggt['代码'].astype(str).str.zfill(4) + ".HK"
        return df_ggt[['yf_ticker', '名称']].drop_duplicates(subset=['yf_ticker'])
    except Exception:
        fallback = [
            ("9660.HK", "地平線機器人"), ("6082.HK", "壁仞科技"),
            ("9880.HK", "優必選"), ("1686.HK", "新意網集團"),
            ("2800.HK", "盈富基金"), ("0700.HK", "騰訊控股"),
            ("9988.HK", "阿里巴巴-W"), ("3690.HK", "美團-W")
        ]
        return pd.DataFrame(fallback, columns=['yf_ticker', '名称'])

def calc_macd_hist(close_series: pd.Series) -> pd.Series:
    ema12 = close_series.ewm(span=12, adjust=False).mean()
    ema26 = close_series.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    return dif - dea

def analyze_chan_state(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float], str, int]:
    if df is None or len(df) < 60:
        return None, None, None, "數據不足", 0

    df = df.copy()
    df['hist'] = calc_macd_hist(df['Close'])
    mBars: List[MergedBar] = []
    fractals: List[Fractal] = []
    currMacdSum: float = 0.0

    for _, row in df.iterrows():
        high = float(row['High'])
        low = float(row['Low'])
        hist_val = float(row['hist']) if not pd.isna(row['hist']) else 0.0
        currMacdSum += abs(hist_val)

        if len(mBars) == 0:
            mBars.append(MergedBar(high, low, 1))
        else:
            prev = mBars[-1]
            isInside = (prev.h >= high and prev.l <= low) or (high >= prev.h and low <= prev.l)
            if isInside:
                if prev.dir == 1:
                    prev.h = max(prev.h, high)
                    prev.l = max(prev.l, low)
                else:
                    prev.h = min(prev.h, high)
                    prev.l = min(prev.l, low)
            else:
                newDir = 1 if high > prev.h else -1
                mBars.append(MergedBar(high, low, newDir))

        mSize = len(mBars)
        if mSize >= 3:
            b1, b2, b3 = mBars[-3], mBars[-2], mBars[-1]
            fType, fPrice = 0, 0.0
            if b2.h > b1.h and b2.h > b3.h and b2.l > b1.l and b2.l > b3.l:
                fType, fPrice = 1, b2.h
            elif b2.l < b1.l and b2.l < b3.l and b2.h < b1.h and b2.h < b3.h:
                fType, fPrice = -1, b2.l

            if fType != 0:
                currMIdx = mSize - 2
                if len(fractals) == 0:
                    fractals.append(Fractal(fType, fPrice, currMIdx, currMacdSum))
                    currMacdSum = 0.0
                else:
                    lastF = fractals[-1]
                    if lastF.type == fType:
                        if (fType == 1 and fPrice > lastF.price) or (fType == -1 and fPrice < lastF.price):
                            lastF.price = fPrice
                            lastF.mIdx = currMIdx
                            lastF.macdArea += currMacdSum
                            currMacdSum = 0.0
                    else:
                        if abs(currMIdx - lastF.mIdx) >= 4 and ((fType == 1 and fPrice > lastF.price) or (fType == -1 and fPrice < lastF.price)):
                            fractals.append(Fractal(fType, fPrice, currMIdx, currMacdSum))
                            currMacdSum = 0.0

    latest_close = float(df['Close'].iloc[-1])
    if len(fractals) < 5:
        return latest_close, None, None, "結構醞釀中", 0

    p0, p1, p2, p3 = fractals[-4], fractals[-3], fractals[-2], fractals[-1]
    zg = min(max(p0.price, p1.price), max(p1.price, p2.price), max(p2.price, p3.price))
    zd = max(min(p0.price, p1.price), min(p1.price, p2.price), min(p2.price, p3.price))

    if zg <= zd:
        return latest_close, None, None, "無有效中樞", 0

    if p3.type == -1 and p3.price > zg and latest_close >= zg:
        return latest_close, zg, zd, "★ 三買 (回抽不破ZG)", 3
    elif p3.type == -1 and p1.type == -1 and p3.price > p1.price and latest_close > p3.price:
        return latest_close, zg, zd, "▲ 二買 (次低點確認)", 2
    elif p3.type == -1 and p3.price < zd and p3.macdArea < p1.macdArea:
        return latest_close, zg, zd, "⚡ 一買 (跌破ZD底背馳)", 1
    else:
        return latest_close, zg, zd, "其他狀態", 0

# ==========================================
# 2. 主執行流程
# ==========================================
def main():
    if not is_hk_trading_day_today():
        print("今日為港股休市日（公眾假期），自動跳過掃描。")
        return

    hkt_now = datetime.now(timezone(timedelta(hours=8)))
    date_label = hkt_now.strftime("%Y-%m-%d")

    stock_df = get_hk_connect_stocks()
    tickers = stock_df['yf_ticker'].tolist()
    name_map = dict(zip(stock_df['yf_ticker'], stock_df['名称']))
    print(f"開始掃描 {len(tickers)} 隻港股通標的...")

    batch_size = 100
    buy_signals = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(batch, period="1y", interval="1d", auto_adjust=True, group_by='ticker', progress=False, threads=True)
        except Exception:
            continue

        for ticker in batch:
            try:
                df_single = data.dropna(subset=['Close', 'High', 'Low']) if len(batch) == 1 else data[ticker].dropna(subset=['Close', 'High', 'Low'])
                close_p, zg, zd, status, code = analyze_chan_state(df_single)
                if code in [1, 2, 3]:
                    buy_signals.append({
                        "類型": status,
                        "代碼": ticker,
                        "名稱": name_map.get(ticker, ""),
                        "收盤": round(close_p, 2),
                        "中樞下沿(ZD)": round(zd, 2),
                        "中樞上沿(ZG)": round(zg, 2),
                        "優先級": code
                    })
            except Exception:
                continue
        time.sleep(1)

    print("\n" + "="*60)
    print(f"📊 港股通纏論日線盤後掃描結果 ({date_label})")
    print("="*60)

    if buy_signals:
        res_df = pd.DataFrame(buy_signals).sort_values(by=["優先級", "代碼"]).drop(columns=["優先級"])
        print(res_df.to_string(index=False))

        csv_filename = f"hk_chan_signals_{hkt_now.strftime('%Y%m%d')}.csv"
        res_df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
        print(f"\n✅ 共發現 {len(res_df)} 隻標的，已匯出至檔案: {csv_filename}")
    else:
        print("今日暫無符合嚴格一買、二買或三買條件的標的。")

if __name__ == "__main__":
    main()
