import os
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import akshare as ak
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ==========================================
# 1. 檢查今日是否為港股實際交易日（過濾香港公眾假期）
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
# 2. 纏論數據結構與核心演算法
# ==========================================
@dataclass
class MergedBar:
    h: float
    l: float
    dir: int  # 1: 向上, -1: 向下

@dataclass
class Fractal:
    type: int       # 1: 頂分型, -1: 底分型
    price: float
    mIdx: int
    macdArea: float

def get_hk_connect_stocks() -> pd.DataFrame:
    try:
        df_ggt = ak.stock_hk_ggt_components_em()
        df_ggt["yf_ticker"] = df_ggt["代码"].astype(str).str.zfill(4) + ".HK"
        return df_ggt[["yf_ticker", "名称"]].drop_duplicates(subset=["yf_ticker"])
    except Exception:
        fallback = [
            ("9660.HK", "地平線機器人"), ("6082.HK", "壁仞科技"),
            ("9880.HK", "優必選"), ("1686.HK", "新意網集團"),
            ("2800.HK", "盈富基金"), ("0700.HK", "騰訊控股"),
            ("9988.HK", "阿里巴巴-W"), ("3690.HK", "美團-W")
        ]
        return pd.DataFrame(fallback, columns=["yf_ticker", "名称"])

def calc_macd_hist(close_series: pd.Series) -> pd.Series:
    ema12 = close_series.ewm(span=12, adjust=False).mean()
    ema26 = close_series.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    return dif - dea

def analyze_chan_state(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float], float, str, int]:
    if df is None or len(df) < 60:
        return None, None, None, 0.0, "數據不足", 0

    df = df.copy()
    df["hist"] = calc_macd_hist(df["Close"])
    mBars: List[MergedBar] = []
    fractals: List[Fractal] = []
    currMacdSum: float = 0.0

    for _, row in df.iterrows():
        high = float(row["High"])
        low = float(row["Low"])
        hist_val = float(row["hist"]) if not pd.isna(row["hist"]) else 0.0
        currMacdSum += abs(hist_val)

        # K線包含處理
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

        # 識別頂底分型與成筆
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

    # 中樞（ZG/ZD）與三類買點分類
    latest_close = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else latest_close
    pct_change = ((latest_close - prev_close) / prev_close) * 100 if prev_close != 0 else 0.0

    if len(fractals) < 5:
        return latest_close, None, None, pct_change, "結構醞釀中", 0

    p0, p1, p2, p3 = fractals[-4], fractals[-3], fractals[-2], fractals[-1]
    zg = min(max(p0.price, p1.price), max(p1.price, p2.price), max(p2.price, p3.price))
    zd = max(min(p0.price, p1.price), min(p1.price, p2.price), min(p2.price, p3.price))

    if zg <= zd:
        return latest_close, None, None, pct_change, "無有效中樞", 0

    if p3.type == -1 and p3.price > zg and latest_close >= zg:
        return latest_close, zg, zd, pct_change, "三買 (回抽不破ZG)", 3
    elif p3.type == -1 and p1.type == -1 and p3.price > p1.price and latest_close > p3.price:
        return latest_close, zg, zd, pct_change, "二買 (次低點確認)", 2
    elif p3.type == -1 and p3.price < zd and p3.macdArea < p1.macdArea:
        return latest_close, zg, zd, pct_change, "一買 (跌破ZD底背馳)", 1
    else:
        return latest_close, zg, zd, pct_change, "其他狀態", 0

# ==========================================
# 3. 生成 GitHub Pages 靜態網頁（綠升紅跌 + 篩選按鈕）
# ==========================================
def generate_html_dashboard(buy_signals: list, update_time: str):
    os.makedirs("public", exist_ok=True)
    
    count_all = len(buy_signals)
    count_b1 = sum(1 for x in buy_signals if x["優先級"] == 1)
    count_b2 = sum(1 for x in buy_signals if x["優先級"] == 2)
    count_b3 = sum(1 for x in buy_signals if x["優先級"] == 3)

    rows_html = ""
    for item in buy_signals:
        code_num = item["代碼"].replace(".HK", "")
        tv_url = f"https://www.tradingview.com/chart/?symbol=HKEX%3A{code_num}"
        pct = item["漲跌幅"]
        pct_class = "up" if pct > 0 else ("down" if pct < 0 else "flat")
        pct_str = f"+{pct:.2f}%" if pct > 0 else f"{pct:.2f}%"
        badge_class = f"badge-b{item['優先級']}"
        
        rows_html += f"""
        <tr data-type="{item['優先級']}">
            <td><a href="{tv_url}" target="_blank" class="ticker-link">{item['代碼']}</a></td>
            <td>{item['名稱']}</td>
            <td class="num">{item['收盤']:.2f}</td>
            <td class="num {pct_class}">{pct_str}</td>
            <td><span class="badge {badge_class}">{item['類型']}</span></td>
            <td class="num">{item['中樞下沿(ZD)']:.2f} - {item['中樞上沿(ZG)']:.2f}</td>
            <td class="num stop-loss">{item['參考止損']:.2f}</td>
        </tr>
        """

    if not rows_html:
        rows_html = """<tr><td colspan="7" class="empty-msg">今日掃描完成，暫無符合嚴格一買、二買或三買條件的港股通標的。</td></tr>"""

    html_content = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>港股通纏論日線掃描看板</title>
    <style>
        :root {{
            --bg-main: #0f172a;
            --bg-card: #1e293b;
            --bg-hover: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border: #334155;
            --green: #00c805;
            --red: #ff333a;
            --blue: #38bdf8;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans TC", sans-serif;
            background-color: var(--bg-main);
            color: var(--text-main);
            padding: 24px;
            line-height: 1.5;
        }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        header {{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border);
        }}
        h1 {{ font-size: 1.5rem; font-weight: 700; }}
        .meta {{ color: var(--text-muted); font-size: 0.9rem; margin-top: 4px; }}
        .controls {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }}
        .filter-group {{ display: flex; flex-wrap: wrap; gap: 8px; }}
        button.filter-btn {{
            background: var(--bg-card);
            color: var(--text-main);
            border: 1px solid var(--border);
            padding: 8px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.15s;
        }}
        button.filter-btn:hover {{ background: var(--bg-hover); }}
        button.filter-btn.active {{
            background: var(--blue);
            color: #0f172a;
            font-weight: 600;
            border-color: var(--blue);
        }}
        input.search-box {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 8px 14px;
            border-radius: 6px;
            font-size: 0.9rem;
            width: 220px;
        }}
        .table-wrap {{
            background: var(--bg-card);
            border-radius: 10px;
            overflow-x: auto;
            border: 1px solid var(--border);
        }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; }}
        th, td {{ padding: 12px 16px; border-bottom: 1px solid var(--border); white-space: nowrap; }}
        th {{ background: rgba(15, 23, 42, 0.6); color: var(--text-muted); font-weight: 600; font-size: 0.85rem; }}
        tr:hover {{ background: var(--bg-hover); }}
        .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
        .up {{ color: var(--green); font-weight: 600; }}
        .down {{ color: var(--red); font-weight: 600; }}
        .flat {{ color: var(--text-muted); }}
        .stop-loss {{ color: var(--red); }}
        .ticker-link {{ color: var(--blue); text-decoration: none; font-weight: 600; }}
        .ticker-link:hover {{ text-decoration: underline; }}
        .badge {{
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        .badge-b1 {{ background: rgba(0, 200, 5, 0.18); color: #4ade80; border: 1px solid rgba(0, 200, 5, 0.4); }}
        .badge-b2 {{ background: rgba(56, 189, 248, 0.18); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); }}
        .badge-b3 {{ background: rgba(250, 204, 21, 0.18); color: #facc15; border: 1px solid rgba(250, 204, 21, 0.4); }}
        .empty-msg {{ text-align: center; padding: 32px; color: var(--text-muted); }}
    </style>
</head>
<body>
<div class="container">
    <header>
        <div>
            <h1>港股通纏論日線買點掃描看板</h1>
            <div class="meta">最後更新時間 (HKT)：{update_time} ｜ 點擊代號可直接開啟 TradingView 圖表</div>
        </div>
    </header>

    <div class="controls">
        <div class="filter-group">
            <button class="filter-btn active" onclick="filterTable('all', this)">全部買點 ({count_all})</button>
            <button class="filter-btn" onclick="filterTable('1', this)">⚡ 一買背馳 ({count_b1})</button>
            <button class="filter-btn" onclick="filterTable('2', this)">▲ 二買確認 ({count_b2})</button>
            <button class="filter-btn" onclick="filterTable('3', this)">★ 三買突破 ({count_b3})</button>
        </div>
        <input type="text" id="searchInput" class="search-box" onkeyup="searchTable()" placeholder="搜尋代號或股票名稱...">
    </div>

    <div class="table-wrap">
        <table id="signalTable">
            <thead>
                <tr>
                    <th>代號</th>
                    <th>名稱</th>
                    <th class="num">最新收盤</th>
                    <th class="num">日漲跌幅</th>
                    <th>纏論買點信號</th>
                    <th class="num">最近中樞 (ZD - ZG)</th>
                    <th class="num">結構參考止損</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
</div>

<script>
    let currentType = 'all';

    function filterTable(type, btn) {{
        currentType = type;
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        applyFilters();
    }}

    function searchTable() {{
        applyFilters();
    }}

    function applyFilters() {{
        const keyword = document.getElementById('searchInput').value.toLowerCase();
        const rows = document.querySelectorAll('#signalTable tbody tr');
        rows.forEach(row => {{
            const rowType = row.getAttribute('data-type');
            if (!rowType) return;
            const text = row.innerText.toLowerCase();
            const matchType = (currentType === 'all' || rowType === currentType);
            const matchText = text.includes(keyword);
            row.style.display = (matchType && matchText) ? '' : 'none';
        }});
    }}
</script>
</body>
</html>
"""
    with open("public/index.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("✅ 已成功生成網頁看板: public/index.html")

# ==========================================
# 4. 主執行流程
# ==========================================
def main():
    hkt_now = datetime.now(timezone(timedelta(hours=8)))
    update_time_str = hkt_now.strftime("%Y-%m-%d %H:%M:%S")

    if not is_hk_trading_day_today():
        print("今日為港股休市日，僅更新頁面時間戳。")
        if not os.path.exists("public/index.html"):
            generate_html_dashboard([], update_time_str + " (今日休市)")
        return

    stock_df = get_hk_connect_stocks()
    tickers = stock_df["yf_ticker"].tolist()
    name_map = dict(zip(stock_df["yf_ticker"], stock_df["名称"]))
    print(f"開始掃描 {len(tickers)} 隻港股通標的...")

    batch_size = 100
    buy_signals = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            data = yf.download(
                batch, period="1y", interval="1d",
                auto_adjust=True, group_by="ticker", progress=False, threads=True
            )
        except Exception:
            continue

        for ticker in batch:
            try:
                df_single = data.dropna(subset=["Close", "High", "Low"]) if len(batch) == 1 else data[ticker].dropna(subset=["Close", "High", "Low"])
                close_p, zg, zd, pct_chg, status, code = analyze_chan_state(df_single)
                if code in [1, 2, 3]:
                    stop_loss = zg if code == 3 else zd
                    buy_signals.append({
                        "類型": status,
                        "代碼": ticker,
                        "名稱": name_map.get(ticker, ""),
                        "收盤": close_p,
                        "漲跌幅": pct_chg,
                        "中樞下沿(ZD)": zd,
                        "中樞上沿(ZG)": zg,
                        "參考止損": stop_loss,
                        "優先級": code
                    })
            except Exception:
                continue
        time.sleep(1)

    buy_signals.sort(key=lambda x: (x["優先級"], x["代碼"]))
    generate_html_dashboard(buy_signals, update_time_str)

if __name__ == "__main__":
    main()
