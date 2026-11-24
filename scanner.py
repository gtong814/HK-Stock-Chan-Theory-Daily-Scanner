import os
import datetime
import requests
import akshare as ak
import pandas as pd
import numpy as np
import yfinance as yf

def get_all_hk_connect_stocks():
    """從新浪財經或東方財富接口抓取全市場約 500+ 隻港股通成份股名單"""
    stocks = {}
    
    # 來源 1：新浪財經港股通成份股
    try:
        df_sina = ak.stock_hk_ggt_components_sina()
        if df_sina is not None and not df_sina.empty:
            for _, row in df_sina.iterrows():
                code = str(row['symbol']).strip().zfill(5)
                name = str(row['name']).strip()
                yf_code = f"{code[1:]}.HK" if code.startswith("0") else f"{code}.HK"
                stocks[yf_code] = name
            if len(stocks) > 400:
                return stocks, "全市場港股通股份"
    except Exception as e:
        print(f"新浪接口嘗試失敗: {e}")

    # 來源 2：新浪港股通分頁 API
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        for page in range(1, 10):
            url = f"http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHKStockData?page={page}&num=80&sort=symbol&asc=1&node=hgt_hk"
            resp = requests.get(url, headers=headers, timeout=10)
            data = resp.json()
            if not data:
                break
            for item in data:
                code = str(item['symbol']).strip().zfill(5)
                name = str(item['name']).strip()
                yf_code = f"{code[1:]}.HK" if code.startswith("0") else f"{code}.HK"
                stocks[yf_code] = name
        if len(stocks) > 400:
            return stocks, "全市場港股通股份"
    except Exception as e:
        print(f"新浪 API 備援失敗: {e}")

    # 來源 3：東方財富接口
    try:
        df_em = ak.stock_hk_ggt_components_em()
        if df_em is not None and not df_em.empty:
            for _, row in df_em.iterrows():
                code = str(row['代碼']).strip().zfill(5)
                yf_code = f"{code[1:]}.HK" if code.startswith("0") else f"{code}.HK"
                stocks[yf_code] = str(row['名稱']).strip()
            if len(stocks) > 400:
                return stocks, "全市場港股通股份"
    except Exception:
        pass

    return stocks, "港股通股份"

def analyze_single_df(ticker, name, df):
    """計算個股纏論結構（分型、中樞、買點分類及底分型止損價）"""
    try:
        df = df.dropna(subset=['Close', 'High', 'Low', 'Volume'])
        if len(df) < 35:
            return None

        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values

        current_price = float(close[-1])
        prev_price = float(close[-2])
        if current_price <= 0 or prev_price <= 0:
            return None

        change_pct = (current_price - prev_price) / prev_price * 100
        turnover_hkd = current_price * float(volume[-1])

        # 過濾成交額低於 1,000 萬港幣的冷門股
        if turnover_hkd < 10000000:
            return None

        # 計算近 30 日中樞區間 (ZD / ZG)
        zs_zg = round(float(np.percentile(high[-30:], 70)), 2)
        zs_zd = round(float(np.percentile(low[-30:], 30)), 2)

        # 底分型定義：中間 K 線（-2）最低點為近三日最低，且最新收盤價站上中間 K 線高點
        fractal_low = float(low[-2])
        is_bottom_fractal = (fractal_low < float(low[-3])) and (fractal_low < float(low[-1])) and (current_price > float(high[-2]))
        
        if not is_bottom_fractal:
            return None

        # 計算防守止損位（底分型最低點）與止損距離百分比
        stop_loss = round(fractal_low, 2)
        stop_loss_pct = round((stop_loss - current_price) / current_price * 100, 2)

        # 檢查近期是否創出 20 日新低後反彈（一買背馳特徵）
        min_20 = float(np.min(low[-20:-3]))
        is_new_low_rebound = (fractal_low <= min_20) and (current_price > prev_price)

        category = None
        signal = None

        if current_price > zs_zg:
            category = "breakout"
            signal = "🔥 三買確認（中樞上方回踩底分型）"
        elif is_new_low_rebound and current_price <= zs_zd:
            category = "reversal"
            signal = "⭐ 一買潛力（中樞下方底背馳分型）"
        elif zs_zd <= current_price <= zs_zg:
            category = "reversal"
            signal = "✅ 二買觀察（中樞下沿／箱體底分型）"

        if category and signal:
            return {
                "category": category,
                "代號": ticker,
                "名稱": name,
                "現價": round(current_price, 2),
                "升跌幅": f"{change_pct:+.2f}%",
                "成交額(億)": round(turnover_hkd / 100000000, 2),
                "中樞區間(ZD-ZG)": f"{zs_zd} - {zs_zg}",
                "建議止損價": stop_loss,
                "止損幅度": f"{stop_loss_pct:.2f}%",
                "纏論買點訊號": signal,
                "_turnover": turnover_hkd
            }
        return None
    except Exception:
        return None

def build_table_html(df_subset, empty_msg):
    if df_subset.empty:
        return f"<p style='text-align:center; padding: 20px; color: #888;'>{empty_msg}</p>"
    df_display = df_subset.sort_values(by="_turnover", ascending=False).drop(columns=["category", "_turnover"])
    return df_display.to_html(index=False, border=0, classes="stock-table")

def main():
    stock_dict, source_label = get_all_hk_connect_stocks()
    tickers_list = list(stock_dict.keys())
    total_list_count = len(tickers_list)
    print(f"準備批量下載 {total_list_count} 隻港股通股份的歷史 K 線...")

    results = []
    batch_size = 150
    for i in range(0, total_list_count, batch_size):
        batch_tickers = tickers_list[i:i + batch_size]
        try:
            data = yf.download(batch_tickers, period="6mo", interval="1d", group_by="ticker", progress=False, threads=True)
            for ticker in batch_tickers:
                if ticker in data.columns.get_level_values(0):
                    df_ticker = data[ticker]
                    res = analyze_single_df(ticker, stock_dict[ticker], df_ticker)
                    if res:
                        results.append(res)
        except Exception as e:
            print(f"批次 {i} 下載異常: {e}")

    df_all = pd.DataFrame(results)
    if not df_all.empty:
        df_breakout = df_all[df_all["category"] == "breakout"].copy()
        df_reversal = df_all[df_all["category"] == "reversal"].copy()
    else:
        df_breakout = pd.DataFrame()
        df_reversal = pd.DataFrame()

    breakout_html = build_table_html(df_breakout, "今日暫無符合「強勢突破（三買）」條件的港股通股份。")
    reversal_html = build_table_html(df_reversal, "今日暫無符合「低位反轉（一買／二買）」條件的港股通股份。")

    hk_tz = datetime.timezone(datetime.timedelta(hours=8))
    update_time = datetime.datetime.now(hk_tz).strftime("%Y-%m-%d %H:%M:%S")
    total_matched = len(df_all)

    html_content = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>港股通纏論雙策略掃描 (Chan Theory Scanner)</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: auto; background: #1e1e1e; padding: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.6); overflow-x: auto; }}
        h1 {{ text-align: center; color: #ffffff; margin-bottom: 8px; font-size: 24px; }}
        .meta {{ text-align: center; color: #9e9e9e; font-size: 14px; margin-bottom: 25px; }}
        .badge {{ background: #2c2c2c; color: #00e676; padding: 3px 10px; border-radius: 12px; font-weight: bold; }}
        .section-title {{ margin-top: 30px; margin-bottom: 12px; padding-left: 12px; font-size: 18px; font-weight: bold; color: #ffffff; }}
        .title-breakout {{ border-left: 4px solid #ff9800; }}
        .title-reversal {{ border-left: 4px solid #00e676; }}
        .section-desc {{ font-size: 13px; color: #9e9e9e; margin-bottom: 12px; padding-left: 16px; }}
        table.stock-table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 14px; }}
        table.stock-table th, table.stock-table td {{ padding: 11px 13px; text-align: left; border-bottom: 1px solid #333; white-space: nowrap; }}
        table.stock-table th {{ background-color: #262626; color: #00e676; font-weight: 600; }}
        table.stock-table tr:hover {{ background-color: #2a2a2a; }}
        .footer-note {{ margin-top: 25px; padding-top: 15px; border-top: 1px solid #333; font-size: 13px; color: #888; text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>港股通纏論雙策略買點掃描 (Chan Theory Scanner)</h1>
        <div class="meta">
            掃描範圍：{source_label} ({total_list_count} 隻) ｜ 
            總符合數：<span class="badge">{total_matched} 隻</span> ｜ 
            更新時間：{update_time} (HKT)
        </div>

        <div class="section-title title-breakout">🔥 策略一：強勢突破組 — 三買確認（共 {len(df_breakout)} 隻）</div>
        <div class="section-desc">特徵：股價已突破近 30 日中樞上沿 (ZG)，回踩不破並出現底分型，適合順勢捕捉主升段。</div>
        {breakout_html}

        <div class="section-title title-reversal">⭐ 策略二：低位反轉組 — 一買／二買觀察（共 {len(df_reversal)} 隻）</div>
        <div class="section-desc">特徵：股價處於中樞下沿 (ZD) 附近或底背馳止跌形成底分型，適合低吸佈局、止損空間小。</div>
        {reversal_html}

        <div class="footer-note">
            紀律提示：「建議止損價」為該底分型中間 K 線之最低價；若後續日線收盤跌破此價位，代表分型結構失效。
        </div>
    </div>
</body>
</html>
"""
    os.makedirs("public", exist_ok=True)
    with open("public/index.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"掃描完成！三買 {len(df_breakout)} 隻，一/二買 {len(df_reversal)} 隻。")

if __name__ == "__main__":
    main()
