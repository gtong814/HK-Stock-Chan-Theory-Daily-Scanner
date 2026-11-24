import os
import datetime
import requests
import akshare as ak
import pandas as pd
import numpy as np
import yfinance as yf

def get_all_hk_connect_stocks():
    """從新浪財經或騰訊財經接口抓取全市場約 500+ 隻港股通成份股名單（支援海外伺服器）"""
    stocks = {}
    
    # 來源 1：新浪財經港股通成份股（通常包含 500+ 隻全部港股通標的）
    try:
        df_sina = ak.stock_hk_ggt_components_sina()
        if df_sina is not None and not df_sina.empty:
            for _, row in df_sina.iterrows():
                code = str(row['symbol']).strip().zfill(5)
                name = str(row['name']).strip()
                yf_code = f"{code[1:]}.HK" if code.startswith("0") else f"{code}.HK"
                stocks[yf_code] = name
            if len(stocks) > 400:
                print(f"成功從新浪財經獲取全市場港股通名單：共 {len(stocks)} 隻")
                return stocks, "全市場港股通股份"
    except Exception as e:
        print(f"新浪接口嘗試失敗: {e}")

    # 來源 2：新浪港股通分頁 API 直接抓取（備援確保抓滿 500+ 隻）
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
            print(f"成功從新浪 API 獲取全市場港股通名單：共 {len(stocks)} 隻")
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
    """計算單隻股票的纏論結構（分型、中樞區間與買點訊號）"""
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

        # 過濾掉成交額低於 1,000 萬港幣的極低流動性股份
        if turnover_hkd < 10000000:
            return None

        # 計算近 30 日中樞區間 (ZD / ZG)
        zs_zg = round(float(np.percentile(high[-30:], 70)), 2)
        zs_zd = round(float(np.percentile(low[-30:], 30)), 2)

        # 標準底分型條件：中間 K 線最低點為三根最低，且第三根收盤站上中間 K 線高點
        is_bottom_fractal = (low[-2] < low[-3]) and (low[-2] < low[-1]) and (close[-1] > high[-2])
        
        # 檢查近期是否出現底背馳跡象（創出 20 日新低後止跌回升）
        min_20 = np.min(low[-20:-3])
        is_new_low_rebound = (low[-2] <= min_20) and (close[-1] > close[-2])

        signal = None
        if is_bottom_fractal:
            if current_price > zs_zg:
                signal = "🔥 三買確認（中樞上方底分型）"
            elif is_new_low_rebound and current_price <= zs_zd:
                signal = "⭐ 一買潛力（中樞下方底背馳分型）"
            elif zs_zd <= current_price <= zs_zg:
                signal = "✅ 二買/中樞震盪底分型"

        if signal:
            return {
                "代號": ticker,
                "名稱": name,
                "現價": round(current_price, 2),
                "升跌幅": f"{change_pct:+.2f}%",
                "成交額(億)": round(turnover_hkd / 100000000, 2),
                "中樞下沿(ZD)": zs_zd,
                "中樞上沿(ZG)": zs_zg,
                "纏論訊號": signal,
                "_turnover": turnover_hkd
            }
        return None
    except Exception:
        return None

def main():
    stock_dict, source_label = get_all_hk_connect_stocks()
    tickers_list = list(stock_dict.keys())
    total_list_count = len(tickers_list)
    print(f"準備批量下載 {total_list_count} 隻港股通股份的歷史 K 線...")

    results = []
    # 分批（每批 150 隻）批量下載，速度極快且穩定
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

    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res = df_res.sort_values(by="_turnover", ascending=False).drop(columns=["_turnover"])

    hk_tz = datetime.timezone(datetime.timedelta(hours=8))
    update_time = datetime.datetime.now(hk_tz).strftime("%Y-%m-%d %H:%M:%S")
    matched_count = len(df_res)

    if not df_res.empty:
        table_html = df_res.to_html(index=False, border=0, classes="stock-table")
    else:
        table_html = "<p style='text-align:center; padding: 30px; color: #aaa;'>今日全市場港股通股份中暫無符合纏論底分型買點條件的標的。</p>"

    html_content = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>港股通纏論每日掃描 (Chan Theory Scanner)</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; padding: 20px; }}
        .container {{ max-width: 1150px; margin: auto; background: #1e1e1e; padding: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.6); overflow-x: auto; }}
        h1 {{ text-align: center; color: #ffffff; margin-bottom: 8px; font-size: 24px; }}
        .meta {{ text-align: center; color: #9e9e9e; font-size: 14px; margin-bottom: 22px; }}
        .badge {{ background: #2c2c2c; color: #00e676; padding: 3px 10px; border-radius: 12px; font-weight: bold; }}
        table.stock-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 15px; }}
        table.stock-table th, table.stock-table td {{ padding: 12px 14px; text-align: left; border-bottom: 1px solid #333; white-space: nowrap; }}
        table.stock-table th {{ background-color: #262626; color: #00e676; font-weight: 600; }}
        table.stock-table tr:hover {{ background-color: #2a2a2a; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>港股通纏論每日買點掃描 (Chan Theory Scanner)</h1>
        <div class="meta">
            掃描範圍：{source_label} ({total_list_count} 隻) ｜ 
            符合條件：<span class="badge">{matched_count} 隻</span> ｜ 
            更新時間：{update_time} (HKT)
        </div>
        {table_html}
    </div>
</body>
</html>
"""
    os.makedirs("public", exist_ok=True)
    with open("public/index.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"掃描完成！共掃描 {total_list_count} 隻，篩選出 {matched_count} 隻符合條件的港股通股份。")

if __name__ == "__main__":
    main()
