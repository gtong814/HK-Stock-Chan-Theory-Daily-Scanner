import os
import datetime
import time
import akshare as ak
import pandas as pd
import numpy as np
import yfinance as yf

def get_hk_connect_stocks():
    """透過 akshare 獲取最新港股通（滬港通+深港通）成份股名單與即時行情"""
    try:
        df_ggt = ak.stock_hk_ggt_components_em()
        df_ggt['成交額'] = pd.to_numeric(df_ggt['成交額'], errors='coerce').fillna(0)
        df_ggt['最新價'] = pd.to_numeric(df_ggt['最新價'], errors='coerce').fillna(0)
        df_ggt['漲跌幅'] = pd.to_numeric(df_ggt['漲跌幅'], errors='coerce').fillna(0)
        
        active_df = df_ggt[(df_ggt['最新價'] > 0) & (df_ggt['成交額'] >= 10000000)].copy()
        active_df = active_df.sort_values(by='成交額', ascending=False)
        return active_df
    except Exception as e:
        print(f"獲取港股通名單失敗: {e}")
        return pd.DataFrame()

def analyze_chan_theory(code, name, current_price, change_pct, turnover_hkd):
    """計算個股纏論結構（分型、中樞區間與買點訊號）"""
    yf_ticker = f"{code[1:]}.HK" if code.startswith("0") else f"{code}.HK"
    
    try:
        df = yf.download(yf_ticker, period="6mo", interval="1d", progress=False)
        if df.empty or len(df) < 35:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values

        zs_zg = round(float(np.percentile(high[-30:], 70)), 2)
        zs_zd = round(float(np.percentile(low[-30:], 30)), 2)
        
        is_bottom_fractal = (low[-2] < low[-3]) and (low[-2] < low[-1]) and (close[-1] > high[-2])
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
                "代號": f"{code}.HK",
                "名稱": name,
                "現價": round(current_price, 2),
                "升跌幅": f"{change_pct:+.2f}%",
                "成交額(億)": round(turnover_hkd / 100000000, 2),
                "中樞下沿(ZD)": zs_zd,
                "中樞上沿(ZG)": zs_zg,
                "纏論訊號": signal
            }
        return None
    except Exception:
        return None

def main():
    print("正在獲取最新港股通成份股名單...")
    ggt_df = get_hk_connect_stocks()
    print(f"成功獲取 {len(ggt_df)} 隻活躍港股通股份，開始進行纏論掃描...")

    results = []
    for _, row in ggt_df.iterrows():
        code = str(row['代碼']).zfill(5)
        name = str(row['名稱'])
        price = float(row['最新價'])
        pct = float(row['漲跌幅'])
        turnover = float(row['成交額'])

        res = analyze_chan_theory(code, name, price, pct, turnover)
        if res:
            results.append(res)
        
        time.sleep(0.05)

    df_res = pd.DataFrame(results)

    hk_tz = datetime.timezone(datetime.timedelta(hours=8))
    update_time = datetime.datetime.now(hk_tz).strftime("%Y-%m-%d %H:%M:%S")
    total_scanned = len(ggt_df)
    matched_count = len(df_res)

    if not df_res.empty:
        table_html = df_res.to_html(index=False, border=0, classes="stock-table")
    else:
        table_html = "<p style='text-align:center; padding: 30px; color: #aaa;'>今日港股通股份中暫無符合纏論底分型買點條件的標的。</p>"

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
            掃描範圍：活躍港股通股份 ({total_scanned} 隻) ｜ 
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
    print(f"掃描完成！共篩選出 {matched_count} 隻符合條件的港股通股份。")

if __name__ == "__main__":
    main()
