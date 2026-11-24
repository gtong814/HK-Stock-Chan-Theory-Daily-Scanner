import os
import datetime
import time
import akshare as ak
import pandas as pd
import numpy as np
import yfinance as yf

# 內建核心港股通精選池（當海外伺服器無法連接內地接口時的 100% 穩定備援，涵蓋南向資金最活躍標的）
CORE_HK_CONNECT = {
    "0700.HK": "騰訊控股", "9988.HK": "阿里巴巴-W", "3690.HK": "美團-W", "1810.HK": "小米集團-W",
    "0981.HK": "中芯國際", "9660.HK": "地平線機器人-W", "1211.HK": "比亞迪股份", "2015.HK": "理想汽車-W",
    "9868.HK": "小鵬汽車-W", "9866.HK": "蔚來-SW", "0175.HK": "吉利汽車", "2333.HK": "長城汽車",
    "0388.HK": "香港交易所", "0005.HK": "匯豐控股", "1299.HK": "友邦保險", "2318.HK": "中國平安",
    "2628.HK": "中國人壽", "3968.HK": "招商銀行", "0939.HK": "建設銀行", "1398.HK": "工商銀行",
    "3988.HK": "中國銀行", "1288.HK": "農業銀行", "0883.HK": "中國海洋石油", "0857.HK": "中國石油股份",
    "0386.HK": "中國石油化工", "1088.HK": "中國神華", "1898.HK": "中煤能源", "2899.HK": "紫金礦業",
    "3993.HK": "洛陽鉬業", "1378.HK": "中國宏橋", "2600.HK": "中國鋁業", "1818.HK": "招金礦業",
    "0941.HK": "中國移動", "0728.HK": "中國電信", "0762.HK": "中國聯通", "0788.HK": "中國鐵塔",
    "1024.HK": "快手-W", "9618.HK": "京東集團-SW", "9999.HK": "網易-S", "9888.HK": "百度集團-SW",
    "9626.HK": "嗶哩嗶哩-W", "9961.HK": "攜程集團-S", "2382.HK": "舜宇光學科技", "2018.HK": "瑞聲科技",
    "0285.HK": "比亞迪電子", "0992.HK": "聯想集團", "1347.HK": "華虹半導體", "0268.HK": "金蝶國際",
    "3888.HK": "金山軟件", "0020.HK": "商湯-W", "9880.HK": "優必選", "2432.HK": "越疆",
    "2269.HK": "藥明生物", "2359.HK": "藥明康德", "6160.HK": "百濟神州", "1801.HK": "信達生物",
    "9926.HK": "康方生物", "1093.HK": "石藥集團", "1177.HK": "中國生物製藥", "2252.HK": "微創機器人-B",
    "2331.HK": "李寧", "2020.HK": "安踏體育", "9633.HK": "農夫山泉", "9992.HK": "泡泡瑪特",
    "6862.HK": "海底撈", "0291.HK": "華潤啤酒", "2319.HK": "蒙牛乳業", "1929.HK": "周大福",
    "1109.HK": "華潤置地", "0688.HK": "中國海外發展", "1997.HK": "九龍倉置業", "0016.HK": "新鴻基地產",
    "1113.HK": "長實集團", "0001.HK": "長和", "0027.HK": "銀河娛樂", "1928.HK": "金沙中國有限公司",
    "0916.HK": "龍源電力", "0836.HK": "華潤電力", "2380.HK": "中國電力", "1816.HK": "中廣核電力",
    "0902.HK": "華能國際電力", "1072.HK": "東方電氣", "2727.HK": "上海電氣", "1766.HK": "中國中車",
    "0390.HK": "中國中鐵", "1186.HK": "中國鐵建", "1800.HK": "中國交通建設", "1919.HK": "中遠海控",
    "0316.HK": "東方海外國際", "2618.HK": "京東物流", "2057.HK": "中通快遞-W", "6618.HK": "京東健康",
    "1833.HK": "平安好醫生", "0241.HK": "阿里健康", "6060.HK": "眾安在綫", "1359.HK": "中國信達",
    "2800.HK": "盈富基金", "3033.HK": "南方恒生科技", "2828.HK": "恒生中國企業"
}

def get_hk_connect_list():
    """優先嘗試從 akshare 獲取全市場港股通名單，若海外網路受阻則自動切換至核心港股通池"""
    stocks = {}
    source_label = "全市場港股通動態名單"
    
    # 嘗試方法 1：akshare 東方財富港股通
    try:
        df_ggt = ak.stock_hk_ggt_components_em()
        if df_ggt is not None and not df_ggt.empty:
            for _, row in df_ggt.iterrows():
                code = str(row['代碼']).zfill(5)
                yf_code = f"{code[1:]}.HK" if code.startswith("0") else f"{code}.HK"
                stocks[yf_code] = str(row['名稱'])
            if len(stocks) > 50:
                print(f"成功從 akshare 獲取 {len(stocks)} 隻港股通股份")
                return stocks, source_label
    except Exception as e:
        print(f"東方財富接口連接超時 ({e})，切換備援方案...")

    # 嘗試方法 2：使用內建核心港股通名單（100% 穩定）
    print(f"啟用內建核心港股通精選名單（共 {len(CORE_HK_CONNECT)} 隻活躍標的）")
    return CORE_HK_CONNECT, "核心港股通精選池"

def analyze_chan_theory(ticker, name):
    """透過 yfinance 計算日線成交額與纏論結構（分型、中樞區間、買點訊號）"""
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if df.empty or len(df) < 35:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

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

        # 過濾掉成交額低於 1,000 萬港幣的冷門股
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
    stock_dict, source_label = get_hk_connect_list()
    
    results = []
    scanned_count = 0
    for ticker, name in stock_dict.items():
        scanned_count += 1
        res = analyze_chan_theory(ticker, name)
        if res:
            results.append(res)
        time.sleep(0.05)

    df_res = pd.DataFrame(results)
    if not df_res.empty:
        # 按當日成交額由高至低排序
        df_res = df_res.sort_values(by="_turnover", ascending=False).drop(columns=["_turnover"])

    hk_tz = datetime.timezone(datetime.timedelta(hours=8))
    update_time = datetime.datetime.now(hk_tz).strftime("%Y-%m-%d %H:%M:%S")
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
            掃描範圍：{source_label} ({scanned_count} 隻) ｜ 
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
    print(f"掃描完成！共掃描 {scanned_count} 隻，篩選出 {matched_count} 隻符合條件的港股通股份。")

if __name__ == "__main__":
    main()
