import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
import os
import json
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor
import akshare as ak  
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime, timedelta
import sys
from streamlit.web import cli as stcli 

# ==========================================
# 0. 全局配置与环境适配
# ==========================================
st.set_page_config(page_title="指挥官战略终端 v4.3", layout="wide")

def bootstrap():
    if not st.runtime.exists():
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())

is_local = any(os.path.exists(p) for p in ["D:\\vscode", "D:\\code", "D:\\cloud"])
if is_local:
    os.environ['HTTP_PROXY'] = "http://127.0.0.1:7892"
    os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7892"
else:
    os.environ.pop('HTTP_PROXY', None)
    os.environ.pop('HTTPS_PROXY', None)

DB_FILE = "stocks.json"

st.markdown("""
    <style>
    .metric-card { border: 1px solid rgba(128, 128, 128, 0.2); border-radius: 8px; padding: 12px; margin-bottom: 5px; min-height: 110px; }
    .rec-buy { color: #ff4b4b; font-weight: bold; background: rgba(255, 75, 75, 0.1); padding: 5px 10px; border-radius: 4px; border: 1px solid #ff4b4b; }
    .rec-sell { color: #00eb93; font-weight: bold; background: rgba(0, 235, 147, 0.1); padding: 5px 10px; border-radius: 4px; border: 1px solid #00eb93; }
    .rec-hold { color: #ffa500; font-weight: bold; background: rgba(255, 165, 0, 0.1); padding: 5px 10px; border-radius: 4px; border: 1px solid #ffa500; }
    .news-tag { font-size: 0.7em; padding: 2px 6px; border-radius: 10px; margin-right: 5px; color: white; }
    .tag-finance { background-color: #4a90e2; }
    .tag-world { background-color: #e2a14a; }
    .tag-iran { background-color: #d0021b; }
    .guba-post { font-size: 0.9em; padding: 5px 0; border-bottom: 1px dashed rgba(128,128,128,0.2); }
    .guba-post a { color: #dcdcdc; text-decoration: none; transition: 0.3s; }
    .guba-post a:hover { color: #ff4b4b; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. 核心数据与算法函数
# ==========================================

# 🟢 升级版气象雷达：支持 24h 变温与穿搭推演
@st.cache_data(ttl=3600, show_spinner=False)
def get_weather(city_en, city_zh):
    try:
        res = requests.get(f"https://wttr.in/{city_en}?format=j1&lang=zh", timeout=5).json()
        curr = res['current_condition'][0]
        temp = int(curr['temp_C'])
        desc = curr.get('lang_zh', curr.get('weatherDesc', [{'value': '未知'}]))[0]['value']
        
        today = res['weather'][0]
        t_max, t_min = today['maxtempC'], today['mintempC']
        
        if temp >= 28: wear = "短袖夏装 🩳"
        elif 20 <= temp < 28: wear = "长袖或薄外套 🧥"
        elif 10 <= temp < 20: wear = "夹克、风衣或薄毛衣 🧣"
        elif 0 <= temp < 10: wear = "呢子大衣、薄羽绒服 🧤"
        else: wear = "厚羽绒服、防寒内衣 ❄️"
        
        if '雨' in desc or 'rain' in desc.lower() or '淋' in desc: 
            wear += " (记得带伞 ☔)"
            
        return f"**{city_zh}**：{desc} {temp}°C (24h变温: {t_min}°C~{t_max}°C)  \n💡穿搭建议: {wear}"
    except:
        return f"**{city_zh}**：气象卫星连接超时"

@st.cache_data(ttl=300, show_spinner=False)
def get_guba_posts(ticker):
    code = ''.join(filter(str.isdigit, ticker))
    if len(code) != 6: return []
    try:
        url = f"https://guba.eastmoney.com/list,{code}.html"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, proxies={"http": None, "https": None}, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        posts = []
        for a in soup.find_all('a'):
            href = a.get('href', '')
            title = a.get('title') or a.text.strip()
            if '/news,' in href and title and len(title) > 5 and '$' not in title:
                link = href if href.startswith("http") else "https://guba.eastmoney.com" + href
                if title not in [p['t'] for p in posts]: posts.append({"t": title, "l": link})
            if len(posts) >= 6: break
        return posts
    except: return []

# 🟢 终极替换：当日分时图 (使用 push2his 的 1分钟级数据，完美解决基金无数据和中午断层直线问题)
@st.cache_data(ttl=60, show_spinner=False)
def get_intraday_data(ticker):
    try:
        clean_t = str(ticker).upper().replace('.SS', '').replace('.SZ', '')
        is_cn = clean_t.isdigit() and len(clean_t) == 6
        if is_cn:
            market = "1" if clean_t.startswith(('6', '5', '9')) else "0"
            # 💣 核心突防：改用 push2his 接口，klt=1 (1分钟K线)，lmt=240 (A股一天刚好 4小时=240分钟)
            url = f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={market}.{clean_t}&fields1=f1,f2,f3&fields2=f51,f52,f53,f54,f55,f56&klt=1&fqt=1&end=20500101&lmt=240"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'https://quote.eastmoney.com/',
                'Connection': 'close'
            }
            # 强制直连绕过代理
            r = requests.get(url, headers=headers, proxies={"http": None, "https": None}, timeout=8).json()
            
            if r.get('data') and r.get('data').get('klines'):
                parsed_data = []
                for k in r['data']['klines']:
                    parts = k.split(',')
                    if len(parts) >= 3:
                        # 提取时间并截取为 "HH:MM" 的纯文本格式 (完美消除中午 11:30 - 13:00 的大直线)
                        time_str = parts[0][11:16] 
                        # parts[2]是收盘价，连成线就是完美的分时走势
                        parsed_data.append({'Time': time_str, 'Price': float(parts[2])})
                return pd.DataFrame(parsed_data)
        else:
            y_ticker = f"{clean_t}.SS" if (is_cn and clean_t.startswith(('6', '5', '9'))) else f"{clean_t}.SZ" if is_cn else ticker
            df = yf.Ticker(y_ticker).history(period="1d", interval="1m")
            if not df.empty:
                df_res = df[['Close']].reset_index().rename(columns={'Datetime':'Time', 'Close':'Price'})
                df_res['Time'] = df_res['Time'].dt.strftime('%H:%M') 
                return df_res
    except Exception as e:
        print(f"分时图底层数据异常: {e}")
        pass 
    return pd.DataFrame()

# 🟢 终极替换：五日图 (使用 push2his 的 5分钟级数据，完美规避基金无数据 Bug)
@st.cache_data(ttl=60, show_spinner=False)
def get_5d_data(ticker):
    try:
        clean_t = str(ticker).upper().replace('.SS', '').replace('.SZ', '')
        is_cn = clean_t.isdigit() and len(clean_t) == 6
        if is_cn:
            market = "1" if clean_t.startswith(('6', '5', '9')) else "0"
            # 💣 核心突防：改用 push2his 接口，klt=5 (5分钟K线)，lmt=240 (5天的数据量)
            url = f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={market}.{clean_t}&fields1=f1,f2,f3&fields2=f51,f52,f53,f54,f55,f56&klt=5&fqt=1&end=20500101&lmt=240"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'https://quote.eastmoney.com/',
                'Connection': 'close'
            }
            # 强制直连绕过代理
            r = requests.get(url, headers=headers, proxies={"http": None, "https": None}, timeout=8).json()
            
            if r.get('data') and r.get('data').get('klines'):
                parsed_data = []
                for k in r['data']['klines']:
                    parts = k.split(',')
                    if len(parts) >= 3:
                        # 提取时间并截取为 "MM-DD HH:MM" 的纯文本格式 (完美消除周末大直线)
                        time_str = parts[0][5:16] 
                        # parts[2]是收盘价，连成线就是完美的五日走势
                        parsed_data.append({'Time': time_str, 'Price': float(parts[2])})
                return pd.DataFrame(parsed_data)
        else:
            y_ticker = f"{clean_t}.SS" if (is_cn and clean_t.startswith(('6', '5', '9'))) else f"{clean_t}.SZ" if is_cn else ticker
            df = yf.Ticker(y_ticker).history(period="5d", interval="5m")
            if not df.empty:
                df_res = df[['Close']].reset_index().rename(columns={'Datetime':'Time', 'Close':'Price'})
                df_res['Time'] = df_res['Time'].dt.strftime('%m-%d %H:%M') 
                return df_res
    except Exception as e:
        print(f"五日图底层数据异常: {e}") # 如果再崩，终端会打印具体死因
        pass
    return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def get_stock_data(ticker):
    df = pd.DataFrame(); info = {}
    try:
        clean_t = str(ticker).upper().replace('.SS', '').replace('.SZ', '')
        is_cn = clean_t.isdigit() and len(clean_t) == 6
        if is_cn:
            market = "1." if clean_t.startswith(('6', '5', '9')) else "0."
            headers = {'User-Agent': 'Mozilla/5.0'}
            for f_mode in ["1", "0"]:
                url = f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={market}{clean_t}&fields1=f1,f2,f3&fields2=f51,f52,f53,f54,f55,f56&klt=101&fqt={f_mode}&end=20500101&lmt=10000"
                try:
                    r = requests.get(url, headers=headers, proxies={"http": None, "https": None}, timeout=3).json()
                    d = r.get('data')
                    if d and d.get('klines'):
                        info['shortName'] = d.get('name', clean_t)
                        klines = [k.split(',') for k in d['klines']]
                        df = pd.DataFrame(klines, columns=['Date', 'Open', 'Close', 'High', 'Low', 'Volume'])
                        df['Date'] = pd.to_datetime(df['Date']); df.set_index('Date', inplace=True); df = df.astype(float)
                        break
                except: continue
            if df.empty:
                sina_sym = f"sh{clean_t}" if clean_t.startswith(('6', '5', '9')) else f"sz{clean_t}"
                sina_url = f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={sina_sym}&scale=240&ma=no&datalen=10000"
                try:
                    res = requests.get(sina_url, proxies={"http": None, "https": None}, timeout=4).json()
                    if res and len(res) > 0:
                        df = pd.DataFrame(res)
                        df.rename(columns={'day':'Date', 'open':'Open', 'close':'Close', 'high':'High', 'low':'Low', 'volume':'Volume'}, inplace=True)
                        df['Date'] = pd.to_datetime(df['Date']); df.set_index('Date', inplace=True); df = df.astype(float)
                except: pass
        if df.empty:
            y_ticker = f"{clean_t}.SS" if (is_cn and clean_t.startswith(('6', '5', '9'))) else f"{clean_t}.SZ" if is_cn else ticker
            try:
                t_obj = yf.Ticker(y_ticker); df = t_obj.history(period="max"); info = t_obj.info
            except: pass
        if not df.empty and len(df) >= 2:
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            diff = df['Close'].diff()
            up, down = diff.copy(), diff.copy()
            up[up < 0] = 0; down[down > 0] = 0
            roll_up = up.rolling(14).mean(); roll_down = down.abs().rolling(14).mean()
            df['RSI'] = 100.0 - (100.0 / (1.0 + (roll_up / (roll_down + 1e-9))))
        return df, info
    except: return pd.DataFrame(), {}

@st.cache_data(ttl=900, show_spinner=False)
def fetch_intel():
    intel = {"finance": [], "world": [], "iran": []}
    translator = GoogleTranslator(source='auto', target='zh-CN')
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        r = requests.get("https://www.cnbc.com/world-markets/", headers=headers, timeout=5)
        soup = BeautifulSoup(r.text, 'html.parser')
        for l in soup.select(".Card-title")[:5]:
            intel["finance"].append({"t": translator.translate(l.get_text(strip=True)), "l": "https://www.cnbc.com"+l.get('href') if not l.get('href').startswith("http") else l.get('href')})
    except: pass
    sources = ["http://feeds.bbci.co.uk/news/world/rss.xml", "https://www.aljazeera.com/xml/rss/all.xml"]
    iran_kws = ["iran", "middle east", "israel", "hezbollah", "gaza", "伊朗", "中东", "以色列"]
    seen = set()
    for url in sources:
        try:
            r = requests.get(url, headers=headers, timeout=6)
            soup = BeautifulSoup(r.text, 'xml')
            for item in soup.find_all('item')[:6]:
                raw_t = item.title.text.strip()
                if raw_t not in seen:
                    seen.add(raw_t)
                    zh_t = translator.translate(raw_t)
                    link = item.link.text.strip()
                    if any(k in raw_t.lower() or k in zh_t for k in iran_kws):
                        intel["iran"].append({"t": zh_t, "l": link})
                    else:
                        intel["world"].append({"t": zh_t, "l": link})
        except: continue
    return intel

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_dxy_trend():
    try:
        df = ak.futures_foreign_hist(symbol="DX") 
        df.columns = [str(c).lower() for c in df.columns]
        if 'close' in df.columns:
            series = pd.to_numeric(df['close'], errors='coerce').dropna()
            if len(series) >= 5: return (series.iloc[-1] - series.iloc[-5]) / series.iloc[-5] 
    except: pass
    return 0.0

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data_v23(asset_type):
    symbol = "GC" if "金" in asset_type else "CL"
    try:
        df = ak.futures_foreign_hist(symbol=symbol)
        if df is not None and not df.empty:
            df.columns = [str(c).lower() for c in df.columns]
            if 'date' in df.columns and 'close' in df.columns:
                vol_col = [c for c in df.columns if 'vol' in c or 'volume' in c]
                v_col = vol_col[0] if vol_col else 'close' 
                df_clean = df[['date', 'close', v_col]].copy()
                df_clean.columns = ['Date', 'TARGET', 'Volume']
                df_clean['Date'] = pd.to_datetime(df_clean['Date'])
                df_clean['TARGET'] = pd.to_numeric(df_clean['TARGET'], errors='coerce')
                df_clean['Volume'] = pd.to_numeric(df_clean['Volume'], errors='coerce').fillna(0.0)
                if v_col == 'close': df_clean['Volume'] = 1.0 
                df_clean = df_clean.dropna().sort_values('Date').reset_index(drop=True)
                return df_clean
        return pd.DataFrame()
    except: return pd.DataFrame()

def execute_prediction(df, days, manual_score, panic_premium, backtest_days=0):
    if backtest_days > 0:
        df_train = df.iloc[:-backtest_days].tail(150).copy()
        df_truth = df.iloc[-backtest_days:].copy()
    else:
        df_train = df.tail(150).copy()
        df_truth = pd.DataFrame()

    df_train['Ordinal'] = df_train['Date'].map(datetime.toordinal)
    X = df_train[['Ordinal']].values; y = df_train['TARGET'].values
    weights = df_train['Volume'].values
    if np.nansum(weights) <= 0: weights = np.ones_like(weights)
    else:
        weights = weights / (np.nanmean(weights) + 1e-9)
        weights = np.clip(weights, 0.01, None)
    
    poly = PolynomialFeatures(degree=2); X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y, sample_weight=weights)
    train_pred = model.predict(X_poly)
    rmse = np.sqrt(np.mean((y - train_pred)**2)); mae = np.mean(np.abs(y - train_pred))
    
    dxy_change = fetch_dxy_trend()
    last_price = df_train['TARGET'].iloc[-1]; ma60 = df_train['TARGET'].rolling(window=60).mean().iloc[-1]
    value_anchor = df_train['TARGET'].median(); recent_vol = df_train['TARGET'].tail(15).pct_change().std()
    is_gold = last_price > 500 

    last_date = df_train['Date'].max()
    pred_window = backtest_days if backtest_days > 0 else days
    f_dates = [last_date + timedelta(days=i) for i in range(1, pred_window + 1)]
    f_ordinals = np.array([d.toordinal() for d in f_dates]).reshape(-1, 1)
    
    base_path = model.predict(poly.transform(f_ordinals))
    t_path = []; current_sim_price = last_price; np.random.seed(42) 
    
    for i, p in enumerate(base_path):
        step_ratio = (i + 1) / pred_window
        decay = 1 / (1 + np.exp(0.5 * (i - pred_window/2))) 
        tactical_impact = 1 + (manual_score / 10.0 * 0.15 * (1 - decay))
        panic_decay = np.log1p(i + 1) / np.log1p(pred_window)
        panic_impact = 1 + (panic_premium / 100.0 * panic_decay)
        gravity_pull = (value_anchor - p) / p * 0.3 * step_ratio
        dxy_impact = -1 * dxy_change * 0.5 * step_ratio
        m = f_dates[i].month
        season_impact = 0.02 * step_ratio if is_gold and m in [1,2,8,9,12] else (0.03 * step_ratio if not is_gold and m in [5,6,7,8] else 0)
        noise = 1 + np.random.normal(0, recent_vol * 0.5)
        final_p = p * tactical_impact * panic_impact * (1 + gravity_pull + season_impact + dxy_impact) * noise
        hard_floor = df_train['TARGET'].min() * 0.95
        if final_p < hard_floor: final_p = hard_floor + (np.random.random() * 20) 
        t_path.append(final_p); current_sim_price = final_p
    return df_train, df_truth, f_dates, t_path, rmse, mae


# ==========================================
# 2. 页面渲染模块
# ==========================================

def render_strategic_terminal():
    """模块 1：情报大盘终端"""
    with st.sidebar:
        st.header("🛠️ 战区设置")
        new_in = st.text_input("➕ 接入新目标:").strip().upper()
        if st.button("确定接入", use_container_width=True):
            if new_in and new_in not in st.session_state.my_stocks:
                st.session_state.my_stocks.append(new_in)
                with open(DB_FILE, 'w') as f: json.dump(st.session_state.my_stocks, f)
                st.rerun()
        st.divider()
        if st.button("🔥 重置/清空所有", use_container_width=True, type="primary"):
            st.session_state.my_stocks = ["GC=F", "CL=F", "160416", "000001"]
            with open(DB_FILE, 'w') as f: json.dump(st.session_state.my_stocks, f)
            st.rerun()
            
    st.title("🛰️ 全球局势 & 金融命脉战略终端")

    data_dict = {}
    if st.session_state.my_stocks:
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(get_stock_data, st.session_state.my_stocks))
        data_dict = dict(zip(st.session_state.my_stocks, results))

    if st.session_state.my_stocks:
        cols = st.columns(len(st.session_state.my_stocks))
        for i, t in enumerate(st.session_state.my_stocks):
            with cols[i]:
                df_q, info_q = data_dict.get(t, (pd.DataFrame(), {}))
                if not df_q.empty and len(df_q) >= 2:
                    cur, prev = df_q['Close'].iloc[-1], df_q['Close'].iloc[-2]
                    chg = cur - prev
                    color = "#ff4b4b" if chg >= 0 else "#00eb93"
                    st.markdown(f'<div class="metric-card"><div style="font-size:0.8em;color:gray;">{t}</div>'
                                f'<div style="font-size:1.1em;font-weight:bold;">{cur:,.2f}</div>'
                                f'<div style="color:{color};font-size:0.8em;">{"↑" if chg>=0 else "↓"} {abs(chg):.2f}</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="metric-card"><div style="font-size:0.8em;color:gray;">{t}</div>'
                                f'<div style="font-size:1.1em;font-weight:bold;">N/A</div>'
                                f'<div style="color:gray;font-size:0.8em;">数据故障</div></div>', unsafe_allow_html=True)
                
                if st.button("❌ 移除", key=f"del_{t}_{i}", use_container_width=True):
                    st.session_state.my_stocks.remove(t)
                    with open(DB_FILE, 'w') as f: json.dump(st.session_state.my_stocks, f)
                    st.rerun()

    if st.session_state.my_stocks:
        target = st.selectbox("🎯 聚焦研判目标:", options=st.session_state.my_stocks, key="target_selector")
        df_h, info_h = data_dict.get(target, (pd.DataFrame(), {}))
        
        if not df_h.empty and len(df_h) >= 2:
            l, r = st.columns([1, 2.5])
            with l:
                rsi = df_h['RSI'].iloc[-1] if 'RSI' in df_h.columns else 50
                advice = "💎 建议吸筹" if rsi < 30 else "🚀 强势看涨" if df_h['Close'].iloc[-1] > df_h['MA5'].iloc[-1] else "🟡 震荡整固"
                st.write(f"### {info_h.get('shortName', target)}")
                st.markdown(f'<div class="rec-hold">{advice}</div>', unsafe_allow_html=True)
                st.write(f"**RSI:** `{rsi:.2f}` | **现价:** `{df_h['Close'].iloc[-1]:,.2f}`")
                
                clean_target = target.upper().replace('.SS', '').replace('.SZ', '')
                if clean_target.isdigit() and len(clean_target) == 6:
                    st.divider()
                    st.write("🗣️ **东方财富股吧·前沿热议**")
                    guba_posts = get_guba_posts(clean_target)
                    if guba_posts:
                        for p in guba_posts:
                            st.markdown(f'<div class="guba-post">💬 <a href="{p["l"]}" target="_blank">{p["t"]}</a></div>', unsafe_allow_html=True)
                    else: st.caption("暂无最新讨论")

            with r:
                t_k, t_5d, t_intra = st.tabs(["📉 核心K线分析", "📈 五日图", "⏱️ 今日分时"])
                with t_k:
                    k_type = st.radio("时间跨度:", ["日K(1个月)", "日K(1年)", "周K(1年)", "月K(3年)", "最大周K(建仓以来)"], index=0, horizontal=True, label_visibility="collapsed")
                    if "日K" in k_type: df_plot = df_h.tail(22) if "1个月" in k_type else df_h.tail(250)
                    else:
                        rule = 'W-FRI' if '周K' in k_type else 'ME' 
                        df_res = df_h.resample(rule).agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
                        if 'MA5' in df_h.columns:
                            df_res['MA5'] = df_res['Close'].rolling(5).mean(); df_res['MA20'] = df_res['Close'].rolling(20).mean()
                        if "1年" in k_type: df_plot = df_res.tail(52)
                        elif "3年" in k_type: df_plot = df_res.tail(36)
                        else: df_plot = df_res

                    fig = go.Figure(data=[go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], increasing_line_color='#ff4b4b', decreasing_line_color='#00eb93', name="K线")])
                    if 'MA5' in df_plot.columns and 'MA20' in df_plot.columns:
                        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA5'], name="MA5", line=dict(color='#4a90e2', width=1.2)))
                        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA20'], name="MA20", line=dict(color='#ffa500', width=1.2)))
                    
                    if not df_plot.empty:
                        ymin, ymax = df_plot['Low'].min(), df_plot['High'].max()
                        pad = (ymax - ymin) * 0.05 if ymax != ymin else 0.1
                        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=350, template="plotly_dark", xaxis_rangeslider_visible=False, yaxis=dict(range=[ymin - pad, ymax + pad]), dragmode='pan')
                    else: fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=350, template="plotly_dark", xaxis_rangeslider_visible=False, dragmode='pan')
                    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
                
                with t_5d:
                    df_5d = get_5d_data(target)
                    if not df_5d.empty:
                        fig_5d = go.Figure(go.Scatter(x=df_5d['Time'], y=df_5d['Price'], mode='lines', line=dict(color='#4a90e2', width=2)))
                        ymin, ymax = df_5d['Price'].min(), df_5d['Price'].max()
                        pad = (ymax - ymin) * 0.05 if ymax != ymin else 0.1
                        fig_5d.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=350, template="plotly_dark", yaxis=dict(range=[ymin - pad, ymax + pad]), dragmode='pan')
                        st.plotly_chart(fig_5d, use_container_width=True, config={'scrollZoom': True})
                    else: st.info("当前暂无五日图信号")

                with t_intra:
                    df_intra = get_intraday_data(target)
                    if not df_intra.empty:
                        fig_intra = go.Figure(go.Scatter(x=df_intra['Time'], y=df_intra['Price'], mode='lines', line=dict(color='#00eb93', width=2)))
                        ymin, ymax = df_intra['Price'].min(), df_intra['Price'].max()
                        pad = (ymax - ymin) * 0.05 if ymax != ymin else 0.1
                        fig_intra.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=350, template="plotly_dark", yaxis=dict(range=[ymin - pad, ymax + pad]), dragmode='pan')
                        st.plotly_chart(fig_intra, use_container_width=True, config={'scrollZoom': True})
                    else: st.info("当前时间段暂无分时信号")

    st.divider()
    st.subheader("📡 综合情报汇总")
    t1, t2, t3 = st.tabs(["📊 金融命脉", "🌐 全球热点", "🇮🇷 中东战区"])
    intel = fetch_intel()
    def render_news(items, cls_name, tag):
        if not items: st.write("🛰️ 暂无最新信号")
        for it in items:
            with st.expander(f"📌 {it['t'][:50]}..."):
                st.markdown(f'<span class="news-tag {cls_name}">{tag}</span> <b>{it["t"]}</b>', unsafe_allow_html=True)
                st.link_button("原文报道", it['l'])
    with t1: render_news(intel["finance"], "tag-finance", "FINANCE")
    with t2: render_news(intel["world"], "tag-world", "GLOBAL")
    with t3: render_news(intel["iran"], "tag-iran", "MIDDLE-EAST")


def render_commodity_quant():
    """模块 2：金油量化推演"""
    st.title("🛢️ 大宗商品深度量化推演")
    with st.sidebar:
        st.header("🎯 量化参数控制")
        mode = st.radio("监控品种:", ["📀 国际黄金 (COMEX)", "🛢️ WTI 原油 (NYMEX)"])
        st.divider()
        st.header("🕰️ 时光机 (测试模式)")
        enable_backtest = st.toggle("开启样本外回测", key="cmd_bt")
        backtest_days = st.slider("倒退天数", 5, 60, 20, key="cmd_bd") if enable_backtest else 0
        
        auto_window = st.toggle("🤖 开启 AI 智能动态周期", key="cmd_aw") if not enable_backtest else False
        if enable_backtest: p_window = 0
        elif auto_window: 
            p_window = 7 
            st.info("已开启：由AI根据系统熵值自动判定推演视野")
        else: p_window = st.slider("前瞻预测天数", 1, 30, 7, key="cmd_pw")
            
        st.divider()
        st.header("🕹️ 指挥官决策")
        m_score = st.slider("常规战术修正", -10.0, 10.0, 0.0, step=0.5, key="cmd_ms")
        panic_val = st.slider("🌋 黑天鹅恐慌溢价 (%)", -100, 100, 0, step=5, key="cmd_pv")

    df_main = fetch_data_v23(mode)
    
    if auto_window and not enable_backtest and not df_main.empty:
        recent_vol = df_main['TARGET'].tail(15).pct_change().std()
        if np.isnan(recent_vol) or recent_vol == 0: recent_vol = 0.01
        p_window = int(max(3, min(25, 0.12 / recent_vol)))
        st.toast(f"🤖 市场波动率: {recent_vol*100:.2f}% | AI 已锁定最佳推演视野为 {p_window} 天")

    _render_quant_ui(df_main, p_window, m_score, panic_val, backtest_days, enable_backtest)


def render_stock_quant():
    """模块 3：股票深度量化推演"""
    st.title("📈 股票深度量化推演")
    
    stock_options = [s for s in st.session_state.my_stocks if s not in ["GC=F", "CL=F"]]
    if not stock_options:
        st.warning("⚠️ 你的关注列表里没有股票数据！请先在【全球情报监控】页面添加股票代码（如 000001, AAPL）。")
        return

    with st.sidebar:
        st.header("🎯 目标股票锁定")
        target_stock = st.selectbox("选择要进行量化推演的股票:", stock_options)
        st.divider()
        st.header("🕰️ 时光机 (测试模式)")
        enable_backtest = st.toggle("开启样本外回测", key="stk_bt")
        backtest_days = st.slider("倒退天数", 5, 60, 20, key="stk_bd") if enable_backtest else 0
        
        auto_window = st.toggle("🤖 开启 AI 智能动态周期", key="stk_aw") if not enable_backtest else False
        if enable_backtest: p_window = 0
        elif auto_window: 
            p_window = 7
            st.info("已开启：由AI根据系统熵值自动判定推演视野")
        else: p_window = st.slider("前瞻预测天数", 1, 30, 7, key="stk_pw")
            
        st.divider()
        st.header("🕹️ 指挥官决策")
        m_score = st.slider("常规战术修正", -10.0, 10.0, 0.0, step=0.5, key="stk_ms")
        panic_val = st.slider("🌋 黑天鹅恐慌溢价 (%)", -100, 100, 0, step=5, key="stk_pv")

    with st.spinner(f"正在抽取 {target_stock} 的历史 K 线与成交量特征..."):
        raw_df, _ = get_stock_data(target_stock)
        
    if not raw_df.empty and len(raw_df) > 60:
        df_main = raw_df.reset_index()
        if 'Close' in df_main.columns:
            df_main = df_main.rename(columns={'Close': 'TARGET'})
        if 'Volume' not in df_main.columns:
            df_main['Volume'] = 1.0
        
        df_main['Date'] = pd.to_datetime(df_main['Date'])
        df_main = df_main.sort_values('Date').reset_index(drop=True)
        
        if auto_window and not enable_backtest:
            recent_vol = df_main['TARGET'].tail(15).pct_change().std()
            if np.isnan(recent_vol) or recent_vol == 0: recent_vol = 0.01
            p_window = int(max(3, min(25, 0.12 / recent_vol)))
            st.toast(f"🤖 个股活跃度: {recent_vol*100:.2f}% | AI 已锁定最佳推演视野为 {p_window} 天")
        
        _render_quant_ui(df_main, p_window, m_score, panic_val, backtest_days, enable_backtest)
    else:
        st.error("获取的股票历史数据不足（少于60天），无法启动量化引擎！")


# ==========================================
# UI 绘图渲染复用工具
# ==========================================
def _render_quant_ui(df_main, p_window, m_score, panic_val, backtest_days, enable_backtest):
    if not df_main.empty:
        df_main['MA5'] = df_main['TARGET'].rolling(window=5).mean()
        df_main['MA20'] = df_main['TARGET'].rolling(window=20).mean()
        df_main['STD20'] = df_main['TARGET'].rolling(window=20).std()
        df_main['BB_UP'] = df_main['MA20'] + 2 * df_main['STD20']
        df_main['BB_DN'] = df_main['MA20'] - 2 * df_main['STD20']

        df_train, df_truth, f_dates, t_path, rmse, mae = execute_prediction(df_main, p_window, m_score, panic_val, backtest_days)
        
        hist_tail = df_train['TARGET'].tail(19).tolist()
        sim_prices = hist_tail + t_path
        sim_series = pd.Series(sim_prices)
        sim_ma20 = sim_series.rolling(20).mean().dropna().tolist()
        sim_std20 = sim_series.rolling(20).std().dropna().tolist()
        sim_bb_up = [m + 2*s for m, s in zip(sim_ma20, sim_std20)]
        sim_bb_dn = [m - 2*s for m, s in zip(sim_ma20, sim_std20)]

        c1, c2, c3, c4 = st.columns(4)
        if enable_backtest:
            actual_end_p = df_truth['TARGET'].iloc[-1]; pred_end_p = t_path[-1]
            err = (pred_end_p - actual_end_p) / actual_end_p * 100
            c1.metric("现实真相价", f"{actual_end_p:.2f}")
            c2.metric("预测终点价", f"{pred_end_p:.2f}", f"误差 {err:+.2f}%")
        else:
            cur_p = df_train['TARGET'].iloc[-1]
            c1.metric("今日现价", f"{cur_p:.2f}")
            c2.metric("前瞻预测价", f"{t_path[-1]:.2f}", f"{(t_path[-1]-cur_p):+.2f}")
        
        c3.metric("拟合方差 (RMSE)", f"{rmse:.2f}")
        c4.metric("绝对误差 (MAE)", f"{mae:.2f}")

        fig = go.Figure()
        df_show = df_train.tail(100)
        
        fig.add_trace(go.Scatter(x=df_show['Date'], y=df_show['BB_UP'], line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=df_show['Date'], y=df_show['BB_DN'], fill='tonexty', fillcolor='rgba(155, 89, 182, 0.15)', line=dict(width=0), name="历史置信带(2σ)"))

        f_px = [df_show['Date'].iloc[-1]] + f_dates
        f_bb_up = [df_show['BB_UP'].iloc[-1]] + sim_bb_up
        f_bb_dn = [df_show['BB_DN'].iloc[-1]] + sim_bb_dn
        
        fig.add_trace(go.Scatter(x=f_px, y=f_bb_up, line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=f_px, y=f_bb_dn, fill='tonexty', fillcolor='rgba(255, 75, 75, 0.15)', line=dict(width=0), name="预测推演云团(2σ)"))

        fig.add_trace(go.Scatter(x=df_show['Date'], y=df_show['TARGET'], name="收盘价", line=dict(color='#4A90E2', width=2)))
        fig.add_trace(go.Scatter(x=df_show['Date'], y=df_show['MA5'], name="MA5", line=dict(color='#E2A14A', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=df_show['Date'], y=df_show['MA20'], name="MA20", line=dict(color='#9B59B6', width=1.5)))

        if enable_backtest and not df_truth.empty:
            fig.add_trace(go.Scatter(x=[df_show['Date'].iloc[-1]] + list(df_truth['Date']), y=[df_show['TARGET'].iloc[-1]] + list(df_truth['TARGET']), name="现实真相", line=dict(color='#00FA9A', width=3)))

        fig.add_trace(go.Scatter(x=f_px, y=[df_show['TARGET'].iloc[-1]] + list(t_path), name="量化推演路径", line=dict(color='#ff4b4b', width=3, dash='dash')))

        fig.update_layout(template="plotly_dark", height=600, margin=dict(l=0,r=0,t=20,b=0), dragmode='pan', hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})


# ==========================================
# 3. 中枢神经 (页面路由与启动)
# ==========================================

def main():
    bootstrap()
    
    if 'my_stocks' not in st.session_state:
        if os.path.exists(DB_FILE):
            try:
                with open(DB_FILE, 'r') as f: st.session_state.my_stocks = json.load(f)
            except: st.session_state.my_stocks = ["GC=F", "CL=F", "160416", "000001"]
        else: st.session_state.my_stocks = ["GC=F", "CL=F", "160416", "000001"]

    with st.sidebar:
        st.title("🎮 控制中心")
        
        # 🟢 渲染升级后的两地天气模块
        st.info(get_weather('Chengdu', '成都')) 
        st.info(get_weather('Mianyang', '绵阳')) 
        
        st.divider()
        menu = st.radio("切换任务模块:", ["🌐 全球情报监控", "📀 金油量化推演", "📈 股票量化推演"], index=0)
        st.divider()

    if menu == "🌐 全球情报监控":
        render_strategic_terminal()
    elif menu == "📀 金油量化推演":
        render_commodity_quant()
    else:
        render_stock_quant()

if __name__ == "__main__":
    main()