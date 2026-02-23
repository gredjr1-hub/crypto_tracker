import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import base64
import os
import csv
import requests
from datetime import datetime

# --- SESSION STATE INITIALIZATION ---
if 'startup_sound_played' not in st.session_state:
    st.session_state.startup_sound_played = False
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'crypto_portfolio' not in st.session_state:
    st.session_state.crypto_portfolio = {}

st.set_page_config(page_title="Crypto Quant Command Center", layout="wide", page_icon="₿")

# --- PROJECT PROMISE & UTILITY TIERS ---
CRYPTO_TIERS = {
    'BTC': 1, 'ETH': 1, 
    'SOL': 2, 'ADA': 2, 'AVAX': 2, 'DOT': 2, 'LINK': 2, 'MATIC': 2, 'NEAR': 2, 'APT': 2, 'OP': 2, 'INJ': 2, 'XRP': 2, 'BNB': 2, 'TRX': 2, 'LTC': 2, 
    'DOGE': 3, 'SHIB': 3, 'PEPE': 3, 'FLOKI': 3, 'BONK': 3, 'WIF': 3 
}

# --- DATA LOADERS ---
@st.cache_data(ttl=60)
def load_score_history():
    if os.path.exists("historical_crypto_scores.csv"):
        try:
            df = pd.read_csv("historical_crypto_scores.csv")
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            return df
        except Exception: pass
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_fear_and_greed():
    try:
        r = requests.get('https://api.alternative.me/fng/?limit=1')
        data = r.json()
        val = int(data['data'][0]['value'])
        classification = data['data'][0]['value_classification']
        return val, classification
    except:
        return 50, "Neutral"

fng_val, fng_class = get_fear_and_greed()

# --- ALGORITHMIC HELPER FUNCTIONS ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series):
    exp1 = series.ewm(span=12, adjust=False).mean()
    exp2 = series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_bbands(series, window=20, num_std=2):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper = rolling_mean + (rolling_std * num_std)
    lower = rolling_mean - (rolling_std * num_std)
    return upper, lower

def calculate_obv(close, volume):
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv

def log_scores_to_csv(portfolio_data):
    filename = "historical_crypto_scores.csv"
    today_str = datetime.today().strftime('%Y-%m-%d')
    file_exists = os.path.isfile(filename)
    existing_records = set()
    
    if file_exists:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader: existing_records.add((row['Date'], row['Ticker']))
        except Exception: pass
            
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        fieldnames = ['Date', 'Ticker', 'Price', 'Score', 'Decision', 'Risk_Pts', 'Drawdown', 'RSI', 'Vol']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists: writer.writeheader()
        for coin in portfolio_data:
            if (today_str, coin['Ticker']) not in existing_records:
                writer.writerow({'Date': today_str, 'Ticker': coin['Ticker'], 'Price': round(coin['Price'], 4), 'Score': coin['Score'], 'Decision': coin['Decision'], 'Risk_Pts': coin['Risk_Pts'], 'Drawdown': coin.get('Drawdown', 'N/A'), 'RSI': coin.get('RSI', 'N/A'), 'Vol': coin.get('Vol', 'N/A')})

# --- SIDEBAR & MANUAL ENTRY ---
st.sidebar.header("👝 Manage Crypto Portfolio")
st.sidebar.markdown("Enter your coins manually below.")

with st.sidebar.form("add_crypto_form"):
    new_coin = st.text_input("Coin Symbol (e.g., BTC)").strip().upper()
    new_qty = st.number_input("Quantity Owned", min_value=0.0, format="%.6f")
    new_avg = st.number_input("Average Cost ($)", min_value=0.0, format="%.4f")
    submit_add = st.form_submit_button("➕ Add to Portfolio")

    if submit_add and new_coin:
        ticker = f"{new_coin}-USD" if not new_coin.endswith("-USD") else new_coin
        if ticker in st.session_state.crypto_portfolio:
            prev_shares = st.session_state.crypto_portfolio[ticker]['shares']
            prev_avg = st.session_state.crypto_portfolio[ticker]['avg_price']
            new_total_shares = prev_shares + new_qty
            new_total_avg = ((prev_shares * prev_avg) + (new_qty * new_avg)) / new_total_shares if new_total_shares > 0 else 0
            st.session_state.crypto_portfolio[ticker] = {'shares': new_total_shares, 'avg_price': new_total_avg}
        else:
            st.session_state.crypto_portfolio[ticker] = {'shares': new_qty, 'avg_price': new_avg}
        st.rerun()

st.sidebar.divider()
hide_dollars = st.sidebar.toggle("🙈 Hide Dollar Values", value=False)

if st.session_state.crypto_portfolio:
    st.sidebar.markdown("**Current Holdings:**")
    for tkr, data in list(st.session_state.crypto_portfolio.items()):
        colA, colB = st.sidebar.columns([3, 1])
        with colA: st.write(f"**{tkr.replace('-USD','')}**: {data['shares']:g}")
        with colB:
            if st.button("❌", key=f"del_{tkr}"):
                del st.session_state.crypto_portfolio[tkr]
                st.rerun()

# --- QUANT DATA FETCHING ENGINE ---
timeframes = {'1M': 30, '3M': 90, '6M': 180, '1Y': 365, '5Y': 1825}

@st.cache_data(ttl=900) 
def get_crypto_data(port_dict, global_fng_val):
    if not port_dict: return [], {}, 0 
    
    portfolio_data = []
    all_histories = {} 
    total_value = 0

    for ticker, data in port_dict.items():
        yf_ticker = f"{ticker}-USD" if not ticker.endswith("-USD") else ticker
        display_ticker = ticker.replace('-USD', '')
        
        shares, avg_price = data.get('shares', 0), data.get('avg_price', 0)
        coin = yf.Ticker(yf_ticker)
        
        hist = coin.history(period='max')
        if hist.empty: continue
            
        current_price = hist['Close'].iloc[-1]
        hist.index = hist.index.tz_localize(None)
        all_histories[ticker] = hist
            
        volatility, drawdown, ath = 0.0, 0.0, 0.0
        rsi_14, macd_val, sig_val, bb_upper, bb_lower = 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'
        sma_50, sma_200 = 'N/A', 'N/A'
        obv_trend = "Neutral"
        
        if not hist.empty:
            ath = hist['Close'].max()
            drawdown = ((current_price - ath) / ath) * 100 if ath > 0 else 0
            
            hist['200_WMA'] = hist['Close'].rolling(window=1400).mean()
            hist['50_SMA'] = hist['Close'].rolling(window=50).mean()
            hist['200_SMA'] = hist['Close'].rolling(window=200).mean()
            
            hist['OBV'] = calculate_obv(hist['Close'], hist['Volume'])
            hist['OBV_SMA_20'] = hist['OBV'].rolling(window=20).mean()
            
            sma_50 = hist['50_SMA'].iloc[-1] if not pd.isna(hist['50_SMA'].iloc[-1]) else 0
            sma_200 = hist['200_SMA'].iloc[-1] if not pd.isna(hist['200_SMA'].iloc[-1]) else 0
            
            if not pd.isna(hist['OBV'].iloc[-1]) and not pd.isna(hist['OBV_SMA_20'].iloc[-1]):
                obv_trend = "Accumulating" if hist['OBV'].iloc[-1] > hist['OBV_SMA_20'].iloc[-1] else "Distributing"
            
            if len(hist) >= 365:
                volatility = hist['Close'].tail(365).pct_change().std() * np.sqrt(365) * 100
            
            if len(hist) > 50:
                rsi_series = calculate_rsi(hist['Close'])
                macd_line, signal_line = calculate_macd(hist['Close'])
                upper_b, lower_b = calculate_bbands(hist['Close'])
                
                rsi_14 = round(rsi_series.iloc[-1], 2)
                macd_val, sig_val = macd_line.iloc[-1], signal_line.iloc[-1]
                bb_upper, bb_lower = upper_b.iloc[-1], lower_b.iloc[-1]

        # --- THE CONTINUOUS SPECTRUM CRYPTO ALGORITHM ---
        score = 50.0 
        risk_points = 0
        
        tier = CRYPTO_TIERS.get(display_ticker, 4)
        tier_str = "Bluechip" if tier == 1 else "Utility/L1" if tier == 2 else "Meme" if tier == 3 else "Speculative/Alt"
        breakdown = [f"**Base Score:** 50 pts", f"🧬 **Project Classification:** Tier {tier} ({tier_str})"]
        
        if tier == 4: risk_points += 1 
        if tier == 3: risk_points += 1 

        # 1. Historical Drawdown (Continuous Scaling)
        dd_abs = abs(drawdown)
        if tier <= 2:
            if dd_abs <= 10:
                dd_pts = 0
                breakdown.append(f"❌ **Drawdown ({drawdown:.1f}%):** +0 pts (Near ATH / FOMO Zone)")
            elif dd_abs <= 40:
                dd_pts = (dd_abs - 10) * (10.0 / 30.0) 
                score += dd_pts
                breakdown.append(f"✅ **Drawdown ({drawdown:.1f}%):** +{dd_pts:.1f} pts (Mild Discount)")
            elif dd_abs <= 80:
                dd_pts = 10.0 + ((dd_abs - 40.0) * (20.0 / 40.0)) 
                score += dd_pts
                breakdown.append(f"✅ **Drawdown ({drawdown:.1f}%):** +{dd_pts:.1f} pts (Deep Value Zone)")
            else:
                dd_pts = 30
                score += dd_pts
                breakdown.append(f"✅ **Drawdown ({drawdown:.1f}%):** +30 pts (Maximum Capitulation)")
        elif tier == 3: 
            if dd_abs <= 20: dd_pts = 0
            elif dd_abs <= 60: dd_pts = 10
            else: dd_pts = 15 
            score += dd_pts
            breakdown.append(f"⚠️ **Meme Drawdown ({drawdown:.1f}%):** +{dd_pts} pts (Risk Capped)")
        else: 
            if dd_abs > 80:
                dd_pts = -20
                score += dd_pts; risk_points += 2
                breakdown.append(f"🚨 **Dead Altcoin Risk ({drawdown:.1f}%):** -20 pts (Likely abandoned) [+2 Risk]")
            else:
                breakdown.append(f"➖ **Altcoin Drawdown ({drawdown:.1f}%):** 0 pts")

        # 2. Distance from 200 SMA (Continuous Mapping)
        if sma_200 != 0 and current_price > 0:
            sma_dist = ((current_price - sma_200) / sma_200) * 100
            if sma_dist > 60:
                sma_pts = -15
                breakdown.append(f"❌ **SMA Ext ({sma_dist:+.1f}%):** {sma_pts} pts (Severely Overextended)")
            elif sma_dist > 20:
                sma_pts = -5
                breakdown.append(f"⚠️ **SMA Ext ({sma_dist:+.1f}%):** {sma_pts} pts (Cooling Off Needed)")
            elif sma_dist >= 0:
                sma_pts = 15.0 - (sma_dist * 0.75) 
                breakdown.append(f"✅ **SMA Ext ({sma_dist:+.1f}%):** +{sma_pts:.1f} pts (Fresh Breakout / Support)")
            elif sma_dist >= -20:
                sma_pts = 10.0 + (sma_dist * 0.5) 
                breakdown.append(f"✅ **SMA Ext ({sma_dist:+.1f}%):** +{sma_pts:.1f} pts (Accumulating just under trend)")
            else:
                sma_pts = -10
                breakdown.append(f"❌ **SMA Ext ({sma_dist:+.1f}%):** {sma_pts} pts (Macro Breakdown)")
            score += sma_pts

        if sma_50 > 0 and sma_200 > 0:
            if sma_50 > sma_200:
                score += 5; breakdown.append("✅ **Golden Cross:** +5 pts")
            else:
                score -= 5; breakdown.append("❌ **Death Cross:** -5 pts")

        # 3. RSI Momentum (Tiered Spectrum)
        if isinstance(rsi_14, (float, int)):
            if rsi_14 < 30: rsi_pts = 15
            elif rsi_14 < 40: rsi_pts = 10
            elif rsi_14 < 55: rsi_pts = 5
            elif rsi_14 < 65: rsi_pts = 0
            elif rsi_14 < 75: rsi_pts = -10
            else: rsi_pts = -20
            score += rsi_pts
            breakdown.append(f"{'✅' if rsi_pts > 0 else '❌' if rsi_pts < 0 else '➖'} **RSI ({rsi_14}):** {rsi_pts:+} pts")

        # 4. Bollinger Bands Positioning (Spectrum)
        if isinstance(bb_upper, (float, int)) and isinstance(bb_lower, (float, int)) and bb_upper != bb_lower:
            bb_pos = ((current_price - bb_lower) / (bb_upper - bb_lower)) * 100
            if bb_pos < 0: bb_pts = 10
            elif bb_pos < 20: bb_pts = 5
            elif bb_pos < 80: bb_pts = 0
            elif bb_pos <= 100: bb_pts = -5
            else: bb_pts = -10
            score += bb_pts
            breakdown.append(f"{'✅' if bb_pts > 0 else '❌' if bb_pts < 0 else '➖'} **BB Position ({bb_pos:.0f}%):** {bb_pts:+} pts")

        # 5. OBV & MACD
        if obv_trend == "Accumulating":
            score += 5; breakdown.append(f"🐋 **OBV:** +5 pts (Whale Accumulation)")
        elif obv_trend == "Distributing":
            score -= 5; breakdown.append(f"🚨 **OBV:** -5 pts (Whale Distribution)")

        if isinstance(macd_val, (float, int)) and isinstance(sig_val, (float, int)):
            if macd_val > sig_val: score += 5; breakdown.append("✅ **MACD:** +5 pts (Bullish)")
            else: score -= 5; breakdown.append("❌ **MACD:** -5 pts (Bearish)")

        # 6. Global Sentiment Modifier (Continuous Contrarian Rule)
        fng_adj = (50 - global_fng_val) * 0.2 
        score += fng_adj
        breakdown.append(f"{'✅' if fng_adj > 0 else '❌' if fng_adj < 0 else '➖'} **F&G Contrarian ({global_fng_val}):** {fng_adj:+.1f} pts")

        if volatility > 100: risk_points += 2; breakdown.append("⚠️ **Extreme Volatility (>100%):** [+2 Risk]")
        elif volatility < 40: risk_points -= 1; breakdown.append("🛡️ **Low Volatility (<40%):** [-1 Risk]")

        score = max(0, min(100, int(score))) 
        breakdown.append(f"---\n🎯 **Holistic Quant Score: {score}/100**")
        
        if score >= 80: decision, d_color = "ACCUMULATE HEAVILY 🟩", "#28a745"
        elif score >= 60: decision, d_color = "DCA / HOLD 🟨", "#17a2b8"
        elif score >= 40: decision, d_color = "HOLD ⬜", "#6c757d"
        elif score >= 30: decision, d_color = "TRIM PROFITS 🟧", "#fd7e14"
        else: decision, d_color = "SELL / AVOID 🟥", "#dc3545"
            
        if risk_points >= 3: risk_lvl, r_color = "HIGH ⚠️", "red"
        elif risk_points <= 0: risk_lvl, r_color = "LOW 🛡️", "green"
        else: risk_lvl, r_color = "MODERATE ⚖️", "orange"

        val = current_price * shares
        total_value += val
        
        portfolio_data.append({
            'Ticker': display_ticker, 'Val': val, 'Price': current_price, 'Shares': shares, 'Avg': avg_price,
            'Drawdown': drawdown, 'ATH': ath, 'SMA_50': sma_50, 'SMA_200': sma_200,
            'RSI': rsi_14, 'MACD': macd_val, 'MACD_Sig': sig_val, 'Vol': volatility, 'Tier': tier, 'OBV': obv_trend,
            'Score': int(score), 'Decision': decision, 'D_Color': d_color, 'Risk': risk_lvl, 'R_Color': r_color, 'Risk_Pts': risk_points,
            'Upper_BB': bb_upper, 'Lower_BB': bb_lower,
            'Breakdown': breakdown
        })

    log_scores_to_csv(portfolio_data)
    return portfolio_data, all_histories, total_value

# --- UI HELPER: SCORE CARD COMPONENT ---
def render_score_card(coin, today_date, score_history=None, is_watchlist=False, hide_dollars=False):
    """Reusable function to draw just the analysis score card without charts."""
    ticker = coin['Ticker']
    
    signal_tooltips = {
        "ACCUMULATE HEAVILY 🟩": "Whales are buying, macro trend is safe, and the asset is deeply discounted. Deploy capital.",
        "DCA / HOLD 🟨": "Healthy metrics but not a generational buy. Scale in slowly.",
        "HOLD ⬜": "Metrics are contradicting each other. Wait for a clearer setup.",
        "TRIM PROFITS 🟧": "Market is greedy, RSI is overbought, or whales are starting to distribute. Take chips off the table.",
        "SELL / AVOID 🟥": "Death Cross, high volatility, and smart money is fleeing. Avoid completely."
    }
    
    title_col, btn_col = st.columns([3, 1])
    with title_col: 
        st.markdown(f"### **{ticker}** <span style='font-size: 14px; color: gray;'>(Tier {coin['Tier']})</span>", unsafe_allow_html=True)
        
        # --- WHALE INSPECTOR HUB ---
        arkham_url = f"https://platform.arkhamintelligence.com/explorer/token/{ticker.lower()}"
        dune_url = f"https://dune.com/search?q={ticker}&time_range=all"
        
        st.markdown(
            f"""
            <div style="display: flex; gap: 10px; font-size: 13px; margin-bottom: 10px;">
                <a href='https://finance.yahoo.com/quote/{ticker}-USD' target='_blank' style='text-decoration: none;'>📈 Chart</a>
                <a href='{arkham_url}' target='_blank' style='text-decoration: none;'>🐋 Arkham</a>
                <a href='{dune_url}' target='_blank' style='text-decoration: none;'>📊 Dune</a>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
    with btn_col:
        if is_watchlist:
            if st.button("❌", key=f"remove_{ticker}"):
                st.session_state.watchlist.remove(ticker)
                st.rerun()

    trend_html = ""
    if score_history is not None and not score_history.empty:
        t_scores = score_history[score_history['Ticker'] == ticker].copy()
        if not t_scores.empty:
            ninety_days_ago = today_date - timedelta(days=90)
            t_scores_quarter = t_scores[t_scores['Date'] >= ninety_days_ago].sort_values('Date')
            if not t_scores_quarter.empty:
                oldest_score = int(t_scores_quarter.iloc[0]['Score'])
                current_score = int(coin['Score'])
                diff = current_score - oldest_score
                if diff > 0: trend_html = f" <span style='color:#2ca02c; font-size:13px;'><b>⬆️ (+{diff})</b></span>"
                elif diff < 0: trend_html = f" <span style='color:#d62728; font-size:13px;'><b>⬇️ ({diff})</b></span>"
                elif len(t_scores_quarter) > 1: trend_html = f" <span style='color:gray; font-size:13px;'><b>➖</b></span>"

    hover_text = signal_tooltips.get(coin['Decision'], "Quant Engine Signal")

    st.markdown(
        f"<div title='{hover_text}' style='border:1px solid {coin['D_Color']}; padding: 10px; border-radius: 5px; margin-bottom: 5px; cursor: help;'>"
        f"<h4 style='margin:0; color:{coin['D_Color']};'>Signal: {coin['Decision']}</h4>"
        f"<p style='margin:0; font-size:14px;'>Crypto Score: <b>{coin['Score']}/100</b>{trend_html} | Risk: <span style='color:{coin['R_Color']};'><b>{coin['Risk']}</b></span></p>"
        f"</div>", unsafe_allow_html=True
    )
    
    with st.popover("📊 Score Breakdown", use_container_width=True):
        st.markdown(f"### {ticker} Algorithmic Breakdown")
        for line in coin.get('Breakdown', []):
            st.write(line)
    
    sub1, sub2 = st.columns(2)
    with sub1:
        price_format = "${:.6f}" if coin.get('Price', 0.0) < 1 else "${:,.2f}"
        st.write(f"**Price:** {price_format.format(coin.get('Price', 0.0))}")
        dd = coin.get('Drawdown', 0)
        ath = coin.get('ATH', 0)
        st.write(f"**ATH:** {price_format.format(ath)}")
        dd_color = "green" if dd < -60 else ("orange" if dd < -30 else "red")
        st.markdown(f"**Drawdown:** :{dd_color}[{dd:.1f}%]")
        
    with sub2:
        st.write(f"**RSI:** {coin['RSI']}")
        
        obv_color = "green" if coin['OBV'] == "Accumulating" else "red"
        st.markdown(f"**Whale Vol:** :{obv_color}[{coin['OBV']}]")
        
        sma50, sma200 = coin.get('SMA_50', 0), coin.get('SMA_200', 0)
        cross = "Golden" if sma50 > sma200 else "Death"
        cross_color = "green" if cross == "Golden" else "red"
        st.markdown(f"**Trend:** :{cross_color}[{cross} Cross]")

    is_search_or_watch = coin['Shares'] == 0 
    if not is_search_or_watch:
        ret = ((coin['Price'] - coin['Avg']) / coin['Avg']) * 100 if coin['Avg'] > 0 else 0
        avg_str = "$••••" if hide_dollars else (f"${coin['Avg']:.6f}" if coin['Avg'] < 1 else f"${coin['Avg']:.2f}")
        val_str = "$••••" if hide_dollars else f"${coin['Val']:,.0f}"
        ret_color = "#2ca02c" if ret >= 0 else "#d62728"
        
        html_string = (
            f"<div style='font-size: 15px; margin-top: 5px; margin-bottom: 5px;'>"
            f"<b>My Return:</b> <span style='color:{ret_color}; font-weight:bold;'>{ret:+.2f}%</span> &nbsp;|&nbsp; "
            f"<b>Avg Cost:</b> {avg_str} &nbsp;|&nbsp; "
            f"<b>Value:</b> {val_str}"
            f"</div>"
        )
        st.markdown(html_string, unsafe_allow_html=True)

# --- UI HELPER: FULL ROW WITH CHARTS ---
def draw_crypto_row(coin, histories, today_date, is_watchlist=False, hide_dollars=False, score_history=None):
    cols = st.columns([1.6, 1, 1, 1, 1, 1]) 
    
    with cols[0]:
        render_score_card(coin, today_date, score_history, is_watchlist, hide_dollars)
            
    ticker = coin['Ticker']
    yf_ticker = f"{ticker}-USD" if not ticker.endswith("-USD") else ticker
    master_hist = histories.get(yf_ticker, histories.get(ticker))
    
    if master_hist is not None and not master_hist.empty:
        if score_history is not None and not score_history.empty:
            t_scores = score_history[score_history['Ticker'] == ticker].copy()
            if not t_scores.empty:
                t_scores.set_index('Date', inplace=True)
                t_scores = t_scores[~t_scores.index.duplicated(keep='last')]
                master_hist = master_hist.join(t_scores['Score'], how='left')
                master_hist['Score'] = master_hist['Score'].ffill()
                
        if len(master_hist) > 20:
            master_hist['BB_Upper'], master_hist['BB_Lower'] = calculate_bbands(master_hist['Close'])

        for i, (tf_label, days_back) in enumerate(timeframes.items()):
            with cols[i+1]:
                start_date = today_date - timedelta(days=days_back)
                sliced_hist = master_hist[master_hist.index >= start_date]
                if not sliced_hist.empty:
                    start_p = sliced_hist['Close'].iloc[0]
                    end_p = sliced_hist['Close'].iloc[-1]
                    line_color = '#2ca02c' if end_p >= start_p else '#d62728'
                    
                    tf_ret = ((end_p - start_p) / start_p) * 100 if start_p > 0 else 0
                    header_text = f"{tf_label} <span style='color:{line_color}; font-size:13px;'>({tf_ret:+.2f}%)</span>"
                    
                    fig = go.Figure()
                    
                    if 'Score' in sliced_hist.columns and not sliced_hist['Score'].dropna().empty:
                        fig.add_trace(go.Scatter(x=sliced_hist.index, y=sliced_hist['Score'], mode='lines', name='Crypto Score', line=dict(color='fuchsia', width=2, dash='dot'), yaxis='y2'))

                    if 'BB_Upper' in sliced_hist.columns and not sliced_hist['BB_Upper'].dropna().empty:
                        fig.add_trace(go.Scatter(x=sliced_hist.index, y=sliced_hist['BB_Upper'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                        fig.add_trace(go.Scatter(x=sliced_hist.index, y=sliced_hist['BB_Lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(173, 216, 230, 0.2)', showlegend=False, hoverinfo='skip'))

                    fig.add_trace(go.Scatter(x=sliced_hist.index, y=sliced_hist['Close'], mode='lines', name='Price', line=dict(color=line_color, width=2.5)))
                    
                    if '200_WMA' in sliced_hist.columns and not sliced_hist['200_WMA'].dropna().empty:
                        fig.add_trace(go.Scatter(x=sliced_hist.index, y=sliced_hist['200_WMA'], mode='lines', name='200 WMA', line=dict(color='darkorange', width=2, dash='dash')))
                    if '50_SMA' in sliced_hist.columns and not sliced_hist['50_SMA'].dropna().empty:
                        fig.add_trace(go.Scatter(x=sliced_hist.index, y=sliced_hist['50_SMA'], mode='lines', name='50 SMA', line=dict(color='gold', width=1.5, dash='dot')))
                    if '200_SMA' in sliced_hist.columns and not sliced_hist['200_SMA'].dropna().empty:
                        fig.add_trace(go.Scatter(x=sliced_hist.index, y=sliced_hist['200_SMA'], mode='lines', name='200 SMA', line=dict(color='mediumpurple', width=2, dash='dash')))
                    
                    if coin['Avg'] > 0:
                        fig.add_hline(y=coin['Avg'], line_dash="dot", line_color="deepskyblue", line_width=2, opacity=0.8)
                    
                    fig.update_layout(title=dict(text=header_text, font=dict(size=14)), margin=dict(l=0, r=0, t=30, b=0), xaxis=dict(visible=False), yaxis=dict(visible=False), yaxis2=dict(range=[0, 100], overlaying='y', side='right', visible=False), showlegend=False, height=190, plot_bgcolor='rgba(0,0,0,0)', hovermode='x unified')
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.divider()

# --- APP LAYOUT ---
st.title("₿ Nightshift Crypto Command Center")
today = pd.Timestamp.today().tz_localize(None)

# --- FEAR & GREED INDEX WIDGET ---
fng_color = "red" if fng_val < 30 else ("orange" if fng_val < 50 else ("green" if fng_val < 75 else "darkgreen"))
st.markdown(f"**🌍 Global Market Sentiment:** :{fng_color}[**{fng_val}/100 ({fng_class})**]")
st.markdown("*Note: The algorithm is dynamically altering scores based on this global sentiment to enforce a contrarian strategy.*")

global_scores_df = load_score_history()
st.divider()

# --- NEW: PERMANENT BLUECHIP TRACKERS ---
st.markdown("### 👑 Major Market Movers")
bc_dict = {'BTC': {'shares': 0, 'avg_price': 0}, 'ETH': {'shares': 0, 'avg_price': 0}, 'SOL': {'shares': 0, 'avg_price': 0}}
with st.spinner("Fetching live bluechip data..."):
    bc_data, _, _ = get_crypto_data(bc_dict, fng_val)
    
if bc_data:
    # Display the score cards across 3 neat columns without the charts
    bc_cols = st.columns(3)
    for i, coin in enumerate(bc_data):
        if i < 3:
            with bc_cols[i]:
                render_score_card(coin, today, score_history=global_scores_df, hide_dollars=hide_dollars)
st.divider()

# --- RESEARCH STATION ---
st.markdown("### 🔍 Crypto Research Station")
search_query = st.text_input("Enter Coin Symbol (e.g. ADA, LINK, DOGE):", "").strip().upper()

if search_query:
    with st.spinner(f"Running historical breakdown on {search_query}..."):
        search_data, search_hist, _ = get_crypto_data({search_query: {'shares': 0, 'avg_price': 0}}, fng_val)
        if search_data and search_data[0]['Price'] > 0:
            col1, col2 = st.columns([8, 1])
            with col2:
                if search_query not in st.session_state.watchlist:
                    if st.button("⭐ Watch", key="add_watch"):
                        st.session_state.watchlist.append(search_query)
                        st.rerun()
                else: st.button("✅ Added", disabled=True)
            draw_crypto_row(search_data[0], search_hist, today, hide_dollars=hide_dollars, score_history=global_scores_df)
        else:
            st.warning(f"Could not find valid data for '{search_query}'. Ensure it's a valid symbol (e.g., BTC, ADA).")
st.divider()

if st.session_state.watchlist:
    st.markdown("### ⭐ My Watchlist")
    watch_dict = {ticker: {} for ticker in st.session_state.watchlist}
    with st.spinner("Updating Watchlist algorithms..."):
        watch_data, watch_hist, _ = get_crypto_data(watch_dict, fng_val)
    for coin in watch_data: draw_crypto_row(coin, watch_hist, today, is_watchlist=True, hide_dollars=hide_dollars, score_history=global_scores_df)

st.markdown("### 🏆 Top Market Scanner")
st.markdown("Live technical scan of the top layer-1s, layer-2s, and blue-chip tokens (Stablecoins excluded).")

global_universe = [
    'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOGE', 'DOT',
    'TRX', 'LINK', 'MATIC', 'TON', 'SHIB', 'LTC', 'BCH', 'UNI', 'NEAR', 'ATOM',
    'XLM', 'APT', 'OP', 'INJ', 'FIL', 'LDO', 'AR', 'RNDR', 'FTM', 'HBAR', 'PEPE'
]

if st.checkbox("Run Crypto Market Scan (Takes ~10 seconds)"):
    scan_dict = {ticker: {} for ticker in global_universe}
    with st.spinner("Scanning global assets..."):
        market_data, market_hist, _ = get_crypto_data(scan_dict, fng_val)
        
    if market_data:
        df_market = pd.DataFrame(market_data)
        df_top10 = df_market.sort_values(by=['Score', 'Drawdown'], ascending=[False, True]).head(10)
        
        export_cols = ['Ticker', 'Price', 'Score', 'Decision', 'Risk', 'Drawdown', 'RSI', 'Vol']
        df_export = df_top10[export_cols].copy()
        
        csv_data = df_export.to_csv(index=False).encode('utf-8')
        col_space, col_btn = st.columns([8, 2])
        with col_btn:
            st.download_button("💾 Export Top 10 to CSV", data=csv_data, file_name=f"Crypto_Scan_{today.strftime('%Y-%m-%d')}.csv", mime="text/csv")
        
        for idx, row in df_top10.iterrows():
            draw_crypto_row(row.to_dict(), market_hist, today, hide_dollars=hide_dollars, score_history=global_scores_df)
st.divider()

if st.session_state.crypto_portfolio:
    with st.spinner("Crunching your Crypto Portfolio..."):
        data, histories, total_val = get_crypto_data(st.session_state.crypto_portfolio, fng_val)

    if data:
        total_val_str = "$••••" if hide_dollars else f"${total_val:,.2f}"
        
        col_title, col_export = st.columns([8, 2])
        with col_title:
            st.subheader(f"Total Live Portfolio Value: {total_val_str}")
            
        with col_export:
            df_port = pd.DataFrame(data)
            port_cols = ['Ticker', 'Shares', 'Avg', 'Price', 'Score', 'Decision', 'Risk', 'Drawdown', 'RSI']
            csv_port = df_port[port_cols].to_csv(index=False).encode('utf-8')
            st.download_button("💾 Export Portfolio Grades", data=csv_port, file_name=f"My_Crypto_Grades_{today.strftime('%Y-%m-%d')}.csv", mime="text/csv")
            
        st.markdown("### 🩺 Portfolio Health")
        
        df_metrics = pd.DataFrame(data)
        df_metrics['Weight'] = 0.0 
        
        if total_val > 0:
            df_metrics['Weight'] = df_metrics['Val'] / total_val
            weighted_score = (df_metrics['Score'] * df_metrics['Weight']).sum()
            avg_risk = (df_metrics['Risk_Pts'] * df_metrics['Weight']).sum()
        else:
            weighted_score, avg_risk = 50, 0
            
        health_color = "normal" if weighted_score >= 50 else "inverse"
        overall_health = "Excellent" if weighted_score >= 65 else "Good" if weighted_score >= 50 else "Warning"
        
        h_cols = st.columns([1, 1, 2])
        
        with h_cols[0]:
            st.metric(label="Overall Health Status", value=overall_health, delta=f"Quant Score: {weighted_score:.1f}/100", delta_color=health_color)
            r_str = "High ⚠️" if avg_risk >= 2 else "Low 🛡️" if avg_risk <= 0 else "Moderate ⚖️"
            st.metric(label="Aggregated Portfolio Risk", value=r_str)
            
        with h_cols[1]:
            st.markdown("**Suggestions & Warnings:**")
            suggestions = []
            max_pos = df_metrics.loc[df_metrics['Weight'].idxmax()]
            if max_pos['Weight'] > 0.40:
                suggestions.append(f"⚠️ **Concentration:** {max_pos['Ticker']} makes up {max_pos['Weight']*100:.1f}% of your portfolio.")
            
            sell_candidates = df_metrics[df_metrics['Score'] < 40]
            if not sell_candidates.empty:
                tickers_str = ", ".join(sell_candidates['Ticker'].tolist())
                suggestions.append(f"💡 **Action Needed:** Algorithm flagged {tickers_str} as TRIM or SELL.")
                
            if not suggestions: st.success("✅ Well-balanced portfolio.")
            else:
                for sug in suggestions: st.info(sug)
                
        with h_cols[2]:
            fig_asset = go.Figure(data=[go.Pie(labels=df_metrics['Ticker'], values=df_metrics['Weight'], hole=.5, textinfo='label+percent')])
            fig_asset.update_layout(title_text="Asset Diversification", margin=dict(t=30, b=0, l=0, r=0), height=250)
            st.plotly_chart(fig_asset, use_container_width=True)

        st.divider()
        st.markdown("### 🎯 Algorithmic Asset Analysis")
        for coin in data: draw_crypto_row(coin, histories, today, hide_dollars=hide_dollars, score_history=global_scores_df)