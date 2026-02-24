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
try:
    from pytrends.request import TrendReq
except ImportError:
    pass

# --- SESSION STATE INITIALIZATION ---
if 'startup_sound_played' not in st.session_state:
    st.session_state.startup_sound_played = False
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'crypto_portfolio' not in st.session_state:
    st.session_state.crypto_portfolio = {}

st.set_page_config(page_title="Crypto Quant Command Center", layout="wide", page_icon="₿")

# --- YAHOO FINANCE TICKER MAP ---
YF_TICKER_MAP = {
    'SUI': 'SUI20947-USD', 'TAO': 'TAO22974-USD', 'PEPE': 'PEPE24478-USD', 'WIF': 'WIF28507-USD',
    'BONK': 'BONK23095-USD', 'TON': 'TON11419-USD', 'APT': 'APT21794-USD', 'OP': 'OP21594-USD',
    'UNI': 'UNI7083-USD', 'DOT': 'DOT-USD', 'MATIC': 'MATIC-USD', 'NEAR': 'NEAR-USD', 'INJ': 'INJ-USD',
    'FIL': 'FIL-USD', 'LDO': 'LDO-USD', 'AR': 'AR-USD', 'RNDR': 'RNDR-USD', 'FTM': 'FTM-USD', 
    'HBAR': 'HBAR-USD', 'BCH': 'BCH-USD', 'XLM': 'XLM-USD', 'TRX': 'TRX-USD', 'LTC': 'LTC-USD'
}

def get_yf_ticker(symbol):
    clean_symbol = symbol.replace('-USD', '').upper()
    return YF_TICKER_MAP.get(clean_symbol, f"{clean_symbol}-USD")

# --- PROJECT PROMISE & UTILITY TIERS ---
CRYPTO_TIERS = {
    'BTC': 1, 'ETH': 1, 
    'SOL': 2, 'ADA': 2, 'AVAX': 2, 'DOT': 2, 'LINK': 2, 'MATIC': 2, 'NEAR': 2, 'APT': 2, 'OP': 2, 'INJ': 2, 'XRP': 2, 'BNB': 2, 'TRX': 2, 'LTC': 2, 'SUI': 2, 'TAO': 2,
    'FIL': 2, 'LDO': 2, 'AR': 2, 'RNDR': 2, 'FTM': 2, 'HBAR': 2, 'TON': 2, 'BCH': 2, 'UNI': 2, 'XLM': 2,
    'DOGE': 3, 'SHIB': 3, 'PEPE': 3, 'FLOKI': 3, 'BONK': 3, 'WIF': 3 
}

# --- QUALITATIVE FUNDAMENTAL DATA & MACRO TARGETS ---
CRYPTO_META = {
    'BTC': {'desc': "The decentralized digital gold.", 'utility': 95, 'decentralization': 100, 'staked': 0, 'target': 150000, 'trend_term': "Bitcoin"},
    'ETH': {'desc': "The leading smart contract platform.", 'utility': 98, 'decentralization': 85, 'staked': 27, 'target': 8000, 'trend_term': "Ethereum"},
    'SOL': {'desc': "High-speed monolithic L1 optimized for adoption.", 'utility': 90, 'decentralization': 50, 'staked': 68, 'target': 500, 'trend_term': "Solana"}, 
    'SUI': {'desc': "Next-gen L1 built with Move programming language.", 'utility': 85, 'decentralization': 45, 'staked': 80, 'target': 5, 'trend_term': "Sui Crypto"}, 
    'TAO': {'desc': "Decentralized open-source AI machine learning.", 'utility': 92, 'decentralization': 75, 'staked': 72, 'target': 1200, 'trend_term': "Bittensor"},
    'LINK': {'desc': "Industry standard decentralized oracle network.", 'utility': 95, 'decentralization': 70, 'staked': 12, 'target': 50, 'trend_term': "Chainlink"},
    'AVAX': {'desc': "Highly scalable subnet-focused smart contract platform.", 'utility': 85, 'decentralization': 65, 'staked': 55, 'target': 100, 'trend_term': "Avalanche Crypto"},
    'ADA': {'desc': "Peer-reviewed, academic Proof of Stake blockchain.", 'utility': 75, 'decentralization': 80, 'staked': 63, 'target': 2.50, 'trend_term': "Cardano"},
    'XRP': {'desc': "Legacy cross-border institutional payment protocol.", 'utility': 70, 'decentralization': 40, 'staked': 0, 'target': 2, 'trend_term': "XRP"},
    'BNB': {'desc': "Binance ecosystem utility and smart chain token.", 'utility': 80, 'decentralization': 30, 'staked': 15, 'target': 1000, 'trend_term': "Binance Coin"},
    'DOGE': {'desc': "The original PoW meme cryptocurrency.", 'utility': 30, 'decentralization': 75, 'staked': 0, 'target': 1.00, 'trend_term': "Dogecoin"},
    'SHIB': {'desc': "ERC-20 meme token with building DeFi ecosystem.", 'utility': 35, 'decentralization': 60, 'staked': 2, 'target': 0.00008, 'trend_term': "Shiba Inu Coin"},
    'DOT': {'desc': "Interoperability network connecting bespoke parachains.", 'utility': 80, 'decentralization': 75, 'staked': 52, 'target': 25, 'trend_term': "Polkadot Crypto"},
    'MATIC': {'desc': "Ethereum's premier L2 scaling solution (Polygon).", 'utility': 85, 'decentralization': 60, 'staked': 35, 'target': 2.00, 'trend_term': "Polygon Crypto"},
    'NEAR': {'desc': "Highly scalable, sharded Proof-of-Stake L1.", 'utility': 80, 'decentralization': 65, 'staked': 45, 'target': 15, 'trend_term': "Near Protocol"},
    'APT': {'desc': "High-performance L1 spun out of Facebook's Diem project.", 'utility': 80, 'decentralization': 40, 'staked': 80, 'target': 30, 'trend_term': "Aptos Crypto"},
    'OP': {'desc': "Optimistic rollup L2 scaling network for Ethereum.", 'utility': 85, 'decentralization': 50, 'staked': 20, 'target': 8, 'trend_term': "Optimism Crypto"},
    'INJ': {'desc': "App-specific L1 built for decentralized finance.", 'utility': 80, 'decentralization': 65, 'staked': 50, 'target': 60, 'trend_term': "Injective Protocol"},
    'FIL': {'desc': "Decentralized storage network designed to store humanity's info.", 'utility': 85, 'decentralization': 70, 'staked': 30, 'target': 15, 'trend_term': "Filecoin"},
    'LDO': {'desc': "The dominant liquid staking protocol for Ethereum.", 'utility': 90, 'decentralization': 40, 'staked': 10, 'target': 5, 'trend_term': "Lido Crypto"},
    'AR': {'desc': "Decentralized permaweb for immutable data storage.", 'utility': 80, 'decentralization': 75, 'staked': 20, 'target': 60, 'trend_term': "Arweave"},
    'RNDR': {'desc': "Distributed GPU rendering network for creators.", 'utility': 85, 'decentralization': 60, 'staked': 0, 'target': 15, 'trend_term': "Render Crypto"},
    'FTM': {'desc': "High-performance DAG smart contract platform.", 'utility': 75, 'decentralization': 60, 'staked': 45, 'target': 2, 'trend_term': "Fantom Crypto"},
    'HBAR': {'desc': "Enterprise-grade Hashgraph distributed ledger.", 'utility': 80, 'decentralization': 40, 'staked': 40, 'target': 0.20, 'trend_term': "Hedera Hashgraph"},
    'PEPE': {'desc': "A purely speculative frog-themed meme coin.", 'utility': 10, 'decentralization': 60, 'staked': 0, 'target': 0, 'trend_term': "Pepe Coin"},
    'TON': {'desc': "The Open Network L1 closely tied to Telegram.", 'utility': 80, 'decentralization': 40, 'staked': 35, 'target': 15, 'trend_term': "Toncoin"},
    'BCH': {'desc': "Bitcoin fork engineered specifically for daily payments.", 'utility': 60, 'decentralization': 80, 'staked': 0, 'target': 1000, 'trend_term': "Bitcoin Cash"},
    'UNI': {'desc': "The leading decentralized exchange and AMM on Ethereum.", 'utility': 90, 'decentralization': 60, 'staked': 0, 'target': 20, 'trend_term': "Uniswap"},
    'XLM': {'desc': "Payments network for fast, low-cost global transfers.", 'utility': 70, 'decentralization': 50, 'staked': 0, 'target': 0.50, 'trend_term': "Stellar Lumens"},
    'TRX': {'desc': "Entertainment-focused L1 dominating the Tether market.", 'utility': 75, 'decentralization': 40, 'staked': 50, 'target': 0.20, 'trend_term': "Tron Crypto"},
    'LTC': {'desc': "One of the oldest PoW coins, known as digital silver.", 'utility': 60, 'decentralization': 90, 'staked': 0, 'target': 150, 'trend_term': "Litecoin"}
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

@st.cache_data(ttl=86400) 
def get_google_trend(keyword):
    """Fetches real-time Google Search interest. Returns None if Google blocks the API call."""
    try:
        pytrends = TrendReq(hl='en-US', tz=360, timeout=(5,10), retries=2, backoff_factor=0.5)
        pytrends.build_payload([keyword], cat=0, timeframe='today 3-m', geo='', gprop='')
        df = pytrends.interest_over_time()
        if not df.empty:
            return int(df[keyword].iloc[-1])
    except Exception: pass
    return None 

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
        ticker = new_coin.replace('-USD', '')
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
        with colA: st.write(f"**{tkr}**: {data['shares']:g}")
        with colB:
            if st.button("❌", key=f"del_{tkr}"):
                del st.session_state.crypto_portfolio[tkr]
                st.rerun()

# --- QUANT DATA FETCHING ENGINE ---
@st.cache_data(ttl=900) 
def get_crypto_data(port_dict, global_fng_val):
    if not port_dict: return [], {}, 0 
    
    portfolio_data = []
    all_histories = {} 
    total_value = 0

    for display_ticker, data in port_dict.items():
        yf_ticker = get_yf_ticker(display_ticker)
        
        shares, avg_price = data.get('shares', 0), data.get('avg_price', 0)
        coin = yf.Ticker(yf_ticker)
        
        hist = coin.history(period='max')
        if hist.empty: continue
            
        current_price = hist['Close'].iloc[-1]
        hist.index = hist.index.tz_localize(None)
        all_histories[display_ticker] = hist
            
        volatility, drawdown, ath = 0.0, 0.0, 0.0
        rsi_14, macd_val, sig_val, bb_upper, bb_lower = 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'
        sma_50, sma_200 = 'N/A', 'N/A'
        obv_trend = "Neutral"
        
        if not hist.empty:
            # --- 2-YEAR ATH CALCULATION ---
            two_years_ago = hist.index[-1] - timedelta(days=730)
            hist_2y = hist[hist.index >= two_years_ago]
            ath = hist_2y['Close'].max() if not hist_2y.empty else hist['Close'].max()
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

        # --- THE ALGORITHM ---
        score = 30.0 
        risk_points = 0
        
        tier = CRYPTO_TIERS.get(display_ticker, 4)
        meta = CRYPTO_META.get(display_ticker, {'desc': 'No custom fundamentals available. Treat as high-risk altcoin.', 'utility': 30, 'decentralization': 40, 'staked': 0, 'target': 0, 'trend_term': f"{display_ticker} Crypto"})
        
        google_fomo = get_google_trend(meta['trend_term'])
        
        tier_str = "Bluechip" if tier == 1 else "Utility/L1" if tier == 2 else "Meme" if tier == 3 else "Speculative/Alt"
        breakdown = [f"**Base Score:** 30 pts", f"🧬 **Project Classification:** Tier {tier} ({tier_str})"]
        
        if tier == 4: risk_points += 1 
        if tier == 3: risk_points += 1 

        # 0. Fundamental & Macro Targets
        if meta['utility'] >= 85: score += 5; breakdown.append(f"🧠 **High Utility:** +5 pts")
        elif meta['utility'] < 50: score -= 10; breakdown.append(f"📉 **Low Utility/Meme:** -10 pts")
            
        if meta['decentralization'] >= 80: score += 5; breakdown.append(f"🌐 **Highly Decentralized:** +5 pts")
        elif meta['decentralization'] <= 50: score -= 10; breakdown.append(f"🐋 **Centralized/Whale Heavy:** -10 pts")
            
        if meta['staked'] >= 50: score += 10; breakdown.append(f"🔒 **Massive Staking Lockup ({meta['staked']}%):** +10 pts (Supply Shock)")
        elif meta['staked'] >= 20: score += 5; breakdown.append(f"🔒 **Healthy Staking Lockup ({meta['staked']}%):** +5 pts")
            
        if meta['target'] > 0:
            upside = ((meta['target'] - current_price) / current_price) * 100
            if upside <= 0:
                upside_pts = -15
                breakdown.append(f"🚨 **Above Macro Target ({upside:+.1f}%):** -15 pts (Overvalued)")
            elif upside <= 30:
                upside_pts = -15 + (upside / 30.0) * 15
                breakdown.append(f"⚠️ **Near Macro Target ({upside:+.1f}%):** {upside_pts:+.1f} pts (Limited Upside)")
            elif upside <= 200:
                upside_pts = ((upside - 30) / 170.0) * 15
                breakdown.append(f"✅ **Healthy Upside ({upside:+.1f}%):** +{upside_pts:.1f} pts")
            else:
                upside_pts = 15
                breakdown.append(f"🚀 **Massive Upside Potential ({upside:+.1f}%):** +15 pts")
            score += upside_pts

        # 1. Retail FOMO (Google Trends Penalty)
        if google_fomo is not None:
            if google_fomo >= 80:
                score -= 20; risk_points += 1
                breakdown.append(f"🚨 **Extreme Retail FOMO (Search:{google_fomo}):** -20 pts (Blowout Top Warning)")
            elif google_fomo <= 20:
                score += 5
                breakdown.append(f"🤫 **Silent Accumulation (Search:{google_fomo}):** +5 pts (No Retail Interest)")
            else:
                breakdown.append(f"➖ **Retail Interest Neutral (Search:{google_fomo}):** 0 pts")
        else:
            breakdown.append(f"⚠️ **Retail FOMO:** [Google API Rate Limited/Blocked]")

        # 2. Historical Drawdown (2-Year Rolling)
        dd_abs = abs(drawdown)
        if tier <= 2:
            if dd_abs <= 15: dd_pts = -10; breakdown.append(f"❌ **Near 2Y ATH ({drawdown:.1f}%):** -10 pts (FOMO Zone)")
            elif dd_abs <= 40: dd_pts = 0; breakdown.append(f"➖ **Fair Value ({drawdown:.1f}%):** 0 pts")
            elif dd_abs <= 75: dd_pts = 10; breakdown.append(f"✅ **Deep Value ({drawdown:.1f}%):** +10 pts")
            else: dd_pts = 20; breakdown.append(f"✅ **Maximum Capitulation ({drawdown:.1f}%):** +20 pts")
            score += dd_pts
        elif tier == 3: 
            if dd_abs <= 20: dd_pts = -15; breakdown.append(f"❌ **Meme Top Risk ({drawdown:.1f}%):** -15 pts")
            elif dd_abs <= 60: dd_pts = 0; breakdown.append(f"➖ **Meme Drawdown ({drawdown:.1f}%):** 0 pts")
            else: dd_pts = 5; breakdown.append(f"⚠️ **Meme Capitulation ({drawdown:.1f}%):** +5 pts")
            score += dd_pts
        else: 
            if dd_abs > 80: score -= 20; risk_points += 2; breakdown.append(f"🚨 **Dead Altcoin Risk ({drawdown:.1f}%):** -20 pts [+2 Risk]")

        # 3. Distance from 200 SMA
        if sma_200 != 0 and current_price > 0:
            sma_dist = ((current_price - sma_200) / sma_200) * 100
            if sma_dist > 40: sma_pts = -20; breakdown.append(f"❌ **SMA Ext ({sma_dist:+.1f}%):** -20 pts (Severely Overextended)")
            elif sma_dist > 15: sma_pts = -10; breakdown.append(f"⚠️ **SMA Ext ({sma_dist:+.1f}%):** -10 pts (Cooling Off Needed)")
            elif sma_dist >= 0: sma_pts = 5; breakdown.append(f"✅ **SMA Ext ({sma_dist:+.1f}%):** +5 pts (Steady Uptrend)")
            elif sma_dist >= -15: sma_pts = 5; breakdown.append(f"✅ **SMA Ext ({sma_dist:+.1f}%):** +5 pts (Accumulating)")
            else: sma_pts = -15; breakdown.append(f"❌ **SMA Ext ({sma_dist:+.1f}%):** -15 pts (Macro Breakdown)")
            score += sma_pts

        if sma_50 > 0 and sma_200 > 0:
            if sma_50 > sma_200: score += 5; breakdown.append("✅ **Golden Cross:** +5 pts")
            else: score -= 10; breakdown.append("❌ **Death Cross:** -10 pts")

        # 4. RSI Momentum 
        if isinstance(rsi_14, (float, int)):
            if rsi_14 < 30: rsi_pts = 10
            elif rsi_14 < 40: rsi_pts = 5
            elif rsi_14 < 55: rsi_pts = 0
            elif rsi_14 < 65: rsi_pts = -5
            elif rsi_14 < 75: rsi_pts = -15
            else: rsi_pts = -25
            score += rsi_pts; breakdown.append(f"{'✅' if rsi_pts > 0 else '❌' if rsi_pts < 0 else '➖'} **RSI ({rsi_14}):** {rsi_pts:+} pts")

        # 5. OBV & MACD
        if obv_trend == "Accumulating": score += 5; breakdown.append(f"🐋 **OBV:** +5 pts (Whale Accumulation)")
        elif obv_trend == "Distributing": score -= 5; breakdown.append(f"🚨 **OBV:** -5 pts (Whale Distribution)")

        if isinstance(macd_val, (float, int)) and isinstance(sig_val, (float, int)):
            if macd_val > sig_val: score += 5
            else: score -= 5

        # 6. Global Sentiment Modifier
        if global_fng_val > 75: fng_adj = -15; breakdown.append(f"❌ **Market Euphoria ({global_fng_val}):** -15 pts (Take Profits)")
        elif global_fng_val > 60: fng_adj = -5; breakdown.append(f"⚠️ **Market Greed ({global_fng_val}):** -5 pts")
        elif global_fng_val < 30: fng_adj = 10; breakdown.append(f"✅ **Market Capitulation ({global_fng_val}):** +10 pts (Contrarian Buy)")
        else: fng_adj = 0; breakdown.append(f"➖ **Market Neutral ({global_fng_val}):** 0 pts")
        score += fng_adj

        if volatility > 100: risk_points += 2; score -= 5; breakdown.append("⚠️ **Extreme Volatility (>100%):** -5 pts [+2 Risk]")
        elif volatility < 40: risk_points -= 1; score += 5; breakdown.append("🛡️ **Low Volatility (<40%):** +5 pts [-1 Risk]")

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
            'Meta_Desc': meta['desc'], 'Meta_Util': meta['utility'], 'Meta_Decen': meta['decentralization'], 
            'Meta_Staked': meta['staked'], 'Meta_Target': meta['target'], 'Meta_Google': google_fomo,
            'Score': int(score), 'Decision': decision, 'D_Color': d_color, 'Risk': risk_lvl, 'R_Color': r_color, 'Risk_Pts': risk_points,
            'Upper_BB': bb_upper, 'Lower_BB': bb_lower,
            'Breakdown': breakdown
        })

    log_scores_to_csv(portfolio_data)
    return portfolio_data, all_histories, total_value

# --- UI HELPER: SCORE CARD COMPONENT ---
def render_score_card(coin, today_date, score_history=None, is_watchlist=False, hide_dollars=False):
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
        
        arkham_url = f"https://platform.arkhamintelligence.com/explorer/token/{ticker.lower()}"
        dune_url = f"https://dune.com/search?q={ticker}&time_range=all"
        yf_ticker = get_yf_ticker(ticker)
        
        st.markdown(
            f"""
            <div style="display: flex; gap: 10px; font-size: 13px; margin-bottom: 10px;">
                <a href='https://finance.yahoo.com/quote/{yf_ticker}' target='_blank' style='text-decoration: none;'>📈 Chart</a>
                <a href='{arkham_url}' target='_blank' style='text-decoration: none;'>🐋 Arkham</a>
                <a href='{dune_url}' target='_blank' style='text-decoration: none;'>📊 Dune</a>
            </div>
            """, unsafe_allow_html=True
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
        st.write(f"**2Y ATH:** {price_format.format(ath)}")
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

# --- UI HELPER: FULL ROW WITH NEW LAYOUT ---
def draw_crypto_row(coin, histories, today_date, is_watchlist=False, hide_dollars=False, score_history=None):
    cols = st.columns([1.5, 1.5, 3]) 
    
    with cols[0]:
        render_score_card(coin, today_date, score_history, is_watchlist, hide_dollars)
        
    with cols[1]:
        st.markdown("### 🧬 Project Fundamentals")
        st.markdown(f"<p style='font-size: 14px; color: gray; margin-bottom:15px;'>{coin['Meta_Desc']}</p>", unsafe_allow_html=True)
        
        util_val = coin['Meta_Util']
        util_color = "green" if util_val >= 80 else ("orange" if util_val >= 50 else "red")
        st.markdown(f"**Utility & Tech:** :{util_color}[**{util_val}/100**]")
        st.progress(util_val / 100)
        
        decen_val = coin['Meta_Decen']
        decen_color = "green" if decen_val >= 75 else ("orange" if decen_val >= 50 else "red")
        st.markdown(f"**Decentralization:** :{decen_color}[**{decen_val}/100**]")
        st.progress(decen_val / 100)
        
        st.markdown("---")
        subA, subB = st.columns(2)
        with subA:
            stk = coin['Meta_Staked']
            st.write(f"**Staked (Locked):** {stk}%")
            
            tgt = coin['Meta_Target']
            tgt_str = f"${tgt:,.2f}" if tgt > 0 else "N/A"
            st.write(f"**Macro Target:** {tgt_str}")
        with subB:
            g_fomo = coin['Meta_Google']
            if g_fomo is not None:
                g_color = "red" if g_fomo > 80 else ("green" if g_fomo < 30 else "gray")
                st.markdown(f"**Retail FOMO:** :{g_color}[{g_fomo}/100]")
            else:
                st.markdown("**Retail FOMO:** :orange[API Rate Limited]")
            
    ticker = coin['Ticker']
    master_hist = histories.get(ticker)
    
    if master_hist is not None and not master_hist.empty:
        if score_history is not None and not score_history.empty:
            t_scores = score_history[score_history['Ticker'] == ticker].copy()
            if not t_scores.empty:
                t_scores.set_index('Date', inplace=True)
                t_scores = t_scores[~t_scores.index.duplicated(keep='last')]
                # Merge the Quant Score properly so it plots
                master_hist = master_hist.join(t_scores['Score'], how='left')
                master_hist['Score'] = master_hist['Score'].ffill()
                
        if len(master_hist) > 20:
            master_hist['BB_Upper'], master_hist['BB_Lower'] = calculate_bbands(master_hist['Close'])

        with cols[2]:
            fig = go.Figure()
            
            # --- Ensure the Quant Score Line actually renders ---
            if 'Score' in master_hist.columns and not master_hist['Score'].dropna().empty:
                fig.add_trace(go.Scatter(x=master_hist.index, y=master_hist['Score'], mode='lines', name='Quant Score', line=dict(color='rgba(255, 0, 255, 0.4)', width=2, dash='dot'), yaxis='y2'))

            if 'BB_Upper' in master_hist.columns and not master_hist['BB_Upper'].dropna().empty:
                fig.add_trace(go.Scatter(x=master_hist.index, y=master_hist['BB_Upper'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=master_hist.index, y=master_hist['BB_Lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(173, 216, 230, 0.1)', showlegend=False, hoverinfo='skip'))

            fig.add_trace(go.Scatter(x=master_hist.index, y=master_hist['Close'], mode='lines', name='Price', line=dict(color='#2ca02c', width=2.5)))
            
            if '200_WMA' in master_hist.columns and not master_hist['200_WMA'].dropna().empty:
                fig.add_trace(go.Scatter(x=master_hist.index, y=master_hist['200_WMA'], mode='lines', name='200 WMA', line=dict(color='darkorange', width=2, dash='dash')))
            if '50_SMA' in master_hist.columns and not master_hist['50_SMA'].dropna().empty:
                fig.add_trace(go.Scatter(x=master_hist.index, y=master_hist['50_SMA'], mode='lines', name='50 SMA', line=dict(color='gold', width=1.5, dash='dot')))
            if '200_SMA' in master_hist.columns and not master_hist['200_SMA'].dropna().empty:
                fig.add_trace(go.Scatter(x=master_hist.index, y=master_hist['200_SMA'], mode='lines', name='200 SMA', line=dict(color='mediumpurple', width=2, dash='dash')))
            
            if coin['Avg'] > 0:
                fig.add_hline(y=coin['Avg'], line_dash="dot", line_color="deepskyblue", line_width=2, opacity=0.8)
            
            fig.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=3, label="3m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all", label="Max")
                        ]),
                        bgcolor='rgba(150, 150, 150, 0.1)',
                        activecolor='rgba(44, 160, 44, 0.5)'
                    ),
                    type="date"
                ),
                yaxis=dict(visible=True, side='left'), 
                yaxis2=dict(range=[0, 100], overlaying='y', side='right', visible=False), 
                showlegend=False, 
                height=350, 
                plot_bgcolor='rgba(0,0,0,0)', 
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.divider()

# --- APP LAYOUT ---
st.title("₿ Nightshift Crypto Command Center")
today = pd.Timestamp.today().tz_localize(None)

fng_color = "red" if fng_val < 30 else ("orange" if fng_val < 50 else ("green" if fng_val < 75 else "darkgreen"))
st.markdown(f"**🌍 Global Market Sentiment:** :{fng_color}[**{fng_val}/100 ({fng_class})**]")
st.markdown("*Note: The algorithm dynamically alters scores based on macro sentiment and Google Search trends.*")

global_scores_df = load_score_history()
st.divider()

# --- PERMANENT BLUECHIP TRACKERS ---
st.markdown("### 👑 Major Market Movers")
bc_dict = {'BTC': {'shares': 0, 'avg_price': 0}, 'ETH': {'shares': 0, 'avg_price': 0}, 'SOL': {'shares': 0, 'avg_price': 0}}
with st.spinner("Fetching live bluechip data..."):
    bc_data, _, _ = get_crypto_data(bc_dict, fng_val)
    
if bc_data:
    bc_cols = st.columns(3)
    for i, coin in enumerate(bc_data):
        if i < 3:
            with bc_cols[i]:
                render_score_card(coin, today, score_history=global_scores_df, hide_dollars=hide_dollars)
st.divider()

# --- RESEARCH STATION ---
st.markdown("### 🔍 Crypto Research Station")
search_query = st.text_input("Enter Coin Symbol (e.g. ADA, LINK, SUI):", "").strip().upper()

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
st.markdown("Live technical scan of the top layer-1s, layer-2s, and blue-chip tokens.")

global_universe = [
    'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOGE', 'DOT',
    'TRX', 'LINK', 'MATIC', 'TON', 'SHIB', 'LTC', 'BCH', 'UNI', 'NEAR', 'ATOM',
    'XLM', 'APT', 'OP', 'INJ', 'FIL', 'LDO', 'AR', 'RNDR', 'FTM', 'HBAR', 'PEPE', 'SUI', 'TAO'
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