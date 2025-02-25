import streamlit as st
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_ollama.llms import OllamaLLM
import chromadb
from groq import Client
import os
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px
import time
import requests

# ====== Configuration ======
st.set_page_config(
    page_title="FinVision Pro", 
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== API Configuration ======
API_KEY = st.secrets["YF_API_KEY"]
BASE_URL = "https://yfapi.net"
HEADERS = {'x-api-key': API_KEY}

# ====== Helper Functions ======
def get_api_data(endpoint, params=None):
    """Handle API requests with error handling and rate limiting"""
    try:
        time.sleep(0.3)  # Rate limiting for basic plan
        response = requests.get(
            f"{BASE_URL}{endpoint}",
            headers=HEADERS,
            params=params
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        st.error(f"Network Error: {str(e)}")
    return None

def get_safe(data, key, default="N/A"):
    """Safely get values from nested API responses"""
    keys = key.split('.')
    for k in keys:
        data = data.get(k, {}) if isinstance(data, dict) else default
        if data == default: break
    return data if data not in [None, "", []] else default

# ====== Updated Stock Data Functions ======
def get_stock_info(symbol):
    """Get comprehensive stock data from API"""
    data = {}
    
    # Basic quote information
    quote = get_api_data("/v6/finance/quote", params={"symbols": symbol})
    if quote and 'quoteResponse' in quote:
        data['quote'] = quote['quoteResponse']['result'][0]
    
    # Historical data
    history = get_api_data(f"/v8/finance/chart/{symbol}", params={
        "range": "1y",
        "interval": "1d"
    })
    if history and 'chart' in history:
        chart_data = history['chart']['result'][0]
        timestamps = pd.to_datetime(chart_data['timestamp'], unit='s')
        closes = chart_data['indicators']['quote'][0]['close']
        data['history'] = pd.DataFrame({'Date': timestamps, 'Close': closes})
    
    # Options data
    options = get_api_data(f"/v7/finance/options/{symbol}")
    if options and 'optionChain' in options:
        data['options'] = options['optionChain']['result'][0]
    
    return data

def get_options_chain(symbol, expiration):
    """Fetch options chain from API"""
    return get_api_data(f"/v7/finance/options/{symbol}", params={"date": expiration})

def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2) if option_type == 'call' else K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    # Convert to native Python float
    return {'price': float(price)}

# ====== Main App Updates ======
def main():
    st.title("💹 FinVision Pro")
    st.caption("AI-Powered Financial Intelligence Platform")

    # ====== Session State ======
    if 'favorites' not in st.session_state:
        st.session_state.favorites = []
    if 'symbol' not in st.session_state:
        st.session_state.symbol = "AAPL"

    # ====== Sidebar ======
    with st.sidebar:
        st.header("⭐ Favorites")
        new_fav = st.text_input("Add symbol:", key="new_fav")
        if new_fav:
            cleaned_fav = new_fav.upper().strip()
            if cleaned_fav not in [f[0] for f in st.session_state.favorites]:
                st.session_state.favorites.append((cleaned_fav, cleaned_fav, "🏢"))
        
        if st.session_state.favorites:
            st.write("Favorite Stocks:")
            for sym, name, icon in st.session_state.favorites:
                if st.button(f"{icon} {sym}", key=f"fav_{sym}"):
                    st.session_state.symbol = sym
                    st.rerun()

    # ====== Stock Selector ======
    with st.container():
        col1, col2 = st.columns([3,1])
        with col1:
            search_term = st.text_input("🔍 Search any company:", 
                                     value=st.session_state.symbol,
                                     placeholder="Search by name or symbol...")
        with col2:
            if st.button("🎲 Random", help="Pick random popular stock"):
                popular = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'NVDA', 'META']
                st.session_state.symbol = np.random.choice(popular)
                st.rerun()

    # ====== Data Fetching ======
    symbol = st.session_state.symbol
    data = get_stock_info(symbol)
    
    if not data or 'quote' not in data:
        st.error("Failed to fetch data for this symbol")
        st.stop()

    # ====== Key Metrics ======
    quote = data['quote']
    current_price = get_safe(quote, 'regularMarketPrice', 0)
    prev_close = get_safe(quote, 'regularMarketPreviousClose', current_price)
    delta = current_price - prev_close
    pct_change = (delta / prev_close) * 100 if prev_close else 0

    with st.container():
        cols = st.columns(4)
        metrics = [
            ("💵 Current Price", f"${current_price:.2f}", "#2ecc71" if delta >=0 else "#e74c3c"),
            ("📈 52W High", f"${get_safe(quote, 'fiftyTwoWeekHigh', 'N/A')}", "#2ecc71"),
            ("📉 52W Low", f"${get_safe(quote, 'fiftyTwoWeekLow', 'N/A')}", "#e74c3c"),
            ("📊 Volume", f"{get_safe(quote, 'regularMarketVolume', 0):,}", "#3498db")
        ]
        
        for col, (title, value, color) in zip(cols, metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="color: {color}; font-size: 1.2rem; margin-bottom: 0.5rem">{title}</div>
                    <div style="font-size: 1.5rem; font-weight: 700">{value}</div>
                </div>
                """, unsafe_allow_html=True)

    # ====== Interactive Chart ======
    with st.container():
        st.header("📈 Price Analysis", divider="blue")
        if 'history' in data:
            hist_data = data['history']
            fig = px.area(hist_data, x='Date', y='Close',
                         title=f"{symbol} Price Trend",
                         color_discrete_sequence=["#3498db"])
            fig.update_layout(template="plotly_white", height=400,
                             hovermode="x unified", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # ====== AI Analysis ======
    def generate_action_plan():
        client = Client(api_key=st.secrets["GROQ_API_KEY"])
        
        prompt = f"""
        As a top financial analyst, provide {symbol} analysis with clear action steps:
        
        1. Technical Analysis:
        - Key support/resistance levels
        - Momentum indicators (RSI, MACD)
        - Volume analysis
        - Chart patterns
        
        2. Fundamental Analysis:
        - Valuation metrics (P/E, P/S)
        - Growth projections
        - Competitive landscape
        - Management quality
        
        3. Risk Assessment:
        - Market risks
        - Sector risks
        - Company-specific risks
        
        4. Action Plan:
        - Recommendation (Strong Buy/Buy/Hold/Sell/Strong Sell)
        - Entry price range
        - Price targets (3 levels with timelines)
        - Stop-loss levels
        - Position sizing (% of portfolio)
        
        5. Scenarios:
        - Bull case (best scenario)
        - Base case (likely scenario)
        - Bear case (worst scenario)
        
        Format with clear headers (##), bullet points, and emojis. 
        Use simple but professional language.
        """
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a hedge fund manager..."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content

    with st.container():
        st.header("🤖 AI Insights", divider="blue")
        if st.button("🚀 Generate Action Plan", type="primary"):
            with st.spinner("🔭 Building comprehensive strategy..."):
                try:
                    analysis = generate_action_plan()                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 1.1rem; line-height: 1.6">
                        {analysis}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.download_button(
                        label="📥 Download Full Report",
                        data=analysis,
                        file_name=f"{symbol}_report.md",
                        mime="text/markdown"
                    )
                except Exception as e:
                    st.error("Analysis unavailable. Please try again later.")

    # ====== Updated Options Analysis Section ======
    with st.container():
        with st.container():
            st.header("📊 Options Lab", divider="blue")
            if 'options' in data:
                try:
                    options_data = data['options']
                    
                    # Extract expiration dates from options chain
                    expiration_timestamps = options_data.get('expirationDates', [])
                    exp_dates = [
                        datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                        for ts in expiration_timestamps
                    ] if expiration_timestamps else []
                    
                    if exp_dates:
                        with st.expander("🧮 Options Calculator", expanded=True):
                            # Get full options chain for first expiration
                            chain = get_api_data(f"/v7/finance/options/{symbol}", params={"date": expiration_timestamps[0]})
                            calls = chain['optionChain']['result'][0]['options'][0]['calls']
                            puts = chain['optionChain']['result'][0]['options'][0]['puts']
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                option_type = st.selectbox("Contract Type", ["call", "put"])
                                risk_free = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, 2.5) / 100
                            with col2:
                                expiry = st.selectbox("Expiration Date", exp_dates)
                                days_to_expiry = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.now()).days
                            with col3:
                                strikes = sorted({c['strike'] for c in calls + puts})
                                strike = st.selectbox("Strike Price", strikes)
                            
                            if st.button("💵 Calculate Premium", type="primary"):
                                # Find matching contract for volatility
                                contracts = calls if option_type == "call" else puts
                                selected = next((c for c in contracts if c['strike'] == strike), None)
                                
                                if selected:
                                    iv = selected.get('impliedVolatility', 0.3)
                                    result = black_scholes(
                                        current_price,
                                        strike,
                                        days_to_expiry/365,
                                        risk_free,
                                        iv,
                                        option_type
                                    )
                                    
                                    cols = st.columns(2)
                                    cols[0].metric("Theoretical Price", f"${result['price']:.2f}")
                                    cols[1].metric("Implied Volatility", f"{iv*100:.1f}%")
                                else:
                                    st.error("Option contract not found")
                    else:
                        st.warning("No options data available for this stock")
                        
                except Exception as e:
                    st.error(f"Options processing error: {str(e)}")
                    
    # ====== Document Analysis ======
    with st.container():
        st.header("📄 Research Center", divider="blue")
        uploaded_files = st.file_uploader("Upload Financial Documents", type="pdf", accept_multiple_files=True)
        doc_query = st.text_input("Ask about documents:", placeholder="Search for risk factors or strategies...")
        
        if uploaded_files and doc_query:
            with st.status("🔍 Analyzing Documents...", expanded=True):
                try:
                    docs = []
                    for file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False) as tmp:
                            tmp.write(file.read())
                            loader = UnstructuredPDFLoader(tmp.name)
                            docs.extend(loader.load())
                        os.unlink(tmp.name)
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
                    chunks = text_splitter.split_documents(docs)
                    llm = OllamaLLM(model="mistral")
                    
                    cols = st.columns(2)
                    for i, chunk in enumerate(chunks[:4]):
                        with cols[i % 2]:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div style="color: #3498db; margin-bottom: 0.5rem">📌 Insight {i+1}</div>
                                {chunk.page_content[:200]}...
                            </div>
                            """, unsafe_allow_html=True)
                                
                except Exception as e:
                    st.error(f"Document processing error: {str(e)}")

if __name__ == "__main__":
    main()