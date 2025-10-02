import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="Crypto Analysis Pro",
    page_icon="🚀",
    layout="wide"
)

# CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .positive { color: #00aa00; font-weight: bold; }
    .negative { color: #ff0000; font-weight: bold; }
    .neutral { color: #ffaa00; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class CryptoAnalyzer:
    def __init__(self):
        self.cache = {}
    
    @st.cache_data(ttl=3600)
    def get_all_coins(_self):
        """Get semua coin dari CoinGecko"""
        try:
            url = "https://api.coingecko.com/api/v3/coins/list"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            return []
        except:
            return []
    
    def search_coins(self, query, limit=50):
        """Search coins berdasarkan query"""
        all_coins = self.get_all_coins()
        if not query:
            return []
        
        query = query.lower()
        results = []
        
        for coin in all_coins:
            coin_id = coin.get('id', '').lower()
            coin_symbol = coin.get('symbol', '').lower()
            coin_name = coin.get('name', '').lower()
            
            if (query in coin_id or query in coin_symbol or query in coin_name):
                results.append(coin)
            
            if len(results) >= limit:
                break
        
        return results
    
    def format_currency(self, val):
        """Format currency"""
        try:
            val = float(val)
            if val >= 1:
                return f"${val:,.2f}"
            else:
                return f"${val:.6f}"
        except:
            return "N/A"
    
    def get_coin_data(self, coin_id):
        """Get data coin dari CoinGecko API"""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}?localization=false&tickers=false&market_data=true"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                market_data = data.get('market_data', {})
                
                return {
                    'current_price': market_data.get('current_price', {}).get('usd', 0),
                    'price_change_24h': market_data.get('price_change_percentage_24h', 0),
                    'price_change_7d': market_data.get('price_change_percentage_7d', 0),
                    'high_24h': market_data.get('high_24h', {}).get('usd', 0),
                    'low_24h': market_data.get('low_24h', {}).get('usd', 0),
                    'market_cap': market_data.get('market_cap', {}).get('usd', 0),
                    'market_cap_rank': data.get('market_cap_rank', 9999),
                    'volume_24h': market_data.get('total_volume', {}).get('usd', 0),
                    'name': data.get('name', ''),
                    'symbol': data.get('symbol', '').upper(),
                }
            return None
        except:
            return None
    
    def calculate_simple_analysis(self, coin_data):
        """Calculate simple analysis tanpa complex dependencies"""
        if not coin_data:
            return "HOLD", "Data tidak tersedia", 50
        
        price_change_24h = coin_data.get('price_change_24h', 0)
        price_change_7d = coin_data.get('price_change_7d', 0)
        market_cap_rank = coin_data.get('market_cap_rank', 9999)
        
        # Simple scoring system
        score = 50
        
        # Price momentum (40%)
        if price_change_24h > 5 and price_change_7d > 10:
            score += 30
        elif price_change_24h > 2 and price_change_7d > 5:
            score += 15
        elif price_change_24h < -5 and price_change_7d < -10:
            score -= 30
        elif price_change_24h < -2 and price_change_7d < -5:
            score -= 15
        
        # Market position (30%)
        if market_cap_rank <= 50:
            score += 20
        elif market_cap_rank <= 200:
            score += 10
        else:
            score -= 5
        
        # Volume consideration (30%)
        volume = coin_data.get('volume_24h', 0)
        market_cap = coin_data.get('market_cap', 1)
        volume_ratio = volume / market_cap
        
        if volume_ratio > 0.1:
            score += 15
        elif volume_ratio > 0.05:
            score += 5
        else:
            score -= 10
        
        # Generate recommendation
        if score >= 75:
            action = "STRONG BUY"
            explanation = "📈 Momentum harga sangat kuat dengan volume tinggi"
            confidence = min(score, 95)
        elif score >= 65:
            action = "BUY"
            explanation = "📈 Kondisi positif dengan momentum baik"
            confidence = score
        elif score >= 45:
            action = "HOLD"
            explanation = "⚖️ Market dalam kondisi netral"
            confidence = 50
        elif score >= 35:
            action = "SELL"
            explanation = "📉 Momentum negatif dengan volume rendah"
            confidence = 70 - score
        else:
            action = "STRONG SELL"
            explanation = "📉 Kondisi bearish kuat"
            confidence = 95 - score
        
        return action, explanation, confidence

def main():
    st.markdown('<h1 class="main-header">🚀 Crypto Analysis Pro</h1>', unsafe_allow_html=True)
    
    analyzer = CryptoAnalyzer()
    
    # Search Section
    st.subheader("🔍 Search All Cryptocurrencies")
    
    # Session state management
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
    if 'selected_coin' not in st.session_state:
        st.session_state.selected_coin = None
    
    # Search input
    search_query = st.text_input(
        "Cari coin (nama, simbol, atau ID):",
        value=st.session_state.search_query,
        placeholder="Contoh: bitcoin, btc, ethereum, eth...",
        key="search_input"
    )
    
    st.session_state.search_query = search_query
    
    # Search results
    if search_query:
        with st.spinner("Mencari coin..."):
            results = analyzer.search_coins(search_query, limit=30)
        
        if results:
            st.subheader(f"📋 Hasil Pencarian ({len(results)} coin ditemukan)")
            st.info("💡 Klik coin untuk menganalisis!")
            
            # Display results
            cols = st.columns(3)
            for i, coin in enumerate(results):
                with cols[i % 3]:
                    coin_id = coin['id']
                    coin_symbol = coin['symbol'].upper()
                    coin_name = coin['name']
                    
                    if st.button(
                        f"**{coin_symbol}**\n{coin_name}",
                        key=f"coin_{coin_id}",
                        use_container_width=True
                    ):
                        st.session_state.selected_coin = coin
                        st.session_state.search_query = coin_id
                        st.rerun()
        else:
            st.warning("❌ Tidak ada coin yang ditemukan.")
    
    # Analysis Section
    if st.session_state.selected_coin:
        coin_info = st.session_state.selected_coin
        coin_id = coin_info['id']
        coin_name = coin_info['name']
        coin_symbol = coin_info['symbol'].upper()
        
        st.markdown("---")
        st.subheader(f"📊 Analisis {coin_name} ({coin_symbol})")
        
        if st.button("🔄 Analyze Now", type="primary"):
            with st.spinner("Mengambil data..."):
                coin_data = analyzer.get_coin_data(coin_id)
                
                if coin_data:
                    action, explanation, confidence = analyzer.calculate_simple_analysis(coin_data)
                    
                    # Display results
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h2>🎯 Rekomendasi: <span class="{'positive' if 'BUY' in action else 'negative' if 'SELL' in action else 'neutral'}">{action}</span></h2>
                        <p><strong>Confidence:</strong> {confidence}%</p>
                        <p><strong>Alasan:</strong> {explanation}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Market Data
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Harga",
                            analyzer.format_currency(coin_data['current_price']),
                            f"{coin_data['price_change_24h']:.2f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "Market Cap Rank",
                            f"#{coin_data['market_cap_rank']}"
                        )
                    
                    with col3:
                        st.metric(
                            "Volume 24h",
                            analyzer.format_currency(coin_data['volume_24h'])
                        )
                    
                    with col4:
                        st.metric(
                            "Market Cap",
                            analyzer.format_currency(coin_data['market_cap'])
                        )
                    
                    # Additional Info
                    with st.expander("📈 Detail Analisis"):
                        st.write(f"**Perubahan 7 Hari:** {coin_data['price_change_7d']:.2f}%")
                        st.write(f"**Harga Tertinggi 24h:** {analyzer.format_currency(coin_data['high_24h'])}")
                        st.write(f"**Harga Terendah 24h:** {analyzer.format_currency(coin_data['low_24h'])}")
                        
                        # Simple price targets
                        if "BUY" in action:
                            current_price = coin_data['current_price']
                            st.info(f"""
                            **🎯 Target Harga:**
                            - Target 1: {analyzer.format_currency(current_price * 1.05)} (+5%)
                            - Target 2: {analyzer.format_currency(current_price * 1.10)} (+10%)
                            - Stop Loss: {analyzer.format_currency(current_price * 0.95)} (-5%)
                            """)
                else:
                    st.error("❌ Gagal mengambil data.")
    
    # Clear button
    if st.session_state.selected_coin:
        if st.button("🗑️ Clear Selection"):
            st.session_state.selected_coin = None
            st.session_state.search_query = ""
            st.rerun()
    
    # Instructions
    with st.expander("ℹ️ Cara Menggunakan"):
        st.write("""
        1. **Ketik nama/simbol coin** di search box
        2. **Klik coin** yang ingin dianalisis  
        3. **Klik 'Analyze Now'** untuk mendapatkan rekomendasi
        4. **Berdasarkan analisis:** harga, volume, market cap, dan momentum
        """)

if __name__ == "__main__":
    main()
