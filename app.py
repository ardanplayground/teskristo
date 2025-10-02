import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Crypto Analysis Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
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
    .search-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class CoinGeckoManager:
    def __init__(self):
        self.all_coins = None
    
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
        if not self.all_coins:
            self.all_coins = self.get_all_coins()
        
        if not query:
            return []
        
        query = query.lower()
        results = []
        
        for coin in self.all_coins:
            coin_id = coin.get('id', '').lower()
            coin_symbol = coin.get('symbol', '').lower()
            coin_name = coin.get('name', '').lower()
            
            if (query in coin_id or query in coin_symbol or query in coin_name):
                results.append(coin)
            
            if len(results) >= limit:
                break
        
        return results

class CryptoAnalyzer:
    def __init__(self):
        self.cache = {}
        self.coin_gecko = CoinGeckoManager()
    
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
    
    def get_coin_data_from_coingecko(self, coin_id):
        """Get data coin dari CoinGecko API"""
        cache_key = f"coingecko_{coin_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}?localization=false&tickers=false&market_data=true&community_data=false&developer_data=false&sparkline=false"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                market_data = data.get('market_data', {})
                
                result = {
                    'current_price': market_data.get('current_price', {}).get('usd', 0),
                    'price_change_24h': market_data.get('price_change_percentage_24h', 0),
                    'price_change_7d': market_data.get('price_change_percentage_7d', 0),
                    'price_change_30d': market_data.get('price_change_percentage_30d', 0),
                    'high_24h': market_data.get('high_24h', {}).get('usd', 0),
                    'low_24h': market_data.get('low_24h', {}).get('usd', 0),
                    'market_cap': market_data.get('market_cap', {}).get('usd', 0),
                    'market_cap_rank': data.get('market_cap_rank', 9999),
                    'volume_24h': market_data.get('total_volume', {}).get('usd', 0),
                    'circulating_supply': market_data.get('circulating_supply', 0),
                    'total_supply': market_data.get('total_supply', 0),
                    'max_supply': market_data.get('max_supply', 0),
                    'ath': market_data.get('ath', {}).get('usd', 0),
                    'ath_change_percentage': market_data.get('ath_change_percentage', {}).get('usd', 0),
                    'name': data.get('name', ''),
                    'symbol': data.get('symbol', '').upper(),
                }
                
                self.cache[cache_key] = result
                return result
            else:
                return None
                
        except Exception as e:
            return None
    
    def calculate_technical_indicators(self, prices):
        """Calculate simple technical indicators manually"""
        if len(prices) < 20:
            return {}
        
        prices_array = np.array(prices)
        
        # Simple Moving Averages
        ma_7 = np.mean(prices_array[-7:]) if len(prices_array) >= 7 else prices_array[-1]
        ma_20 = np.mean(prices_array[-20:]) if len(prices_array) >= 20 else prices_array[-1]
        ma_50 = np.mean(prices_array[-50:]) if len(prices_array) >= 50 else prices_array[-1]
        
        # RSI Calculation
        def calculate_rsi(prices, period=14):
            if len(prices) < period + 1:
                return 50
            
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        rsi = calculate_rsi(prices_array)
        
        # Trend Analysis
        trend = "NEUTRAL"
        if ma_7 > ma_20 > ma_50:
            trend = "BULLISH"
        elif ma_7 < ma_20 < ma_50:
            trend = "BEARISH"
        
        # Momentum
        momentum = "NEUTRAL"
        if rsi > 70:
            momentum = "OVERBOUGHT"
        elif rsi < 30:
            momentum = "OVERSOLD"
        elif rsi > 50:
            momentum = "BULLISH"
        else:
            momentum = "BEARISH"
        
        # Support & Resistance (simplified)
        support = np.min(prices_array[-20:])
        resistance = np.max(prices_array[-20:])
        current_price = prices_array[-1]
        
        support_resistance = "NEUTRAL"
        if (current_price - support) / current_price < 0.02:  # Within 2% of support
            support_resistance = "NEAR_SUPPORT"
        elif (resistance - current_price) / current_price < 0.02:  # Within 2% of resistance
            support_resistance = "NEAR_RESISTANCE"
        
        return {
            'trend': trend,
            'momentum': momentum,
            'rsi': round(rsi, 2),
            'ma_7': ma_7,
            'ma_20': ma_20,
            'ma_50': ma_50,
            'support': support,
            'resistance': resistance,
            'support_resistance': support_resistance,
            'current_price': current_price
        }
    
    def get_historical_data(self, coin_id, days=90):
        """Get historical data untuk technical analysis"""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                prices = [price[1] for price in data.get('prices', [])]
                
                if len(prices) < 20:
                    return None
                
                return prices
            else:
                return None
                
        except Exception as e:
            return None
    
    def get_fundamental_analysis(self, coin_data):
        """Fundamental analysis sederhana"""
        if not coin_data:
            return {'score': 50, 'risk_level': 'MEDIUM'}
        
        score = 50
        
        # Market Cap Rank
        market_cap_rank = coin_data.get('market_cap_rank', 9999)
        if market_cap_rank <= 50:
            score += 20
        elif market_cap_rank <= 200:
            score += 10
        else:
            score -= 5
        
        # Volume Analysis
        market_cap = coin_data.get('market_cap', 1)
        volume_24h = coin_data.get('volume_24h', 0)
        volume_ratio = volume_24h / market_cap if market_cap > 0 else 0
        
        if volume_ratio > 0.1:
            score += 15
        elif volume_ratio > 0.05:
            score += 5
        
        # Supply Analysis
        circulating = coin_data.get('circulating_supply', 0)
        max_supply = coin_data.get('max_supply', 0)
        
        if max_supply and circulating > 0:
            inflation_rate = (max_supply - circulating) / max_supply
            if inflation_rate < 0.1:
                score += 10
        
        # Risk Level
        if score >= 70:
            risk_level = 'LOW'
        elif score >= 55:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'HIGH'
        
        return {
            'score': score,
            'risk_level': risk_level,
            'market_cap_rank': market_cap_rank,
            'volume_ratio': round(volume_ratio, 4)
        }
    
    def generate_recommendation(self, coin_data, technical_indicators, fundamental_analysis):
        """Generate trading recommendation"""
        if not coin_data or not technical_indicators:
            return "UNAVAILABLE", "Data tidak cukup", 50, {}
        
        score = 50
        
        # Technical Analysis (60%)
        if technical_indicators['trend'] == 'BULLISH':
            score += 20
        elif technical_indicators['trend'] == 'BEARISH':
            score -= 20
        
        if technical_indicators['momentum'] == 'BULLISH':
            score += 15
        elif technical_indicators['momentum'] == 'OVERSOLD':
            score += 10
        elif technical_indicators['momentum'] == 'OVERBOUGHT':
            score -= 15
        
        if technical_indicators['support_resistance'] == 'NEAR_SUPPORT':
            score += 10
        elif technical_indicators['support_resistance'] == 'NEAR_RESISTANCE':
            score -= 10
        
        # Fundamental Analysis (40%)
        fundamental_score = fundamental_analysis['score']
        score += (fundamental_score - 50) * 0.4
        
        # Generate Recommendation
        if score >= 75:
            action = "STRONG BUY"
            explanation = "📈 Trend bullish kuat dengan fundamental baik"
            confidence = min(score, 95)
        elif score >= 65:
            action = "BUY"
            explanation = "📈 Kondisi teknikal positif mendukung entry"
            confidence = score
        elif score >= 45:
            action = "HOLD"
            explanation = "⚖️ Market dalam kondisi netral, tunggu konfirmasi"
            confidence = 50
        elif score >= 35:
            action = "SELL"
            explanation = "📉 Teknikal menunjukkan weakness"
            confidence = 70 - score
        else:
            action = "STRONG SELL"
            explanation = "📉 Kondisi bearish kuat, hindari entry"
            confidence = 95 - score
        
        # Price Targets
        current_price = coin_data['current_price']
        if action in ["STRONG BUY", "BUY"] and current_price > 0:
            if fundamental_analysis['risk_level'] == 'LOW':
                target_1 = current_price * 1.08
                target_2 = current_price * 1.15
                stop_loss = current_price * 0.92
            else:
                target_1 = current_price * 1.05
                target_2 = current_price * 1.10
                stop_loss = current_price * 0.95
        else:
            target_1 = target_2 = stop_loss = current_price
        
        return action, explanation, confidence, {
            'targets': {'target_1': target_1, 'target_2': target_2, 'stop_loss': stop_loss},
            'technical': technical_indicators,
            'fundamental': fundamental_analysis,
            'final_score': score
        }

def main():
    st.markdown('<h1 class="main-header">🚀 Crypto Analysis Pro</h1>', unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = CryptoAnalyzer()
    
    # Search Section dengan Auto-fill
    st.markdown('<div class="search-box">', unsafe_allow_html=True)
    st.subheader("🔍 Search All Cryptocurrencies")
    
    # Session state untuk search
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
    if 'selected_coin' not in st.session_state:
        st.session_state.selected_coin = None
    
    # Search input dengan value dari selected coin
    search_query = st.text_input(
        "Cari coin (nama, simbol, atau ID):",
        value=st.session_state.search_query,
        placeholder="Contoh: bitcoin, btc, ethereum, eth...",
        key="search_input"
    )
    
    # Update search query in session state
    st.session_state.search_query = search_query
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Search results dengan auto-fill functionality
    if search_query:
        with st.spinner("Mencari coin..."):
            results = analyzer.coin_gecko.search_coins(search_query, limit=30)
        
        if results:
            st.subheader(f"📋 Hasil Pencarian ({len(results)} coin ditemukan)")
            st.info("💡 Klik salah satu coin di bawah untuk langsung menganalisis!")
            
            # Display results in a grid
            cols = st.columns(3)
            
            for i, coin in enumerate(results):
                with cols[i % 3]:
                    coin_id = coin['id']
                    coin_symbol = coin['symbol'].upper()
                    coin_name = coin['name']
                    
                    # Create a clickable card
                    if st.button(
                        f"**{coin_symbol}**\n{coin_name}",
                        key=f"coin_{coin_id}",
                        use_container_width=True,
                        help=f"Klik untuk analisis {coin_name}"
                    ):
                        # Auto-fill search box dan set selected coin
                        st.session_state.selected_coin = coin
                        st.session_state.search_query = coin['id']  # Auto-fill dengan coin ID
                        st.experimental_rerun()
        
        else:
            st.warning("❌ Tidak ada coin yang ditemukan. Coba kata kunci lain.")
    
    # Display analysis for selected coin
    if 'selected_coin' in st.session_state and st.session_state.selected_coin:
        coin_info = st.session_state.selected_coin
        coin_id = coin_info['id']
        coin_name = coin_info['name']
        coin_symbol = coin_info['symbol'].upper()
        
        st.markdown("---")
        st.subheader(f"📊 Analisis untuk {coin_name} ({coin_symbol})")
        
        # Tampilkan current search query yang terisi otomatis
        st.info(f"🔍 Coin yang dianalisis: **{coin_name}** ({coin_symbol})")
        
        if st.button("🔄 Analyze Now", type="primary", key="analyze_button"):
            with st.spinner(f"Menganalisis {coin_name}..."):
                # Get all data
                coin_data = analyzer.get_coin_data_from_coingecko(coin_id)
                historical_prices = analyzer.get_historical_data(coin_id)
                
                if coin_data and historical_prices:
                    # Calculate technical indicators
                    technical_indicators = analyzer.calculate_technical_indicators(historical_prices)
                    fundamental_analysis = analyzer.get_fundamental_analysis(coin_data)
                    
                    # Generate recommendation
                    action, explanation, confidence, details = analyzer.generate_recommendation(
                        coin_data, technical_indicators, fundamental_analysis
                    )
                    
                    # Display results
                    st.markdown("---")
                    
                    # Recommendation Card
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h2>🎯 Rekomendasi Trading: <span class="{'positive' if 'BUY' in action else 'negative' if 'SELL' in action else 'neutral'}">{action}</span></h2>
                        <p><strong>Confidence Level:</strong> {confidence:.1f}%</p>
                        <p><strong>Penjelasan:</strong> {explanation}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Market Data
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Harga Sekarang",
                            analyzer.format_currency(coin_data['current_price']),
                            f"{coin_data.get('price_change_24h', 0):.2f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "Market Cap Rank",
                            f"#{coin_data.get('market_cap_rank', 'N/A')}"
                        )
                    
                    with col3:
                        st.metric(
                            "Volume 24h",
                            analyzer.format_currency(coin_data.get('volume_24h', 0))
                        )
                    
                    with col4:
                        st.metric(
                            "Market Cap",
                            analyzer.format_currency(coin_data.get('market_cap', 0))
                        )
                    
                    # Technical Analysis
                    st.subheader("📈 Technical Analysis")
                    tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
                    
                    with tech_col1:
                        st.metric("Trend", technical_indicators['trend'])
                        st.metric("RSI", technical_indicators['rsi'])
                    
                    with tech_col2:
                        st.metric("Momentum", technical_indicators['momentum'])
                        st.metric("Support/Resistance", technical_indicators['support_resistance'])
                    
                    with tech_col3:
                        st.metric("Market Strength", "STRONG" if fundamental_analysis['score'] >= 70 else "GOOD" if fundamental_analysis['score'] >= 55 else "NEUTRAL")
                        st.metric("Risk Level", fundamental_analysis['risk_level'])
                    
                    with tech_col4:
                        st.metric("Fundamental Score", f"{fundamental_analysis['score']}/100")
                        st.metric("Volume Ratio", f"{fundamental_analysis.get('volume_ratio', 0):.2%}")
                    
                    # Price Chart (simple)
                    if len(historical_prices) >= 20:
                        st.subheader("📊 Price Chart (90 Days)")
                        chart_data = pd.DataFrame({
                            'Price': historical_prices,
                            'MA_7': [np.mean(historical_prices[max(0, i-7):i+1]) for i in range(len(historical_prices))],
                            'MA_20': [np.mean(historical_prices[max(0, i-20):i+1]) for i in range(len(historical_prices))]
                        })
                        st.line_chart(chart_data)
                    
                    # Price targets jika BUY
                    if "BUY" in action and coin_data['current_price'] > 0:
                        st.info(f"""
                        **🎯 Price Targets:**
                        - Target 1: {analyzer.format_currency(details['targets']['target_1'])} (+{((details['targets']['target_1']/coin_data['current_price'])-1)*100:.1f}%)
                        - Target 2: {analyzer.format_currency(details['targets']['target_2'])} (+{((details['targets']['target_2']/coin_data['current_price'])-1)*100:.1f}%)
                        - Stop Loss: {analyzer.format_currency(details['targets']['stop_loss'])} (-{((1-details['targets']['stop_loss']/coin_data['current_price']))*100:.1f}%)
                        """)
                    
                    # Detailed Analysis
                    with st.expander("🔍 Detailed Analysis"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Technical Indicators")
                            st.write(f"- **MA 7:** {analyzer.format_currency(technical_indicators['ma_7'])}")
                            st.write(f"- **MA 20:** {analyzer.format_currency(technical_indicators['ma_20'])}")
                            st.write(f"- **MA 50:** {analyzer.format_currency(technical_indicators['ma_50'])}")
                            st.write(f"- **Support:** {analyzer.format_currency(technical_indicators['support'])}")
                            st.write(f"- **Resistance:** {analyzer.format_currency(technical_indicators['resistance'])}")
                        
                        with col2:
                            st.subheader("Fundamental Analysis")
                            st.write(f"- **Market Cap Rank:** #{fundamental_analysis['market_cap_rank']}")
                            st.write(f"- **Volume/MCap Ratio:** {fundamental_analysis.get('volume_ratio', 0):.2%}")
                            st.write(f"- **Circulating Supply:** {coin_data.get('circulating_supply', 0):,.0f}")
                            st.write(f"- **Max Supply:** {coin_data.get('max_supply', 'Unlimited')}")
                            st.write(f"- **From ATH:** {coin_data.get('ath_change_percentage', 0):.1f}%")
                
                else:
                    st.error("❌ Gagal mengambil data coin. Coba lagi nanti.")
    
    # Clear selection button
    if st.session_state.get('selected_coin'):
        if st.button("🗑️ Clear Selection", type="secondary"):
            st.session_state.selected_coin = None
            st.session_state.search_query = ""
            st.experimental_rerun()
    
    # Instructions
    with st.expander("ℹ️ Cara Menggunakan"):
        st.markdown("""
        ### 🔍 Cara Menggunakan:
        1. **Ketik nama/simbol coin** di search box
        2. **Klik coin** yang ingin dianalisis (akan auto-fill)
        3. **Klik 'Analyze Now'** untuk prediksi
        
        ### 📊 Analisis yang Digunakan:
        - **Technical Analysis**: Trend, Momentum, RSI, Support/Resistance
        - **Fundamental Analysis**: Market cap, Volume, Supply metrics
        - **Risk Assessment**: Risk level dan position sizing
        
        ### ⚠️ Disclaimer:
        Untuk edukasi saja. Selalu lakukan research mandiri!
        """)

if __name__ == "__main__":
    main()
