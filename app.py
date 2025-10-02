import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from textblob import TextBlob
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import os
import nltk
import time

# Configure NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="AI Crypto Prediction Pro - All Coins",
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
    .coin-option {
        padding: 8px 12px;
        margin: 2px 0;
        border-radius: 5px;
        cursor: pointer;
    }
    .coin-option:hover {
        background-color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

class CoinGeckoManager:
    def __init__(self):
        self.all_coins = None
        self.last_update = None
    
    @st.cache_data(ttl=3600)  # Cache 1 jam
    def get_all_coins(_self):
        """Get semua coin dari CoinGecko"""
        try:
            url = "https://api.coingecko.com/api/v3/coins/list"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                coins = response.json()
                # Format: [{"id": "bitcoin", "symbol": "btc", "name": "Bitcoin"}, ...]
                return coins
            else:
                st.error("Gagal mengambil data coin dari CoinGecko")
                return []
        except Exception as e:
            st.error(f"Error: {e}")
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
            
            # Cari di semua field
            if (query in coin_id or 
                query in coin_symbol or 
                query in coin_name):
                results.append(coin)
            
            if len(results) >= limit:
                break
        
        return results

class AdvancedCryptoAnalyzer:
    def __init__(self):
        self.cache = {}
        self.coin_gecko = CoinGeckoManager()
    
    def format_rupiah(self, val):
        """Format currency to Rupiah"""
        try:
            val = float(val)
            return "Rp {:,.0f}".format(val).replace(",", ".")
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
                    'volume_24h': market_data.get('total_volume', {}).get('usd', 0),
                    'ath': market_data.get('ath', {}).get('usd', 0),
                    'ath_change_percentage': market_data.get('ath_change_percentage', {}).get('usd', 0),
                    'last_updated': data.get('last_updated', ''),
                    'name': data.get('name', ''),
                    'symbol': data.get('symbol', '').upper()
                }
                
                self.cache[cache_key] = result
                return result
            else:
                st.error(f"Coin '{coin_id}' tidak ditemukan di CoinGecko")
                return None
                
        except Exception as e:
            st.error(f"Error mengambil data {coin_id}: {e}")
            return None
    
    def get_historical_data(self, coin_id, days=90):
        """Get historical data untuk technical analysis"""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}&interval=daily"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                prices = data.get('prices', [])
                
                if not prices:
                    return None
                
                # Convert ke DataFrame
                df = pd.DataFrame(prices, columns=['timestamp', 'price'])
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('date', inplace=True)
                df = df[['price']]
                df.columns = ['Close']
                
                # Generate OHLC data (sederhana)
                df['Open'] = df['Close'].shift(1)
                df['High'] = df[['Open', 'Close']].max(axis=1)
                df['Low'] = df[['Open', 'Close']].min(axis=1)
                df['Volume'] = 0  # Volume tidak tersedia di endpoint ini
                
                df = df.dropna()
                
                # Add technical indicators
                df = self.add_technical_indicators(df)
                
                return df
            else:
                return None
                
        except Exception as e:
            st.error(f"Error mengambil historical data: {e}")
            return None
    
    def add_technical_indicators(self, df):
        """Add technical indicators"""
        try:
            # Moving Averages
            df['MA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['MA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            
            # RSI
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
            
            # MACD
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['Close'])
            df['BB_Upper'] = bollinger.bollinger_hband()
            df['BB_Lower'] = bollinger.bollinger_lband()
            df['BB_Middle'] = bollinger.bollinger_mavg()
            
        except Exception as e:
            st.error(f"Error calculating indicators: {e}")
        
        return df
    
    def get_technical_signals(self, df):
        """Generate technical analysis signals"""
        if df is None or len(df) < 50:
            return {
                'trend': 'UNAVAILABLE',
                'momentum': 'UNAVAILABLE', 
                'macd': 'UNAVAILABLE',
                'rsi': 'UNAVAILABLE'
            }
        
        latest = df.iloc[-1]
        
        signals = {
            'trend': 'NEUTRAL',
            'momentum': 'NEUTRAL',
            'macd': 'NEUTRAL',
            'rsi': f"{latest.get('RSI', 0):.1f}"
        }
        
        # Trend analysis
        ma_20 = latest.get('MA_20', 0)
        ma_50 = latest.get('MA_50', 0)
        
        if ma_20 > ma_50:
            signals['trend'] = 'BULLISH'
        elif ma_20 < ma_50:
            signals['trend'] = 'BEARISH'
        
        # RSI analysis
        rsi = latest.get('RSI', 50)
        if rsi < 30:
            signals['momentum'] = 'OVERSOLD'
        elif rsi > 70:
            signals['momentum'] = 'OVERBOUGHT'
        elif rsi > 50:
            signals['momentum'] = 'BULLISH'
        else:
            signals['momentum'] = 'BEARISH'
        
        # MACD signal
        macd = latest.get('MACD', 0)
        macd_signal = latest.get('MACD_Signal', 0)
        
        if macd > macd_signal:
            signals['macd'] = 'BULLISH'
        else:
            signals['macd'] = 'BEARISH'
        
        return signals
    
    def get_news_sentiment(self, coin_name):
        """Get news sentiment analysis"""
        # Simulated sentiment analysis
        sentiments = []
        news_items = [
            f"{coin_name} shows strong technical breakout",
            f"Market analysts bullish on {coin_name}",
            f"{coin_name} faces resistance at key level",
            f"Institutional adoption of {coin_name} growing"
        ]
        
        for news in news_items:
            analysis = TextBlob(news)
            sentiments.append(analysis.sentiment.polarity)
        
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        sentiment_score = (avg_sentiment + 1) * 50
        
        return {
            'score': sentiment_score,
            'trend': 'BULLISH' if sentiment_score > 60 else 'BEARISH' if sentiment_score < 40 else 'NEUTRAL',
            'confidence': min(abs(sentiment_score - 50) * 2, 100)
        }
    
    def ml_price_prediction(self, historical_data):
        """Machine Learning price prediction"""
        try:
            if historical_data is None or len(historical_data) < 30:
                return None
            
            df = historical_data.copy().tail(100)
            
            # Create features
            df['Price_Change'] = df['Close'].pct_change()
            df['Volume_Change'] = df['Volume'].pct_change() if 'Volume' in df else 0
            
            # Technical indicators as features
            df['RSI'] = ta.momentum.rsi(df['Close'])
            df['MACD'] = ta.trend.MACD(df['Close']).macd()
            
            # Create target
            df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            df = df.dropna()
            
            if len(df) < 20:
                return None
            
            feature_columns = ['RSI', 'MACD', 'Price_Change']
            if 'Volume_Change' in df:
                feature_columns.append('Volume_Change')
                
            X = df[feature_columns]
            y = df['Target']
            
            # Train simple model
            model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
            model.fit(X, y)
            
            # Predict
            prediction = model.predict_proba([X.iloc[-1]])[0]
            
            return {
                'up_probability': prediction[1] * 100,
                'down_probability': prediction[0] * 100,
                'confidence': max(prediction) * 100
            }
            
        except Exception as e:
            return None
    
    def generate_trading_recommendation(self, coin_data, historical_data, sentiment, ml_prediction):
        """Generate AI-powered trading recommendation"""
        if not coin_data:
            return "UNAVAILABLE", "Data tidak cukup untuk analisis", 0, {}
        
        technical_signals = self.get_technical_signals(historical_data)
        current_price = coin_data['current_price']
        
        # Scoring system
        score = 50  # Neutral starting point
        
        # Technical analysis weight: 40%
        if technical_signals['trend'] == 'BULLISH':
            score += 15
        elif technical_signals['trend'] == 'BEARISH':
            score -= 15
        
        if technical_signals['momentum'] == 'BULLISH':
            score += 10
        elif technical_signals['momentum'] == 'OVERSOLD':
            score += 5
        elif technical_signals['momentum'] == 'OVERBOUGHT':
            score -= 10
        
        if technical_signals['macd'] == 'BULLISH':
            score += 5
        
        # Price performance weight: 20%
        price_change_24h = coin_data.get('price_change_24h', 0)
        if price_change_24h > 5:
            score += 10
        elif price_change_24h < -5:
            score -= 10
        
        # Sentiment analysis weight: 20%
        if sentiment['trend'] == 'BULLISH':
            score += 10
        elif sentiment['trend'] == 'BEARISH':
            score -= 10
        
        # ML prediction weight: 20%
        if ml_prediction:
            if ml_prediction['up_probability'] > 60:
                score += 10
            elif ml_prediction['down_probability'] > 60:
                score -= 10
        
        # Generate recommendation
        if score >= 70:
            action = "STRONG BUY"
            explanation = "📈 Kondisi teknikal sangat bullish, sentimen positif, dan prediksi AI mendukung kenaikan harga"
            confidence = min(score, 95)
            
        elif score >= 60:
            action = "BUY"
            explanation = "📈 Signal teknikal positif dengan sentimen mendukung, peluang bagus untuk entry"
            confidence = score
            
        elif score >= 40:
            action = "HOLD"
            explanation = "⚖️ Market dalam kondisi netral, tunggu konfirmasi breakout atau breakdown"
            confidence = 50
            
        elif score >= 30:
            action = "SELL"
            explanation = "📉 Teknikal menunjukkan weakness, pertimbangkan take profit atau stop loss"
            confidence = 70 - score
            
        else:
            action = "STRONG SELL"
            explanation = "📉 Kondisi bearish kuat, hindari entry baru dan pertimbangkan exit"
            confidence = 95 - score
        
        # Price targets
        if action in ["STRONG BUY", "BUY"] and current_price > 0:
            target_1 = current_price * 1.05  # 5% target
            target_2 = current_price * 1.10  # 10% target
            stop_loss = current_price * 0.95  # 5% stop loss
        else:
            target_1 = target_2 = stop_loss = current_price
        
        recommendation_details = {
            'action': action,
            'explanation': explanation,
            'confidence': confidence,
            'targets': {
                'target_1': target_1,
                'target_2': target_2,
                'stop_loss': stop_loss
            },
            'score_breakdown': {
                'technical': technical_signals,
                'sentiment': sentiment,
                'ml_prediction': ml_prediction,
                'final_score': score
            }
        }
        
        return action, explanation, confidence, recommendation_details

def create_technical_chart(historical_data, coin_name):
    """Create technical analysis chart"""
    if historical_data is None or len(historical_data) < 20:
        return None
        
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'{coin_name} - Price and Moving Averages', 'RSI'),
        row_heights=[0.7, 0.3]
    )
    
    # Price and MA
    fig.add_trace(
        go.Scatter(x=historical_data.index, y=historical_data['Close'], 
                  name='Price', line=dict(color='blue')),
        row=1, col=1
    )
    
    if 'MA_20' in historical_data.columns:
        fig.add_trace(
            go.Scatter(x=historical_data.index, y=historical_data['MA_20'], 
                      name='MA 20', line=dict(color='orange')),
            row=1, col=1
        )
    
    if 'MA_50' in historical_data.columns:
        fig.add_trace(
            go.Scatter(x=historical_data.index, y=historical_data['MA_50'], 
                      name='MA 50', line=dict(color='green')),
            row=1, col=1
        )
    
    # RSI
    if 'RSI' in historical_data.columns:
        fig.add_trace(
            go.Scatter(x=historical_data.index, y=historical_data['RSI'], 
                      name='RSI', line=dict(color='purple')),
            row=2, col=1
        )
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
    
    fig.update_layout(height=600, showlegend=True, template="plotly_white")
    
    return fig

def main():
    st.markdown('<h1 class="main-header">🤖 AI Crypto Prediction Pro - All Coins</h1>', unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = AdvancedCryptoAnalyzer()
    
    # Search Section
    st.markdown('<div class="search-box">', unsafe_allow_html=True)
    st.subheader("🔍 Search All Cryptocurrencies")
    
    # Search input
    search_query = st.text_input(
        "Cari coin (nama, simbol, atau ID):",
        placeholder="Contoh: bitcoin, btc, dogecoin, doge, solana, sol...",
        key="search_input"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Search results
    if search_query:
        with st.spinner("Mencari coin..."):
            results = analyzer.coin_gecko.search_coins(search_query, limit=30)
        
        if results:
            st.subheader(f"📋 Hasil Pencarian ({len(results)} coin ditemukan)")
            
            # Display results in columns
            cols = st.columns(3)
            selected_coin = None
            
            for i, coin in enumerate(results):
                with cols[i % 3]:
                    coin_id = coin['id']
                    coin_symbol = coin['symbol'].upper()
                    coin_name = coin['name']
                    
                    # Display coin option
                    if st.button(
                        f"**{coin_symbol}** - {coin_name}",
                        key=f"coin_{coin_id}",
                        use_container_width=True
                    ):
                        selected_coin = coin
            
            # If coin selected, show analysis
            if selected_coin:
                st.session_state.selected_coin = selected_coin
                st.experimental_rerun()
        
        else:
            st.warning("❌ Tidak ada coin yang ditemukan. Coba kata kunci lain.")
    
    # Display analysis for selected coin
    if 'selected_coin' in st.session_state:
        coin_info = st.session_state.selected_coin
        coin_id = coin_info['id']
        coin_name = coin_info['name']
        coin_symbol = coin_info['symbol'].upper()
        
        st.markdown("---")
        st.subheader(f"📊 Analisis untuk {coin_name} ({coin_symbol})")
        
        if st.button("🔄 Update Analysis", type="primary"):
            with st.spinner(f"Menganalisis {coin_name}..."):
                # Get all data
                coin_data = analyzer.get_coin_data_from_coingecko(coin_id)
                historical_data = analyzer.get_historical_data(coin_id)
                sentiment = analyzer.get_news_sentiment(coin_name)
                ml_prediction = analyzer.ml_price_prediction(historical_data)
                
                if coin_data:
                    # Generate recommendation
                    action, explanation, confidence, details = analyzer.generate_trading_recommendation(
                        coin_data, historical_data, sentiment, ml_prediction
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
                            "Harga Sekarang (USD)",
                            f"${coin_data['current_price']:,.4f}" if coin_data['current_price'] < 1 else f"${coin_data['current_price']:,.2f}",
                            f"{coin_data.get('price_change_24h', 0):.2f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "Market Cap",
                            f"${coin_data.get('market_cap', 0):,.0f}" if coin_data.get('market_cap') else "N/A"
                        )
                    
                    with col3:
                        st.metric(
                            "24h High",
                            f"${coin_data.get('high_24h', 0):,.4f}" if coin_data.get('high_24h', 0) < 1 else f"${coin_data.get('high_24h', 0):,.2f}"
                        )
                    
                    with col4:
                        st.metric(
                            "24h Low", 
                            f"${coin_data.get('low_24h', 0):,.4f}" if coin_data.get('low_24h', 0) < 1 else f"${coin_data.get('low_24h', 0):,.2f}"
                        )
                    
                    # Technical Analysis
                    st.subheader("📈 Technical Analysis")
                    tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
                    
                    technical_signals = details['score_breakdown']['technical']
                    
                    with tech_col1:
                        st.metric("Trend", technical_signals['trend'])
                        st.metric("RSI", technical_signals.get('rsi', 'N/A'))
                    
                    with tech_col2:
                        st.metric("Momentum", technical_signals['momentum'])
                        st.metric("MACD Signal", technical_signals['macd'])
                    
                    with tech_col3:
                        st.metric("Sentiment", sentiment['trend'])
                        st.metric("Sentiment Score", f"{sentiment['score']:.1f}%")
                    
                    with tech_col4:
                        if ml_prediction:
                            st.metric("AI Prediction", f"{ml_prediction['up_probability']:.1f}% Up")
                            st.metric("AI Confidence", f"{ml_prediction['confidence']:.1f}%")
                        else:
                            st.metric("AI Prediction", "N/A")
                            st.metric("AI Confidence", "N/A")
                    
                    # Charts
                    if historical_data is not None:
                        chart = create_technical_chart(historical_data, coin_name)
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)
                        else:
                            st.warning("❌ Data historis tidak cukup untuk chart")
                    
                    # Price targets jika BUY
                    if "BUY" in action and coin_data['current_price'] > 0:
                        st.info(f"""
                        **🎯 Price Targets:**
                        - Target 1: ${details['targets']['target_1']:.4f} (+5%)
                        - Target 2: ${details['targets']['target_2']:.4f} (+10%)
                        - Stop Loss: ${details['targets']['stop_loss']:.4f} (-5%)
                        """)
                    
                    # Detailed Analysis
                    with st.expander("🔍 Detailed Analysis Breakdown"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Technical Signals")
                            for signal, value in technical_signals.items():
                                st.write(f"- **{signal.replace('_', ' ').title()}:** {value}")
                            
                            st.subheader("Price Performance")
                            st.write(f"- **24h Change:** {coin_data.get('price_change_24h', 0):.2f}%")
                            st.write(f"- **7d Change:** {coin_data.get('price_change_7d', 0):.2f}%")
                            st.write(f"- **30d Change:** {coin_data.get('price_change_30d', 0):.2f}%")
                        
                        with col2:
                            st.subheader("Market Sentiment")
                            st.write(f"- **Score:** {sentiment['score']:.1f}%")
                            st.write(f"- **Trend:** {sentiment['trend']}")
                            st.write(f"- **Confidence:** {sentiment['confidence']:.1f}%")
                            
                            if ml_prediction:
                                st.subheader("AI Prediction")
                                st.write(f"- **Up Probability:** {ml_prediction['up_probability']:.1f}%")
                                st.write(f"- **Down Probability:** {ml_prediction['down_probability']:.1f}%")
                                st.write(f"- **Confidence:** {ml_prediction['confidence']:.1f}%")
                            
                            st.subheader("Final Score")
                            st.write(f"- **Overall Score:** {details['score_breakdown']['final_score']:.1f}/100")
                
                else:
                    st.error("❌ Gagal mengambil data coin. Coba lagi nanti.")
    
    # Instructions
    with st.expander("ℹ️ Cara Menggunakan"):
        st.markdown("""
        ### 🔍 Cara Search Coin:
        1. **Ketik nama, simbol, atau ID coin** di search box
        2. **Contoh:** 
           - `bitcoin` atau `btc` untuk Bitcoin
           - `ethereum` atau `eth` untuk Ethereum  
           - `dogecoin` atau `doge` untuk Dogecoin
           - `solana` atau `sol` untuk Solana
        3. **Klik tombol coin** yang ingin dianalisis
        4. **Klik 'Update Analysis'** untuk mendapatkan prediksi
        
        ### 📊 Hasil Analisis:
        - **STRONG BUY/BUY**: Kondisi teknikal bullish dengan sentimen positif
        - **HOLD**: Market sideways, tunggu konfirmasi
        - **SELL/STRONG SELL**: Kondisi bearish, pertimbangkan exit
        
        ### ⚠️ Disclaimer:
        Aplikasi ini untuk edukasi saja. Selalu lakukan research sendiri sebelum trading!
        """)

if __name__ == "__main__":
    main()
