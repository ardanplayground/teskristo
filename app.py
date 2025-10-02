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
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="AI Crypto Prediction Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .positive { color: #00aa00; font-weight: bold; }
    .negative { color: #ff0000; font-weight: bold; }
    .neutral { color: #ffaa00; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class AdvancedCryptoAnalyzer:
    def __init__(self):
        self.news_sources = [
            "coindesk", "cointelegraph", "decrypt", "theblock"
        ]
    
    def format_rupiah(self, val):
        """Format currency to Rupiah"""
        try:
            val = float(val)
            return "Rp {:,.0f}".format(val).replace(",", ".")
        except:
            return "N/A"
    
    def get_coin_data(self, coin_symbol):
        """Get comprehensive coin data from multiple sources"""
        try:
            # Convert to yfinance format
            yf_symbol = f"{coin_symbol.upper()}-USD"
            ticker = yf.Ticker(yf_symbol)
            
            # Get historical data
            hist = ticker.history(period="6mo")
            
            if hist.empty:
                return None
                
            # Calculate technical indicators
            hist = self.add_technical_indicators(hist)
            
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2]
            
            price_change_24h = ((current_price - prev_price) / prev_price) * 100
            price_change_7d = ((current_price - hist['Close'].iloc[-8]) / hist['Close'].iloc[-8]) * 100
            price_change_30d = ((current_price - hist['Close'].iloc[-31]) / hist['Close'].iloc[-31]) * 100
            
            return {
                'current_price': current_price,
                'price_change_24h': price_change_24h,
                'price_change_7d': price_change_7d,
                'price_change_30d': price_change_30d,
                'volume_24h': hist['Volume'].iloc[-1],
                'high_24h': hist['High'].iloc[-1],
                'low_24h': hist['Low'].iloc[-1],
                'historical_data': hist,
                'market_cap': current_price * 1e9,  # Estimate
                'technical_indicators': self.get_technical_signals(hist)
            }
        except Exception as e:
            st.error(f"Error getting coin data: {e}")
            return None
    
    def add_technical_indicators(self, df):
        """Add comprehensive technical indicators"""
        # Moving Averages
        df['MA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['MA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['MA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
        
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
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # Volume indicators
        df['Volume_SMA'] = ta.trend.sma_indicator(df['Volume'], window=20)
        
        return df
    
    def get_technical_signals(self, df):
        """Generate technical analysis signals"""
        latest = df.iloc[-1]
        
        signals = {
            'trend': 'NEUTRAL',
            'momentum': 'NEUTRAL',
            'volatility': 'MEDIUM',
            'support_resistance': 'NEUTRAL'
        }
        
        # Trend analysis
        if latest['MA_20'] > latest['MA_50'] > latest['MA_200']:
            signals['trend'] = 'BULLISH'
        elif latest['MA_20'] < latest['MA_50'] < latest['MA_200']:
            signals['trend'] = 'BEARISH'
        
        # Momentum analysis
        if latest['RSI'] < 30:
            signals['momentum'] = 'OVERSOLD'
        elif latest['RSI'] > 70:
            signals['momentum'] = 'OVERBOUGHT'
        elif latest['RSI'] > 50:
            signals['momentum'] = 'BULLISH'
        else:
            signals['momentum'] = 'BEARISH'
        
        # MACD signal
        if latest['MACD'] > latest['MACD_Signal'] and latest['MACD_Histogram'] > 0:
            signals['macd'] = 'BULLISH'
        else:
            signals['macd'] = 'BEARISH'
        
        return signals
    
    def get_news_sentiment(self, coin_name):
        """Get news sentiment analysis (simulated with AI logic)"""
        # In real implementation, integrate with news API
        sentiments = []
        
        # Simulate news analysis
        news_items = [
            f"{coin_name} shows strong technical breakout",
            f"Market analysts bullish on {coin_name}",
            f"{coin_name} faces resistance at key level",
            f"Institutional adoption of {coin_name} growing",
            f"Regulatory concerns for {coin_name}"
        ]
        
        for news in news_items:
            analysis = TextBlob(news)
            sentiments.append(analysis.sentiment.polarity)
        
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        sentiment_score = (avg_sentiment + 1) * 50  # Convert to 0-100 scale
        
        return {
            'score': sentiment_score,
            'trend': 'BULLISH' if sentiment_score > 60 else 'BEARISH' if sentiment_score < 40 else 'NEUTRAL',
            'confidence': min(abs(sentiment_score - 50) * 2, 100)
        }
    
    def ml_price_prediction(self, historical_data):
        """Machine Learning price prediction"""
        try:
            df = historical_data.copy()
            
            # Create features
            df['Price_Change'] = df['Close'].pct_change()
            df['Volume_Change'] = df['Volume'].pct_change()
            df['High_Low_Ratio'] = df['High'] / df['Low']
            
            # Technical indicators as features
            df['RSI'] = ta.momentum.rsi(df['Close'])
            df['MACD'] = ta.trend.MACD(df['Close']).macd()
            
            # Create target (1 if price increases next day, 0 otherwise)
            df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            
            # Prepare features
            feature_columns = ['RSI', 'MACD', 'Price_Change', 'Volume_Change', 'High_Low_Ratio']
            df = df.dropna()
            
            if len(df) < 50:
                return None
            
            X = df[feature_columns]
            y = df['Target']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Predict
            latest_features = scaler.transform([X.iloc[-1][feature_columns]])
            prediction = model.predict_proba(latest_features)[0]
            
            return {
                'up_probability': prediction[1] * 100,
                'down_probability': prediction[0] * 100,
                'confidence': max(prediction) * 100
            }
            
        except Exception as e:
            st.error(f"ML prediction error: {e}")
            return None
    
    def generate_trading_recommendation(self, coin_data, sentiment, ml_prediction):
        """Generate AI-powered trading recommendation"""
        if not coin_data:
            return "UNAVAILABLE", "Data tidak cukup untuk analisis", 0
        
        technical_signals = coin_data['technical_indicators']
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
        
        # Sentiment analysis weight: 30%
        if sentiment['trend'] == 'BULLISH':
            score += 15
        elif sentiment['trend'] == 'BEARISH':
            score -= 15
        
        # ML prediction weight: 30%
        if ml_prediction:
            if ml_prediction['up_probability'] > 60:
                score += 15
            elif ml_prediction['down_probability'] > 60:
                score -= 15
        
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
        if action in ["STRONG BUY", "BUY"]:
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
                'ml_prediction': ml_prediction
            }
        }
        
        return action, explanation, confidence, recommendation_details

def create_technical_chart(historical_data, coin_data):
    """Create advanced technical analysis chart"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price and Moving Averages', 'RSI', 'MACD'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Price and MA
    fig.add_trace(
        go.Candlestick(
            x=historical_data.index,
            open=historical_data['Open'],
            high=historical_data['High'],
            low=historical_data['Low'],
            close=historical_data['Close'],
            name='Price'
        ), row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=historical_data.index, y=historical_data['MA_20'], 
                  name='MA 20', line=dict(color='orange')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=historical_data.index, y=historical_data['MA_50'], 
                  name='MA 50', line=dict(color='green')),
        row=1, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=historical_data.index, y=historical_data['RSI'], 
                  name='RSI', line=dict(color='purple')),
        row=2, col=1
    )
    
    # RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    fig.add_trace(
        go.Scatter(x=historical_data.index, y=historical_data['MACD'], 
                  name='MACD', line=dict(color='blue')),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=historical_data.index, y=historical_data['MACD_Signal'], 
                  name='Signal', line=dict(color='red')),
        row=3, col=1
    )
    
    fig.update_layout(
        height=800,
        title_text="Technical Analysis",
        xaxis_rangeslider_visible=False
    )
    
    return fig

# Main Application
def main():
    st.markdown('<h1 class="main-header">🤖 AI Crypto Prediction Pro</h1>', unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = AdvancedCryptoAnalyzer()
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Coin selection
    popular_coins = {
        "BTC": "bitcoin",
        "ETH": "ethereum", 
        "BNB": "binancecoin",
        "ADA": "cardano",
        "DOT": "polkadot",
        "SOL": "solana",
        "DOGE": "dogecoin",
        "XRP": "ripple"
    }
    
    selected_coin = st.sidebar.selectbox(
        "Pilih Coin:",
        list(popular_coins.keys()),
        format_func=lambda x: f"{x} ({popular_coins[x]})"
    )
    
    coin_symbol = popular_coins[selected_coin]
    
    # Analysis parameters
    st.sidebar.subheader("Analysis Parameters")
    use_ml = st.sidebar.checkbox("Gunakan Machine Learning Prediction", value=True)
    use_sentiment = st.sidebar.checkbox("Gunakan News Sentiment Analysis", value=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📊 Analisis Real-time {selected_coin}")
        
        if st.button("🔄 Update Analysis", type="primary"):
            with st.spinner("Menganalisis data cryptocurrency..."):
                # Get coin data
                coin_data = analyzer.get_coin_data(coin_symbol)
                
                if coin_data:
                    # Get sentiment analysis
                    sentiment = analyzer.get_news_sentiment(selected_coin)
                    
                    # Get ML prediction
                    ml_prediction = None
                    if use_ml:
                        ml_prediction = analyzer.ml_price_prediction(coin_data['historical_data'])
                    
                    # Generate recommendation
                    action, explanation, confidence, details = analyzer.generate_trading_recommendation(
                        coin_data, sentiment, ml_prediction
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
                    
                    # Price targets if BUY recommendation
                    if "BUY" in action:
                        st.info(f"""
                        **🎯 Price Targets:**
                        - Target 1: {analyzer.format_rupiah(details['targets']['target_1'])} (+5%)
                        - Target 2: {analyzer.format_rupiah(details['targets']['target_2'])} (+10%)
                        - Stop Loss: {analyzer.format_rupiah(details['targets']['stop_loss'])} (-5%)
                        """)
                    
                    # Market Data
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Harga Sekarang",
                            analyzer.format_rupiah(coin_data['current_price']),
                            f"{coin_data['price_change_24h']:.2f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "Volume 24h",
                            analyzer.format_rupiah(coin_data['volume_24h'])
                        )
                    
                    with col3:
                        st.metric(
                            "Market Cap",
                            analyzer.format_rupiah(coin_data['market_cap'])
                        )
                    
                    # Technical Analysis
                    st.subheader("📈 Technical Analysis")
                    tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
                    
                    with tech_col1:
                        st.metric("Trend", details['score_breakdown']['technical']['trend'])
                        st.metric("RSI", f"{coin_data['historical_data']['RSI'].iloc[-1]:.1f}")
                    
                    with tech_col2:
                        st.metric("Momentum", details['score_breakdown']['technical']['momentum'])
                        st.metric("MACD Signal", details['score_breakdown']['technical']['macd'])
                    
                    with tech_col3:
                        st.metric("Sentiment", sentiment['trend'])
                        st.metric("Sentiment Score", f"{sentiment['score']:.1f}%")
                    
                    with tech_col4:
                        if ml_prediction:
                            st.metric("AI Prediction", f"{ml_prediction['up_probability']:.1f}% Up")
                            st.metric("AI Confidence", f"{ml_prediction['confidence']:.1f}%")
                    
                    # Charts
                    st.plotly_chart(
                        create_technical_chart(coin_data['historical_data'], coin_data),
                        use_container_width=True
                    )
                    
                    # Detailed Analysis
                    with st.expander("🔍 Detailed Analysis Breakdown"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Technical Signals")
                            for signal, value in details['score_breakdown']['technical'].items():
                                st.write(f"- **{signal.replace('_', ' ').title()}:** {value}")
                        
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
                
                else:
                    st.error("❌ Gagal mengambil data. Coba lagi nanti.")
    
    with col2:
        st.subheader("ℹ️ Cara Menggunakan")
        
        st.markdown("""
        ### 📖 Panduan Analisis
        
        **🟢 STRONG BUY:**
        - Semua indikator teknikal bullish
        - Sentimen pasar sangat positif
        - AI prediction confidence tinggi
        
        **🔵 BUY:**
        - Mayoritas indikator positif
        - Risk-reward ratio menguntungkan
        
        **🟡 HOLD:**
        - Market sideways/consolidation
        - Tunggu konfirmasi breakout
        
        **🔴 SELL:**
        - Indikator menunjukkan weakness
        - Pertimbangkan take profit
        
        **⚫ STRONG SELL:**
        - Trend bearish kuat
        - Hindari entry baru
        """)
        
        st.subheader("📊 Risk Management")
        st.info("""
        - Gunakan stop loss maksimal 5%
        - Risk tidak lebih dari 2% dari total portfolio per trade
        - Diversifikasi ke beberapa coin
        - Selalu update analysis sebelum entry
        """)

if __name__ == "__main__":
    main()
