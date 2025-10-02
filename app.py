import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="AI Crypto Prediction Pro - Technical Analysis",
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
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .coin-option {
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        border: 1px solid #ddd;
        cursor: pointer;
        transition: all 0.3s;
    }
    .coin-option:hover {
        background-color: #e3f2fd;
        border-color: #1f77b4;
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

class AdvancedCryptoAnalyzer:
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
                    'price_change_60d': market_data.get('price_change_percentage_60d', 0),
                    'price_change_1y': market_data.get('price_change_percentage_1y', 0),
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
                    'atl': market_data.get('atl', {}).get('usd', 0),
                    'atl_change_percentage': market_data.get('atl_change_percentage', {}).get('usd', 0),
                    'name': data.get('name', ''),
                    'symbol': data.get('symbol', '').upper(),
                    'last_updated': data.get('last_updated', '')
                }
                
                self.cache[cache_key] = result
                return result
            else:
                return None
                
        except Exception as e:
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
                
                # Generate OHLC data
                df['Open'] = df['Close'].shift(1)
                df['High'] = df[['Open', 'Close']].max(axis=1)
                df['Low'] = df[['Open', 'Close']].min(axis=1)
                df['Volume'] = 0
                
                df = df.dropna()
                
                # Add comprehensive technical indicators
                df = self.add_technical_indicators(df)
                
                return df
            else:
                return None
                
        except Exception as e:
            return None
    
    def add_technical_indicators(self, df):
        """Add comprehensive technical indicators"""
        try:
            # Trend Indicators
            df['MA_7'] = ta.trend.sma_indicator(df['Close'], window=7)
            df['MA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['MA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
            df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
            
            # Momentum Indicators
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
            df['RSI_Smoothed'] = ta.trend.sma_indicator(df['RSI'], window=3)
            df['Stoch_K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14)
            df['Stoch_D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'], window=14)
            df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14)
            df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20)
            
            # MACD
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Histogram'] = macd.macd_diff()
            
            # Volatility Indicators
            bollinger = ta.volatility.BollingerBands(df['Close'])
            df['BB_Upper'] = bollinger.bollinger_hband()
            df['BB_Lower'] = bollinger.bollinger_lband()
            df['BB_Middle'] = bollinger.bollinger_mavg()
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
            
            df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
            
            # Volume-based (simulated)
            df['Volume_MA'] = ta.trend.sma_indicator(df['Volume'], window=20)
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            
            # Support & Resistance
            df['Support'] = df['Close'].rolling(window=20).min()
            df['Resistance'] = df['Close'].rolling(window=20).max()
            df['Support_Distance'] = (df['Close'] - df['Support']) / df['Close'] * 100
            df['Resistance_Distance'] = (df['Resistance'] - df['Close']) / df['Close'] * 100
            
        except Exception as e:
            print(f"Error in technical indicators: {e}")
        
        return df
    
    def get_market_regime(self, df):
        """Determine market regime (Trending/Ranging)"""
        if df is None or len(df) < 50:
            return "UNKNOWN", 0
        
        # Volatility analysis
        volatility = df['Close'].pct_change().std() * np.sqrt(365)  # Annualized volatility
        adx = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14).iloc[-1]
        
        if adx > 25 and volatility > 0.5:
            return "TRENDING", adx
        elif adx < 20 and volatility < 0.3:
            return "RANGING", adx
        else:
            return "TRANSITION", adx
    
    def get_technical_signals(self, df):
        """Generate comprehensive technical analysis signals"""
        if df is None or len(df) < 50:
            return {
                'trend': 'UNAVAILABLE',
                'momentum': 'UNAVAILABLE', 
                'volatility': 'UNAVAILABLE',
                'market_regime': 'UNAVAILABLE',
                'support_resistance': 'UNAVAILABLE'
            }
        
        latest = df.iloc[-1]
        
        signals = {
            'trend': 'NEUTRAL',
            'momentum': 'NEUTRAL',
            'volatility': 'MEDIUM',
            'market_regime': 'UNKNOWN',
            'support_resistance': 'NEUTRAL',
            'rsi': f"{latest.get('RSI', 0):.1f}",
            'macd_signal': 'NEUTRAL',
            'bollinger_signal': 'NEUTRAL'
        }
        
        # Trend analysis (Multiple timeframe)
        ma_7 = latest.get('MA_7', 0)
        ma_20 = latest.get('MA_20', 0)
        ma_50 = latest.get('MA_50', 0)
        
        bullish_count = 0
        if ma_7 > ma_20: bullish_count += 1
        if ma_20 > ma_50: bullish_count += 1
        if latest.get('EMA_12', 0) > latest.get('EMA_26', 0): bullish_count += 1
        
        if bullish_count >= 2:
            signals['trend'] = 'BULLISH'
        elif bullish_count <= 1:
            signals['trend'] = 'BEARISH'
        
        # Momentum analysis
        rsi = latest.get('RSI', 50)
        stoch_k = latest.get('Stoch_K', 50)
        williams_r = latest.get('Williams_R', -50)
        
        momentum_score = 0
        if rsi > 50: momentum_score += 1
        if stoch_k > 50: momentum_score += 1
        if williams_r > -50: momentum_score += 1
        
        if momentum_score >= 2:
            signals['momentum'] = 'BULLISH'
        elif momentum_score <= 1:
            signals['momentum'] = 'BEARISH'
        
        # RSI specific signals
        if rsi < 30:
            signals['momentum'] = 'OVERSOLD'
        elif rsi > 70:
            signals['momentum'] = 'OVERBOUGHT'
        
        # MACD signal
        macd = latest.get('MACD', 0)
        macd_signal = latest.get('MACD_Signal', 0)
        macd_histogram = latest.get('MACD_Histogram', 0)
        
        if macd > macd_signal and macd_histogram > 0:
            signals['macd_signal'] = 'BULLISH'
        else:
            signals['macd_signal'] = 'BEARISH'
        
        # Bollinger Bands signal
        price = latest['Close']
        bb_upper = latest.get('BB_Upper', price)
        bb_lower = latest.get('BB_Lower', price)
        bb_middle = latest.get('BB_Middle', price)
        
        if price <= bb_lower:
            signals['bollinger_signal'] = 'OVERSOLD'
        elif price >= bb_upper:
            signals['bollinger_signal'] = 'OVERBOUGHT'
        elif price > bb_middle:
            signals['bollinger_signal'] = 'BULLISH'
        else:
            signals['bollinger_signal'] = 'BEARISH'
        
        # Support & Resistance
        support_dist = latest.get('Support_Distance', 0)
        resistance_dist = latest.get('Resistance_Distance', 0)
        
        if support_dist < 2:  # Near support
            signals['support_resistance'] = 'NEAR_SUPPORT'
        elif resistance_dist < 2:  # Near resistance
            signals['support_resistance'] = 'NEAR_RESISTANCE'
        
        # Market regime
        regime, adx = self.get_market_regime(df)
        signals['market_regime'] = regime
        signals['adx'] = f"{adx:.1f}"
        
        # Volatility
        atr = latest.get('ATR', 0)
        if atr > df['Close'].mean() * 0.05:  # 5% ATR
            signals['volatility'] = 'HIGH'
        elif atr < df['Close'].mean() * 0.02:  # 2% ATR
            signals['volatility'] = 'LOW'
        
        return signals
    
    def get_fundamental_analysis(self, coin_data):
        """Fundamental analysis berdasarkan on-chain dan market metrics"""
        if not coin_data:
            return {
                'market_strength': 'NEUTRAL',
                'supply_analysis': 'NEUTRAL',
                'risk_level': 'MEDIUM',
                'fundamental_score': 50
            }
        
        score = 50
        factors = {}
        
        # Market Cap Rank (semakin kecil semakin baik)
        market_cap_rank = coin_data.get('market_cap_rank', 9999)
        if market_cap_rank <= 50:
            score += 20
            factors['market_cap_rank'] = 'STRONG'
        elif market_cap_rank <= 200:
            score += 10
            factors['market_cap_rank'] = 'GOOD'
        else:
            score -= 10
            factors['market_cap_rank'] = 'WEAK'
        
        # Volume/Market Cap Ratio (likuiditas)
        market_cap = coin_data.get('market_cap', 1)
        volume_24h = coin_data.get('volume_24h', 0)
        volume_ratio = volume_24h / market_cap if market_cap > 0 else 0
        
        if volume_ratio > 0.1:  # 10% volume/mcap ratio
            score += 15
            factors['liquidity'] = 'HIGH'
        elif volume_ratio > 0.05:  # 5% volume/mcap ratio
            score += 5
            factors['liquidity'] = 'MEDIUM'
        else:
            score -= 10
            factors['liquidity'] = 'LOW'
        
        # Supply Analysis
        circulating = coin_data.get('circulating_supply', 0)
        total_supply = coin_data.get('total_supply', 0)
        max_supply = coin_data.get('max_supply', 0)
        
        if max_supply and circulating > 0:
            inflation_rate = (max_supply - circulating) / max_supply
            if inflation_rate < 0.1:  # Low inflation
                score += 10
                factors['inflation'] = 'LOW'
            else:
                score -= 5
                factors['inflation'] = 'HIGH'
        
        # Price from ATH (recovery potential)
        current_price = coin_data.get('current_price', 0)
        ath = coin_data.get('ath', current_price)
        from_ath = coin_data.get('ath_change_percentage', 0)
        
        if from_ath > -50:  # Less than 50% from ATH
            score += 10
            factors['recovery_potential'] = 'HIGH'
        elif from_ath > -80:  # Less than 80% from ATH
            score += 5
            factors['recovery_potential'] = 'MEDIUM'
        else:
            score -= 5
            factors['recovery_potential'] = 'LOW'
        
        # Determine overall strength
        if score >= 70:
            market_strength = 'STRONG'
            risk_level = 'LOW'
        elif score >= 55:
            market_strength = 'GOOD' 
            risk_level = 'MEDIUM'
        elif score >= 40:
            market_strength = 'NEUTRAL'
            risk_level = 'MEDIUM'
        else:
            market_strength = 'WEAK'
            risk_level = 'HIGH'
        
        return {
            'market_strength': market_strength,
            'supply_analysis': factors.get('inflation', 'NEUTRAL'),
            'risk_level': risk_level,
            'fundamental_score': score,
            'factors': factors
        }
    
    def ml_price_prediction(self, historical_data, coin_data):
        """Enhanced ML prediction dengan lebih banyak features"""
        try:
            if historical_data is None or len(historical_data) < 60:
                return None
            
            df = historical_data.copy().tail(100)
            
            # Create comprehensive features
            df['Price_Change_1d'] = df['Close'].pct_change(1)
            df['Price_Change_3d'] = df['Close'].pct_change(3)
            df['Price_Change_7d'] = df['Close'].pct_change(7)
            df['Volatility_7d'] = df['Close'].pct_change().rolling(7).std()
            
            # Technical features
            df['RSI'] = ta.momentum.rsi(df['Close'])
            df['MACD'] = ta.trend.MACD(df['Close']).macd()
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            df['Volume_Spike'] = df['Volume_Ratio'] > 1.5
            
            # Price position features
            df['Price_MA_Ratio'] = df['Close'] / df['MA_20']
            df['Support_Distance_Pct'] = (df['Close'] - df['Support']) / df['Close'] * 100
            df['Resistance_Distance_Pct'] = (df['Resistance'] - df['Close']) / df['Close'] * 100
            
            # Create target (1 if price increases 5% in next 7 days, 0 otherwise)
            df['Target'] = (df['Close'].shift(-7) > df['Close'] * 1.05).astype(int)
            df = df.dropna()
            
            if len(df) < 30:
                return None
            
            # Feature selection
            feature_columns = [
                'RSI', 'MACD', 'BB_Position', 'Price_MA_Ratio', 
                'Price_Change_1d', 'Price_Change_7d', 'Volatility_7d',
                'Support_Distance_Pct', 'Resistance_Distance_Pct', 'Volume_Spike'
            ]
            
            X = df[feature_columns]
            y = df['Target']
            
            # Train model dengan lebih banyak trees
            model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                max_depth=15,
                min_samples_split=5
            )
            model.fit(X, y)
            
            # Feature importance
            feature_importance = dict(zip(feature_columns, model.feature_importances_))
            
            # Predict
            prediction = model.predict_proba([X.iloc[-1]])[0]
            
            return {
                'up_probability': prediction[1] * 100,
                'down_probability': prediction[0] * 100,
                'confidence': max(prediction) * 100,
                'feature_importance': feature_importance,
                'model_type': 'RandomForest_Enhanced'
            }
            
        except Exception as e:
            return None
    
    def generate_trading_recommendation(self, coin_data, historical_data, ml_prediction):
        """Generate comprehensive trading recommendation"""
        if not coin_data:
            return "UNAVAILABLE", "Data tidak cukup untuk analisis", 0, {}
        
        technical_signals = self.get_technical_signals(historical_data)
        fundamental_analysis = self.get_fundamental_analysis(coin_data)
        current_price = coin_data['current_price']
        
        # Comprehensive scoring system
        score = 50  # Neutral starting point
        
        # Technical Analysis Weight: 40%
        if technical_signals['trend'] == 'BULLISH':
            score += 15
        elif technical_signals['trend'] == 'BEARISH':
            score -= 15
        
        if technical_signals['momentum'] == 'BULLISH':
            score += 10
        elif technical_signals['momentum'] == 'OVERSOLD':
            score += 8
        elif technical_signals['momentum'] == 'OVERBOUGHT':
            score -= 10
        elif technical_signals['momentum'] == 'BEARISH':
            score -= 8
        
        if technical_signals['macd_signal'] == 'BULLISH':
            score += 5
        elif technical_signals['macd_signal'] == 'BEARISH':
            score -= 5
        
        # Fundamental Analysis Weight: 30%
        fundamental_score = fundamental_analysis['fundamental_score']
        score += (fundamental_score - 50) * 0.3
        
        # ML Prediction Weight: 30%
        if ml_prediction:
            if ml_prediction['up_probability'] > 70:
                score += 15
            elif ml_prediction['up_probability'] > 60:
                score += 10
            elif ml_prediction['down_probability'] > 70:
                score -= 15
            elif ml_prediction['down_probability'] > 60:
                score -= 10
        
        # Market Regime Adjustment
        if technical_signals['market_regime'] == 'TRENDING':
            score += 5
        elif technical_signals['market_regime'] == 'RANGING':
            score -= 3
        
        # Support/Resistance Adjustment
        if technical_signals['support_resistance'] == 'NEAR_SUPPORT':
            score += 8
        elif technical_signals['support_resistance'] == 'NEAR_RESISTANCE':
            score -= 8
        
        # Generate recommendation
        if score >= 75:
            action = "STRONG BUY"
            explanation = "📈 Kondisi teknikal sangat bullish, fundamental kuat, dan AI prediksi mendukung kenaikan harga"
            confidence = min(score, 95)
            
        elif score >= 65:
            action = "BUY"
            explanation = "📈 Signal teknikal positif dengan fundamental baik, peluang bagus untuk entry"
            confidence = score
            
        elif score >= 45:
            action = "HOLD"
            explanation = "⚖️ Market dalam kondisi netral, tunggu konfirmasi breakout atau fundamental improvement"
            confidence = 50
            
        elif score >= 35:
            action = "SELL"
            explanation = "📉 Teknikal menunjukkan weakness dengan fundamental lemah, pertimbangkan take profit"
            confidence = 70 - score
            
        else:
            action = "STRONG SELL" 
            explanation = "📉 Kondisi bearish kuat dengan fundamental buruk, hindari entry dan pertimbangkan exit"
            confidence = 95 - score
        
        # Price targets
        if action in ["STRONG BUY", "BUY"] and current_price > 0:
            if fundamental_analysis['risk_level'] == 'LOW':
                target_1 = current_price * 1.08  # 8% target
                target_2 = current_price * 1.15  # 15% target
                stop_loss = current_price * 0.92  # 8% stop loss
            else:
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
                'fundamental': fundamental_analysis,
                'ml_prediction': ml_prediction,
                'final_score': score
            }
        }
        
        return action, explanation, confidence, recommendation_details

def create_technical_chart(historical_data, coin_name):
    """Create comprehensive technical analysis chart"""
    if historical_data is None or len(historical_data) < 20:
        return None
        
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            f'{coin_name} - Price and Indicators', 
            'RSI and Momentum',
            'MACD'
        ),
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Price and Moving Averages
    fig.add_trace(
        go.Scatter(x=historical_data.index, y=historical_data['Close'], 
                  name='Price', line=dict(color='blue', width=2)),
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
    
    # Bollinger Bands
    if 'BB_Upper' in historical_data.columns:
        fig.add_trace(
            go.Scatter(x=historical_data.index, y=historical_data['BB_Upper'], 
                      name='BB Upper', line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=historical_data.index, y=historical_data['BB_Lower'], 
                      name='BB Lower', line=dict(color='green', dash='dash')),
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
    
    # MACD
    if 'MACD' in historical_data.columns:
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
    
    fig.update_layout(height=800, showlegend=True, template="plotly_white")
    
    return fig

def main():
    st.markdown('<h1 class="main-header">🤖 AI Crypto Prediction Pro - All Coins</h1>', unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = AdvancedCryptoAnalyzer()
    
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
        placeholder="Contoh: bitcoin, btc, dogecoin, doge, solana, sol...",
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
        st.info(f"🔍 Coin yang dianalisis: **{coin_name}** ({coin_symbol}) - `{coin_id}`")
        
        if st.button("🔄 Update Analysis", type="primary", key="analyze_button"):
            with st.spinner(f"Menganalisis {coin_name}..."):
                # Get all data
                coin_data = analyzer.get_coin_data_from_coingecko(coin_id)
                historical_data = analyzer.get_historical_data(coin_id)
                ml_prediction = analyzer.ml_price_prediction(historical_data, coin_data)
                
                if coin_data:
                    # Generate recommendation
                    action, explanation, confidence, details = analyzer.generate_trading_recommendation(
                        coin_data, historical_data, ml_prediction
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
                    
                    # Technical Analysis Section
                    st.subheader("📈 Technical Analysis")
                    tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
                    
                    technical_signals = details['score_breakdown']['technical']
                    fundamental = details['score_breakdown']['fundamental']
                    
                    with tech_col1:
                        st.metric("Trend", technical_signals['trend'])
                        st.metric("RSI", technical_signals.get('rsi', 'N/A'))
                        st.metric("Market Regime", technical_signals['market_regime'])
                    
                    with tech_col2:
                        st.metric("Momentum", technical_signals['momentum'])
                        st.metric("MACD Signal", technical_signals['macd_signal'])
                        st.metric("ADX", technical_signals.get('adx', 'N/A'))
                    
                    with tech_col3:
                        st.metric("Market Strength", fundamental['market_strength'])
                        st.metric("Risk Level", fundamental['risk_level'])
                        st.metric("Fundamental Score", f"{fundamental['fundamental_score']}/100")
                    
                    with tech_col4:
                        if ml_prediction:
                            st.metric("AI Prediction", f"{ml_prediction['up_probability']:.1f}% Up")
                            st.metric("AI Confidence", f"{ml_prediction['confidence']:.1f}%")
                            st.metric("Support/Resistance", technical_signals['support_resistance'])
                        else:
                            st.metric("AI Prediction", "N/A")
                            st.metric("AI Confidence", "N/A")
                            st.metric("Support/Resistance", technical_signals['support_resistance'])
                    
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
                        - Target 1: {analyzer.format_currency(details['targets']['target_1'])} (+{((details['targets']['target_1']/coin_data['current_price'])-1)*100:.1f}%)
                        - Target 2: {analyzer.format_currency(details['targets']['target_2'])} (+{((details['targets']['target_2']/coin_data['current_price'])-1)*100:.1f}%)
                        - Stop Loss: {analyzer.format_currency(details['targets']['stop_loss'])} (-{((1-details['targets']['stop_loss']/coin_data['current_price']))*100:.1f}%)
                        """)
                    
                    # Detailed Analysis
                    with st.expander("🔍 Detailed Analysis Breakdown"):
                        tab1, tab2, tab3, tab4 = st.tabs(["Technical", "Fundamental", "ML Analysis", "Risk Management"])
                        
                        with tab1:
                            st.subheader("Technical Signals")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Trend & Momentum:**")
                                for signal, value in list(technical_signals.items())[:5]:
                                    st.write(f"- {signal.replace('_', ' ').title()}: {value}")
                            with col2:
                                st.write("**Volatility & Position:**")
                                for signal, value in list(technical_signals.items())[5:]:
                                    st.write(f"- {signal.replace('_', ' ').title()}: {value}")
                        
                        with tab2:
                            st.subheader("Fundamental Analysis")
                            st.write(f"- **Market Strength:** {fundamental['market_strength']}")
                            st.write(f"- **Risk Level:** {fundamental['risk_level']}")
                            st.write(f"- **Supply Analysis:** {fundamental['supply_analysis']}")
                            st.write(f"- **Overall Score:** {fundamental['fundamental_score']}/100")
                            
                            st.write("**Key Factors:**")
                            for factor, value in fundamental.get('factors', {}).items():
                                st.write(f"- {factor.replace('_', ' ').title()}: {value}")
                        
                        with tab3:
                            st.subheader("Machine Learning Analysis")
                            if ml_prediction:
                                st.write(f"- **Up Probability:** {ml_prediction['up_probability']:.1f}%")
                                st.write(f"- **Down Probability:** {ml_prediction['down_probability']:.1f}%")
                                st.write(f"- **Confidence:** {ml_prediction['confidence']:.1f}%")
                                st.write(f"- **Model Type:** {ml_prediction.get('model_type', 'RandomForest')}")
                                
                                if 'feature_importance' in ml_prediction:
                                    st.write("**Feature Importance:**")
                                    for feature, importance in sorted(ml_prediction['feature_importance'].items(), 
                                                                   key=lambda x: x[1], reverse=True)[:5]:
                                        st.write(f"- {feature}: {importance:.3f}")
                            else:
                                st.write("ML analysis tidak tersedia")
                        
                        with tab4:
                            st.subheader("Risk Management")
                            st.write("**Position Sizing Recommendations:**")
                            if fundamental['risk_level'] == 'LOW':
                                st.write("- **Allocation:** 3-5% of portfolio")
                                st.write("- **Leverage:** Avoid or use minimal (1-2x)")
                            elif fundamental['risk_level'] == 'MEDIUM':
                                st.write("- **Allocation:** 1-3% of portfolio") 
                                st.write("- **Leverage:** Avoid")
                            else:
                                st.write("- **Allocation:** 0.5-1% of portfolio")
                                st.write("- **Leverage:** Strictly avoid")
                            
                            st.write("**Risk Notes:**")
                            st.write("- Always use stop-loss orders")
                            st.write("- Diversify across different crypto sectors")
                            st.write("- Monitor market conditions regularly")
                
                else:
                    st.error("❌ Gagal mengambil data coin. Coba lagi nanti.")
    
    # Clear selection button
    if st.session_state.get('selected_coin'):
        if st.button("🗑️ Clear Selection", type="secondary"):
            st.session_state.selected_coin = None
            st.session_state.search_query = ""
            st.experimental_rerun()
    
    # Instructions
    with st.expander("ℹ️ Cara Menggunakan & Faktor Analisis"):
        st.markdown("""
        ### 🔍 Cara Menggunakan:
        1. **Ketik nama/simbol coin** di search box
        2. **Klik coin** yang ingin dianalisis (akan auto-fill)
        3. **Klik 'Update Analysis'** untuk prediksi
        
        ### 📊 Faktor Analisis yang Digunakan:
        
        **🎯 TECHNICAL ANALYSIS (40% Weight):**
        - **Trend Analysis**: MA 7/20/50, EMA 12/26, ADX
        - **Momentum Indicators**: RSI, Stochastic, Williams %R, CCI  
        - **Volatility**: Bollinger Bands, ATR, BB Width
        - **Support/Resistance**: Price levels, distance calculations
        - **Market Regime**: Trending vs Ranging markets
        
        **🏦 FUNDAMENTAL ANALYSIS (30% Weight):**
        - **Market Position**: Market cap rank, liquidity
        - **Supply Metrics**: Circulating vs max supply, inflation rate
        - **Price History**: Distance from ATH/ATL, recovery potential
        - **Volume Analysis**: Volume/Market cap ratio
        
        **🤖 MACHINE LEARNING (30% Weight):**
        - **Random Forest Classifier** dengan 100+ trees
        - **Feature Engineering**: 10+ technical indicators sebagai input
        - **Price Prediction**: Probability of 5% gain dalam 7 hari
        - **Feature Importance**: Analisis faktor paling berpengaruh
        
        **⚖️ RISK MANAGEMENT:**
        - **Dynamic Position Sizing** berdasarkan risk level
        - **Auto-calculated** stop loss & take profit levels
        - **Portfolio allocation** recommendations
        
        ### ⚠️ Disclaimer:
        Analisis ini untuk edukasi saja. Selalu lakukan research mandiri dan gunakan proper risk management!
        """)

if __name__ == "__main__":
    main()
