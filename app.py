# crypto_chat.py
import streamlit as st
import requests
from textblob import TextBlob
from bs4 import BeautifulSoup
from datetime import datetime
import re

st.set_page_config(page_title="Crypto Chat", page_icon="💰")

st.title("💬 Crypto Chat AI")
st.markdown("Masukkan coin (misal: `btc`, `bnb`, `solana`) dan aku bantu prediksi apakah layak dibeli atau tidak!")

def get_coin_data(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    res = requests.get(url)
    if res.status_code == 200:
        data = res.json()
        price = data["market_data"]["current_price"]["usd"]
        high_24h = data["market_data"]["high_24h"]["usd"]
        low_24h = data["market_data"]["low_24h"]["usd"]
        change_7d = data["market_data"]["price_change_percentage_7d"]
        volume = data["market_data"]["total_volume"]["usd"]
        return {
            "price": price,
            "high": high_24h,
            "low": low_24h,
            "change_7d": change_7d,
            "volume": volume
        }
    return None

def search_news(query):
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://news.google.com/search?q={query}%20cryptocurrency&hl=en"
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    articles = soup.select("article h3")
    headlines = [a.get_text() for a in articles][:5]
    return headlines

def analyze_sentiment(headlines):
    pos, neg = 0, 0
    for h in headlines:
        score = TextBlob(h).sentiment.polarity
        if score > 0:
            pos += 1
        elif score < 0:
            neg += 1
    total = len(headlines)
    pos_percent = round((pos / total) * 100, 2) if total else 0
    neg_percent = round((neg / total) * 100, 2) if total else 0
    return pos_percent, neg_percent

def evaluate(data, pos_percent):
    if data["change_7d"] > 3 and pos_percent > 60:
        return "✅ Layak dibeli"
    elif pos_percent > 50:
        return "⚠️ Berpotensi naik, tapi hati-hati"
    else:
        return "❌ Tidak disarankan beli sekarang"

def predict_sell_point(data):
    # Strategi sederhana: jual di harga tertinggi 7 hari terakhir +5%
    est_sell_price = round(data["high"] * 1.05, 2)
    return f"💰 Disarankan jual saat harga menyentuh sekitar ${est_sell_price}"

# Session & Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan riwayat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input user
user_input = st.chat_input("Masukkan nama coin (misal: btc, solana, bnb)")
if user_input:
    coin = user_input.lower().strip()
    with st.chat_message("user"):
        st.markdown(coin)
    st.session_state.messages.append({"role": "user", "content": coin})

    with st.chat_message("assistant"):
        coin_data = get_coin_data(coin)
        if coin_data:
            news = search_news(coin)
            pos, neg = analyze_sentiment(news)
            rekom = evaluate(coin_data, pos)
            sell = predict_sell_point(coin_data)

            response = f"""
**📊 Harga Sekarang**: ${coin_data['price']}
**📈 High 24h**: ${coin_data['high']} | **Low 24h**: ${coin_data['low']}
**📉 Perubahan 7 Hari**: {coin_data['change_7d']}%
**💵 Volume Transaksi**: ${int(coin_data['volume']):,}

**📰 Sentimen Berita:**
- Positif: {pos}%
- Negatif: {neg}%

{rekom}
{sell}
"""
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.error("❌ Coin tidak ditemukan. Coba pakai nama coin seperti `bitcoin`, `solana`, `bnb`.")
