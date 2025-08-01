import streamlit as st
import requests
import random

def get_coin_data(coin_symbol):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_symbol}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        market_data = data.get("market_data", {})
        return {
            "price": market_data.get("current_price", {}).get("usd", "N/A"),
            "high": market_data.get("high_24h", {}).get("usd", "N/A"),
            "low": market_data.get("low_24h", {}).get("usd", "N/A"),
            "change_7d": market_data.get("price_change_percentage_7d", "N/A"),
            "volume": market_data.get("total_volume", {}).get("usd", "N/A")
        }
    return None

def get_sentiment_data(coin_symbol):
    # Dummy sentiment karena API berita tidak tersedia
    pos = random.randint(0, 100)
    neg = 100 - pos
    return pos, neg

def get_recommendation(pos, coin_data):
    rekom = ""
    sell = ""
    if pos > 60:
        rekom = "✅ **Disarankan beli sekarang!**"
    else:
        rekom = "❌ **Tidak disarankan beli sekarang**"

    if coin_data["price"] != "N/A":
        try:
            price = float(coin_data["price"])
            sell_price = price * 1.08
            sell = f"💰 **Disarankan jual saat harga menyentuh sekitar ${sell_price:.2f}**"
        except:
            sell = ""
    return rekom, sell

st.set_page_config(page_title="Prediksi Crypto", layout="centered")
st.title("🔮 Prediksi Crypto: Beli atau Tidak?")

coin = st.text_input("Masukkan simbol coin (contoh: bitcoin, binancecoin, solana):", value="bitcoin")

if st.button("Prediksi Sekarang"):
    with st.spinner("Mengambil data..."):
        coin_data = get_coin_data(coin.lower())
        pos, neg = get_sentiment_data(coin.lower())

        if coin_data:
            rekom, sell = get_recommendation(pos, coin_data)

            response = (
                f"📊 **Harga Sekarang**: ${coin_data['price']}\n"
                f"📈 **High 24h**: ${coin_data['high']} | **Low 24h**: ${coin_data['low']}\n"
                f"📉 **Perubahan 7 Hari**: {coin_data['change_7d']}%\n"
                f"💵 **Volume Transaksi**: ${int(coin_data['volume']):,}\n\n"
                f"📰 **Sentimen Berita:**\n"
                f"- Positif: {pos}%\n"
                f"- Negatif: {neg}%\n\n"
                f"{rekom}\n"
                f"{sell}"
            )

            st.markdown(response)
        else:
            st.error("Gagal mengambil data coin. Pastikan simbol coin valid.")
