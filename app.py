import streamlit as st
import requests
import random
import pandas as pd

# Format ke rupiah tanpa koma desimal
def format_rupiah(val):
    try:
        val = float(val)
        return "Rp {:,.0f}".format(val).replace(",", ".")
    except:
        return "N/A"

# Ambil data coin dari CoinGecko
def get_coin_data(coin_symbol):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_symbol}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        market_data = data.get("market_data", {})
        return {
            "Harga Sekarang (Rp)": market_data.get("current_price", {}).get("idr", "N/A"),
            "High 24 Jam (Rp)": market_data.get("high_24h", {}).get("idr", "N/A"),
            "Low 24 Jam (Rp)": market_data.get("low_24h", {}).get("idr", "N/A"),
            "Perubahan 7 Hari (%)": market_data.get("price_change_percentage_7d", "N/A"),
            "Volume Transaksi (Rp)": market_data.get("total_volume", {}).get("idr", "N/A")
        }
    return None

# Sentimen dummy
def get_sentiment_data(coin_symbol):
    pos = random.randint(0, 100)
    neg = 100 - pos
    return pos, neg

# Rekomendasi berdasarkan sentimen dan harga
def get_recommendation(pos, harga_rp):
    if isinstance(harga_rp, (int, float)):
        if pos > 60:
            rekom = f"✅ Disarankan beli sekarang dengan harga {format_rupiah(harga_rp)}"
        else:
            rekom = "❌ Tidak disarankan beli saat ini"
        sell_target = harga_rp * 1.08
        sell = f"💰 Jual saat harga menyentuh sekitar {format_rupiah(sell_target)}"
    else:
        rekom = "❌ Tidak disarankan beli saat ini"
        sell = "N/A"
    return rekom, sell

# -------- Streamlit App -------- #
st.set_page_config(page_title="Prediksi Crypto", layout="centered")
st.title("🔮 Prediksi Crypto: Beli atau Tidak?")

# Tombol pop-up list koin
with st.expander("📋 Lihat Daftar Coin Populer"):
    daftar_koin = [
        "bitcoin", "ethereum", "binancecoin", "solana", "ripple", "cardano", "dogecoin",
        "polkadot", "tron", "avalanche", "litecoin", "stellar", "chainlink", "uniswap"
    ]
    search = st.text_input("🔎 Cari coin:")
    hasil_filter = [coin for coin in daftar_koin if search.lower() in coin.lower()]
    st.write("Klik salah satu coin di bawah untuk mengisi input:")
    for coin in hasil_filter:
        if st.button(coin):
            st.session_state["selected_coin"] = coin

# Text input default
default_coin = st.session_state.get("selected_coin", "bitcoin")
coin = st.text_input("Masukkan simbol coin (contoh: bitcoin, binancecoin, solana):", value=default_coin)

# Tombol prediksi
if st.button("Prediksi Sekarang"):
    with st.spinner("Mengambil data..."):
        coin_data = get_coin_data(coin.lower())
        pos, neg = get_sentiment_data(coin.lower())

        if coin_data:
            rekom, sell = get_recommendation(pos, coin_data["Harga Sekarang (Rp)"])

            # Format tampilan tabel
            coin_data_display = {}
            for k, v in coin_data.items():
                if "Rp" in k:
                    coin_data_display[k] = format_rupiah(v)
                elif "%" in k:
                    coin_data_display[k] = f"{v:.2f}%" if isinstance(v, (int, float)) else v
                else:
                    coin_data_display[k] = v

            data_df = pd.DataFrame.from_dict(coin_data_display, orient='index', columns=["Nilai"])
            st.subheader("📊 Data Pasar")
            st.table(data_df)

            sentiment_df = pd.DataFrame({
                "Sentimen": ["Positif", "Negatif"],
                "Persentase": [f"{pos}%", f"{neg}%"]
            })
            st.subheader("📰 Sentimen Berita")
            st.table(sentiment_df)

            st.subheader("📌 Rekomendasi")
            st.success(rekom)
            if sell != "N/A":
                st.info(sell)
        else:
            st.error("Gagal mengambil data coin. Pastikan simbol coin valid.")
