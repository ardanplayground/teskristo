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

# Ambil list semua coin
@st.cache_data
def get_all_coins():
    url = "https://api.coingecko.com/api/v3/coins/list"
    res = requests.get(url)
    if res.status_code == 200:
        return res.json()
    return []

# Ambil data coin detail dari CoinGecko
def get_coin_data(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    res = requests.get(url)
    if res.status_code == 200:
        data = res.json()
        market_data = data.get("market_data", {})
        return {
            "Harga Sekarang (Rp)": market_data.get("current_price", {}).get("idr", "N/A"),
            "High 24 Jam (Rp)": market_data.get("high_24h", {}).get("idr", "N/A"),
            "Low 24 Jam (Rp)": market_data.get("low_24h", {}).get("idr", "N/A"),
            "Perubahan 7 Hari (%)": market_data.get("price_change_percentage_7d", "N/A"),
            "Volume Transaksi (Rp)": market_data.get("total_volume", {}).get("idr", "N/A")
        }
    return None

# Dummy Sentimen
def get_sentiment_data(coin_symbol):
    pos = random.randint(0, 100)
    neg = 100 - pos
    return pos, neg

# Rekomendasi beli/jual
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

# --- Streamlit UI ---
st.set_page_config(page_title="Prediksi Crypto", layout="centered")
st.title("🔮 Prediksi Crypto: Beli atau Tidak?")

# Session State Default Coin
if "selected_coin" not in st.session_state:
    st.session_state.selected_coin = "bitcoin"

all_coins = get_all_coins()
coin_ids = {coin["id"]: coin for coin in all_coins}

# Input Prediksi dari Session State
coin = st.text_input("Masukkan simbol coin (contoh: bitcoin, binancecoin, solana):", value=st.session_state.selected_coin)

if coin.lower() not in coin_ids:
    st.warning("⚠️ Coin tidak ditemukan di database CoinGecko.")
else:
    if st.button("Prediksi Sekarang"):
        with st.spinner("Mengambil data..."):
            coin_data = get_coin_data(coin.lower())
            pos, neg = get_sentiment_data(coin.lower())

            if coin_data:
                rekom, sell = get_recommendation(pos, coin_data["Harga Sekarang (Rp)"])

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
                st.error("❌ Gagal mengambil data dari CoinGecko.")

# --- HR dan Search ---
st.markdown("---")
with st.expander("🔍 Cari Coin dari Database CoinGecko"):
    keyword = st.text_input("Cari coin berdasarkan simbol/nama:")
    if st.button("Cari Coin"):
        filtered = [
            c for c in all_coins
            if keyword.lower() in c["symbol"].lower() or keyword.lower() in c["name"].lower()
        ]
        if filtered:
            st.write("Klik untuk memilih:")
            cols = st.columns(3)
            for i, coin_info in enumerate(filtered[:30]):
                with cols[i % 3]:
                    if st.button(f"{coin_info['symbol'].upper()} ({coin_info['name']})", key=f"select_{coin_info['id']}"):
                        st.session_state.selected_coin = coin_info["id"]
                        st.experimental_rerun()
        else:
            st.warning("Tidak ditemukan coin sesuai kata kunci.")
