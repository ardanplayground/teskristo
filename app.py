import streamlit as st
import requests
import random
import pandas as pd
import math

# Format ke rupiah tanpa koma desimal
def format_rupiah(val):
    try:
        val = float(val)
        return "Rp {:,.0f}".format(val).replace(",", ".")
    except:
        return "N/A"

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

def get_sentiment_data(coin_symbol):
    pos = random.randint(0, 100)
    neg = 100 - pos
    return pos, neg

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

# Ambil semua daftar koin
@st.cache_data(ttl=86400)
def get_all_coins():
    url = "https://api.coingecko.com/api/v3/coins/list"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return []

# ================== STREAMLIT APP ================== #

st.set_page_config(page_title="Prediksi Crypto", layout="centered")
st.title("🔮 Prediksi Crypto: Beli atau Tidak?")

coin = st.text_input("Masukkan simbol coin (contoh: bitcoin, binancecoin, solana):", value="bitcoin")

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
            st.error("Gagal mengambil data coin. Pastikan simbol coin valid.")

# ================== DAFTAR SEMUA COIN ================== #

st.markdown("---")
st.subheader("📋 Daftar Semua Coin di CoinGecko")

all_coins = get_all_coins()

# Pencarian coin
query = st.text_input("Cari coin di daftar semua coin:")
filtered_coins = [
    c for c in all_coins
    if query.lower() in c["id"].lower() or query.lower() in c["symbol"].lower()
] if query else all_coins

# Pagination
coins_per_page = 10
total_items = len(filtered_coins)
total_pages = max(1, math.ceil(total_items / coins_per_page))

if total_items > 0:
    page = st.number_input("Halaman", min_value=1, max_value=total_pages, value=1, step=1)
    start_idx = (page - 1) * coins_per_page
    end_idx = start_idx + coins_per_page
    paginated_coins = filtered_coins[start_idx:end_idx]

    # Tampilkan daftar coin
    for coin in paginated_coins:
        st.markdown(f"- **{coin['id']}** ({coin['symbol']})")

    st.caption(f"Menampilkan {len(filtered_coins)} coin - Halaman {page} dari {total_pages}")
else:
    st.warning("Tidak ada coin yang cocok dengan pencarian kamu.")
