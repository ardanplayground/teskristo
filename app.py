import streamlit as st
import requests
from datetime import datetime
import locale

# Format Rupiah
locale.setlocale(locale.LC_ALL, 'id_ID.UTF-8')


def format_rupiah(amount):
    rupiah = f"Rp {amount:,.0f}".replace(",", ".")
    return rupiah


# Get coin list from CoinGecko
@st.cache_data(show_spinner=False)
def get_all_coins():
    url = "https://api.coingecko.com/api/v3/coins/list"
    response = requests.get(url)
    return response.json()


# Get current price of coin
def get_coin_price(coin_id):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=idr"
    response = requests.get(url)
    data = response.json()
    return data.get(coin_id, {}).get("idr", None)


# Simulasi prediksi & rekomendasi
def predict_recommendation(price):
    if price is None:
        return "Harga tidak tersedia", "Tidak disarankan membeli"

    if price < 20000:
        return "Negatif", f"Tidak disarankan membeli sekarang ({format_rupiah(price)})"
    elif price < 1000000:
        return "Netral", f"Boleh dipertimbangkan beli sekarang ({format_rupiah(price)})"
    else:
        return "Positif", f"Disarankan beli sekarang ({format_rupiah(price)})"


st.set_page_config(page_title="Prediksi Coin Crypto", layout="wide")
st.title("📈 Prediksi Coin Crypto dan Rekomendasi Beli/Jual")

# ========================
# Input dan Prediksi
# ========================
st.markdown("### 🔍 Pilih Coin")

coin_input = st.text_input("Masukkan nama coin atau ID dari CoinGecko (contoh: bitcoin, solana, bnb)", "bitcoin")

if coin_input:
    with st.spinner("Mengambil data..."):
        price = get_coin_price(coin_input.lower())
        sentiment, recommendation = predict_recommendation(price)

    st.markdown("### 🧠 Hasil Prediksi")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Sentimen", sentiment)
    with col2:
        st.metric("Rekomendasi", recommendation)

    if price:
        st.markdown(f"Harga saat ini: **{format_rupiah(price)}**")
    else:
        st.error("Gagal mengambil harga. Pastikan ID coin valid.")

# ========================
# Daftar Semua Koin
# ========================
st.markdown("---")
st.subheader("📜 Daftar Semua Koin di CoinGecko")

# Ambil semua coin
with st.spinner("Memuat semua coin..."):
    all_coins = get_all_coins()

search_query = st.text_input("Cari koin...").lower()

# Filter
if search_query:
    filtered_coins = [coin for coin in all_coins if search_query in coin["name"].lower() or search_query in coin["symbol"].lower()]
else:
    filtered_coins = all_coins

# Pagination
items_per_page = 20
total_items = len(filtered_coins)
total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)

if total_items == 0:
    st.info("Tidak ada koin yang cocok.")
else:
    page = st.number_input("Halaman", min_value=1, max_value=total_pages, value=1, step=1)
    start = (page - 1) * items_per_page
    end = start + items_per_page

    st.table([
        {
            "Nama": coin["name"],
            "Simbol": coin["symbol"].upper(),
            "ID": coin["id"]
        }
        for coin in filtered_coins[start:end]
    ])
