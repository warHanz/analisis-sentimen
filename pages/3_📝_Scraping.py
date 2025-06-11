import streamlit as st
from google_play_scraper import reviews, Sort
import pandas as pd
import re

# Fungsi untuk ekstrak app_id dari URL
def extract_app_id(url):
    match = re.search(r'id=([a-zA-Z0-9._]+)', url)
    if match:
        return match.group(1)
    else:
        return url  # fallback jika user tetap memasukkan app_id langsung

# Judul aplikasi
st.title("Scraping Ulasan Aplikasi Pada Google Play Store")
st.write("Aplikasi ini menampilkan semua ulasan terbaru dari aplikasi di Google Play Store tanpa batasan jumlah.")

# Input URL aplikasi atau app_id
input_url = st.text_input(
    "Masukkan URL aplikasi Google Play Store atau App ID (contoh: https://play.google.com/store/apps/details?id=id.co.bitcoin):",
    value=""
)

# Ekstrak app_id
app_id = extract_app_id(input_url)

# Input jumlah ulasan yang ingin diambil (tanpa batasan maksimal)
count = st.number_input("Masukkan jumlah ulasan yang ingin diambil (0 untuk semua ulasan):", 
                       min_value=0, 
                       value=100)

if st.button("Ambil Ulasan"):
    try:
        # Inisialisasi variabel
        all_reviews = []
        continuation_token = None
        batch_size = 100  # Jumlah ulasan per request

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        with st.spinner("Sedang mengambil ulasan..."):
            while True:
                # Tentukan jumlah ulasan yang akan diambil di batch ini
                if count > 0:
                    remaining = count - len(all_reviews)
                    if remaining <= 0:
                        break
                    current_batch = min(batch_size, remaining)
                else:
                    current_batch = batch_size

                # Ambil ulasan
                result, continuation_token = reviews(
                    app_id,
                    lang='id',
                    country='id',
                    sort=Sort.NEWEST,
                    count=current_batch,
                    continuation_token=continuation_token
                )

                # Tambahkan ulasan ke list
                all_reviews.extend(result)

                # Update progress
                if count > 0:
                    progress = min(1.0, len(all_reviews) / count)
                    progress_bar.progress(progress)
                    status_text.text(f"Mengambil ulasan: {len(all_reviews)} dari {count}")
                else:
                    progress_bar.progress(0)
                    status_text.text(f"Mengambil ulasan: {len(all_reviews)} (tanpa batas)")

                # Jika tidak ada token lanjutan atau tidak ada hasil lagi, hentikan
                if not continuation_token or not result:
                    break

        # Tampilkan jumlah ulasan yang berhasil diambil
        st.success(f"Berhasil mengambil {len(all_reviews)} ulasan!")

        # Format data
        formatted_reviews = []
        for review in all_reviews:
            formatted_reviews.append({
                'Reviews ID': review['reviewId'],
                'Username': review['userName'],
                'Rating': review['score'],
                'Reviews Text': review['content'],
                'Date': review['at'].strftime('%Y-%m-%d %H:%M:%S')
            })

        # Konversi ke DataFrame
        reviews_df = pd.DataFrame(formatted_reviews)

        # Tampilkan ulasan dalam tabel
        st.write("### Data Ulasan:")
        st.dataframe(reviews_df)

        # Tombol untuk mengunduh sebagai CSV
        csv = reviews_df.to_csv(index=False).encode('utf-8')
        file_name = f"ulasan_{app_id}.csv"
        st.download_button(
            label="Unduh Data Ulasan (CSV)",
            data=csv,
            file_name=file_name,
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

# Catatan untuk pengguna
st.write("Pastikan koneksi internet stabil untuk mengambil data.")
st.write("Masukkan 0 pada jumlah ulasan untuk mengambil semua ulasan yang tersedia.")