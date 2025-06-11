import streamlit as st
import pandas as pd
import os
from utils.utils import *

# Konfigurasi halaman
st.set_page_config(layout="wide", page_title="Preprocessing Data")

def main():
    st.title("Preprocessing Data")
    st.markdown("Unggah file CSV/Excel untuk memproses data ulasan.")

    # File uploader
    uploaded_file = st.file_uploader("Unggah file dataset (CSV/Excel)", type=["csv", "xlsx"], key="data_uploader")

    if uploaded_file is None:
        st.info("Silakan unggah file dataset untuk memulai preprocessing.")
        return

    with st.spinner("Memproses data..."):
        # Memuat leksikon sentimen
        positive_words, negative_words = load_sentiment_lexicon()

        # Load data
        data = load_data(uploaded_file)
        if data is None:
            return

        # Initialize counts dictionary
        counts_dict = {
            "Raw Data": len(data),
            "After Deduplication": 0,
            "After Cleaning": 0,
            "After Normalization": 0,
            "After Tokenization": 0,
            "After Stopword Removal": 0,
            "After Stemming": 0
        }

        # Step 1: Preprocess data (deduplication)
        processed_data, dedup_count = preprocess_data(data)
        counts_dict["After Deduplication"] = dedup_count
        if processed_data is None:
            return

        # Step 2: Cleaning
        processed_data['review_text'] = processed_data['Reviews Text']
        processed_data['cleaning'] = processed_data['review_text'].apply(clean_text)
        counts_dict["After Cleaning"] = len(processed_data[processed_data['cleaning'] != ''])

        # Step 3: Normalization
        norm_dict = None
        kamus_path = os.path.join(os.getcwd(), "assets", "kamuskatabaku.xlsx")

        st.info(f"Mencari kamus normalization di: {kamus_path}")
        
        if os.path.exists(kamus_path):
            try:
                kamus = pd.read_excel(kamus_path)
                norm_dict = dict(zip(kamus['tidak_baku'], kamus['kata_baku']))
                processed_data['normalize'] = processed_data['cleaning'].apply(lambda x: normalize_text(x, norm_dict))
            except Exception as e:
                st.warning(f"Gagal membaca kamuskatabaku.xlsx: {e}")
                processed_data['normalize'] = processed_data['cleaning']
        else:
            st.warning("File 'kamuskatabaku.xlsx' tidak ditemukan. Normalisasi dilewati.")
            processed_data['normalize'] = processed_data['cleaning']
        counts_dict["After Normalization"] = len(processed_data[processed_data['normalize'] != ''])

        # Step 4: Tokenization
        processed_data['tokenize'] = processed_data['normalize'].apply(tokenize_text)
        counts_dict["After Tokenization"] = len(processed_data[processed_data['tokenize'].apply(len) > 0])

        # Step 5: Stopword Removal
        processed_data['remove_stopword'] = processed_data['tokenize'].apply(remove_stopwords)
        counts_dict["After Stopword Removal"] = len(processed_data[processed_data['remove_stopword'].apply(len) > 0])

        # Step 6: Stemming
        processed_data['stemmed'] = processed_data['remove_stopword'].apply(stem_tokens)
        counts_dict["After Stemming"] = len(processed_data[processed_data['stemmed'].apply(len) > 0])

        # Filter out empty stemmed data
        processed_data = processed_data[processed_data['stemmed'].apply(len) > 0]

        # Apply sentiment labeling
        processed_data[['Sentiment Score', 'Sentiment']] = processed_data['stemmed'].apply(
            lambda x: pd.Series(label_sentiment(x, positive_words, negative_words)))
        processed_data['Text_for_Model'] = processed_data['stemmed'].apply(preprocess_for_model)

        # Display Overview Table
        st.subheader("📋 Tabel Overview Preprocessing")
        st.write("Tabel menampilkan teks asli dan hasil setiap tahap preprocessing.")
        overview_table = processed_data[['review_text', 'cleaning', 'normalize', 'tokenize', 'remove_stopword', 'stemmed']]
        st.dataframe(overview_table)

        # Display Data Count Graph
        st.subheader("📊 Jumlah Data Setelah Setiap Tahap Pemrosesan")
        st.plotly_chart(plot_processing_counts(counts_dict), use_container_width=True)

        # Save preprocessed data
        if save_preprocessed_data(processed_data):
            st.success("Data telah diproses dan disimpan. Silakan buka halaman 'Analisis' untuk melihat hasil.")
        else:
            st.error("Gagal menyimpan data. Tidak dapat melanjutkan ke analisis.")

if __name__ == "__main__":
    main()