import streamlit as st
import os
import base64
from pathlib import Path

st.set_page_config(
    page_title="Petunjuk Pengguna - Analisis Sentimen",
    page_icon="📖",
    layout="wide",
)

def load_css(css_file):
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.warning(f"CSS file not found: {css_file}")


def about():

    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div style="text-align: center;">
            <h1 style="font-size: 3rem; color: #333; margin-bottom: 1rem; font-weight: 700;">
                📊 CryptoSentiment Analyzer
            </h1>
            <p style="font-size: 1.3rem; color: #666; margin-bottom: 2rem;">
                Platform Analisis Sentimen untuk Ulasan Aplikasi Exchange Cryptocurrency
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tentang Kami Section
    st.markdown("""
    <div class="content-section">
        <h2 class="section-title">Tentang CryptoSentiment Analyzer</h2>
        <div class="section-text">
            <p>
                CryptoSentiment Analyzer adalah platform analisis sentimen canggih yang dirancang  
                untuk menganalisis ulasan dan review aplikasi exchange cryptocurrency dari platform 
                Google Play Store. Aplikasi ini membantu investor, trader, dan pengembang 
                untuk memahami persepsi publik terhadap platform trading crypto favorit mereka.
            </p>
            <br>
            <p>
                Dengan memanfaatkan teknologi Natural Language Processing (NLP) dan Machine Learning, 
                yang memberikan pandangan mendalam tentang sentimen pengguna, tren kepuasan, dan aspek-aspek 
                yang paling diperhatikan oleh komunitas crypto trading.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Cara Kerja Section
    st.markdown("""
    <div class="content-section">
        <h2 class="section-title">Bagaimana Cara Kerjanya?</h2>
        <div class="section-text">
            <p>Platform kami menggunakan pendekatan sistematis untuk memberikan analisis sentimen yang akurat dan actionable:</p>
        </div>
        <ol style="margin-left: 2rem; text-align: left;">
            <li>
                <b>Data Scraping:</b>
                Menggunakan teknik web scraping untuk mengumpulkan review dari Google Play Store menggunakan google play scraper.
            </li>
            <li>
                <b>Data Preprocessing & Cleaning:</b>
                Pembersihan data meliputi penghapusan noise, normalisasi teks, penanganan emoji, stopwords removal,  
                tokenisasi, stemming dan preprocessing untuk bahasa Indonesia.
            </li>
            <li>
                <b>Sentiment Analysis:</b>
                Implementasi model supervised learning dan naive bayes dan penerapan pelabelan data dengan model lexicon based untuk sentimen positif, netral dan negatif.
            </li>
            <li>
                <b>Visualisasi:</b>
                Presentasi hasil dalam dashboard interaktif dengan berbagai tabel, chart, grafik, 
                dan visualisasi yang dapat dipahami.
            </li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # Load CSS
    css_file = os.path.join("assets", "style.css")


if __name__ == "__main__":
    

    about()