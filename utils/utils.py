import os
import re
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from wordcloud import WordCloud
from collections import Counter
import streamlit as st



factory = StemmerFactory()
stemmer = factory.create_stemmer()

STOPWORDS_TO_REMOVE = {
    'tidak', 'bukan', 'belum', 'jangan', 'saya', 'kamu', 'dia', 'mereka', 'kita', 'kami',
    'itu', 'ini', 'ada', 'perlu', 'sama', 'saat', 'seperti', 'dari', 'ke', 'untuk',
    'pada', 'dan', 'atau', 'yang', 'dengan', 'di', 'ya', 'asu', 'sih', 'lah', 'gak',
    'nggak', 'kok', 'dong', 'nih', 'aja', 'bisa', 'cuma', 'woi', 'indodax' 
}

CUSTOM_STOPWORDS = {
    'adalah', 'akan', 'antara', 'apa', 'apabila', 'atas', 'bagaimana', 'bagi', 'bahwa',
    'berada', 'berapa', 'berikut', 'bersama', 'boleh', 'banyak', 'baru', 'bila', 'bilang',
    'cara', 'cukup', 'dapat', 'demi', 'demikian', 'depan', 'dulu', 'harus', 'hanya',
    'hingga', 'ia', 'ialah', 'jika', 'justru', 'kapan', 'karena', 'karenanya', 'kemudian',
    'kini', 'lagi', 'lalu', 'lebih', 'maupun', 'melainkan', 'menjadi', 'menuju', 'meski',
    'mungkin', 'oleh', 'paling', 'para', 'per', 'pernah', 'saja', 'sambil', 'sampai',
    'sangat', 'sebelum', 'sekarang', 'secara', 'sedang', 'selain', 'selama', 'seluruh',
    'semua', 'sesudah', 'setelah', 'setiap', 'sudah', 'supaya', 'tapi', 'tentang',
    'terhadap', 'termasuk', 'ternyata', 'tetap', 'tetapi', 'tiap', 'tuju', 'turut', 'umum',
    'yaitu' 
}

STOPWORDS_ALL = set(stopwords.words('indonesian')).union(STOPWORDS_TO_REMOVE)

try:
    # This will rely on the NLTK data downloaded via Streamlit Cloud's advanced settings
    STOPWORDS_ALL = set(stopwords.words('indonesian')).union(STOPWORDS_TO_REMOVE)
except LookupError:
    st.warning("NLTK 'indonesian' stopwords data not found. Using only custom and removed stopwords.")
    STOPWORDS_ALL = STOPWORDS_TO_REMOVE # Fallback if NLTK stopwords fail

def load_sentiment_lexicon():
    positive_url = "https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv"
    negative_url = "https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv"

    try:
        positive_lexicon = set(pd.read_csv(positive_url, sep="\t", header=None)[0])
    except Exception as e:
        st.warning(f"Gagal memuat positive.tsv dari URL: {e}")
        positive_lexicon = set()

    try:
        negative_lexicon = set(pd.read_csv(negative_url, sep="\t", header=None)[0])
    except Exception as e:
        st.warning(f"Gagal memuat negative.tsv dari URL: {e}")
        negative_lexicon = set()

    return positive_lexicon, negative_lexicon

def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        return pd.read_excel(uploaded_file)
    except Exception:
        st.error("Format file tidak didukung. Harap unggah file CSV atau Excel.")
        return None

def load_preprocessed_data(uploaded_file):
    """Load preprocessed data from an uploaded CSV file."""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            # Convert 'stemmed' column from string representation to list
            if 'stemmed' in df.columns:
                df['stemmed'] = df['stemmed'].apply(
                    lambda x: ast.literal_eval(x.strip()) if isinstance(x, str) and x.strip() else []
                )
            return df
        else:
            st.error("File CSV hasil preprocessing belum diunggah.")
            return None
    except Exception as e:
        st.error(f"Gagal memuat data yang telah diproses: {e}")
        return None

def save_preprocessed_data(data, filepath="temp_preprocessed_data.csv"):
    """Save preprocessed data to a temporary CSV file."""
    try:
        data.to_csv(filepath, index=False)
        return True
    except Exception as e:
        st.error(f"Gagal menyimpan data yang telah diproses: {e}")
        return False

def preprocess_data(data):
    required_columns = ['Date', 'Username', 'Rating', 'Reviews Text']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"Kolom berikut tidak ditemukan: {', '.join(missing_columns)}")
        return None, 0
    processed = data[required_columns].drop_duplicates(subset='Reviews Text', keep='first')
    return processed, len(processed)

def clean_text(text):
    if pd.isna(text):
        return ''
    text = str(text

).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^a-z\s]', '', text).strip()
    return text

def normalize_text(text, norm_dict):
    if not text:
        return ''
    if norm_dict:
        return ' '.join(norm_dict.get(word, word) for word in text.split())
    return text

def tokenize_text(text):
    if text:
        return word_tokenize(text)
    return []

def remove_stopwords(tokens):
    if not tokens:
        return []
    filtered_tokens = [token for token in tokens if token not in STOPWORDS_ALL]
    return filtered_tokens

def stem_tokens(tokens):
    if not tokens:
        return []
    stemmed = [stemmer.stem(token) for token in tokens]
    return [token for token in stemmed if len(token) >= 3]

def label_sentiment(tokens, positive_words, negative_words):
    """
    Label sentiment based on tokenized words, counting matches in positive/negative lexicons.
    Returns a sentiment score and label (Positive, Negative, Neutral).
    """
    positive_count = sum(1 for token in tokens if token in positive_words)
    negative_count = sum(1 for token in tokens if token in negative_words)
    sentiment_score = positive_count - negative_count

    if sentiment_score > 0:
        sentiment = "Positive"
    elif sentiment_score < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment_score, sentiment

def preprocess_for_model(tokens):
    return ' '.join(tokens)

def generate_wordcloud(tokens_list, title, color):
    all_words = ' '.join(word for tokens in tokens_list for word in tokens)
    return WordCloud(width=800, height=400, background_color='white',
                     colormap=color, max_words=100).generate(all_words)

def generate_frequency_chart(tokens_list, top_n=10):
    all_words = [word for tokens in tokens_list for word in tokens]
    word_freq = Counter(all_words).most_common(top_n)
    words, freqs = zip(*word_freq)
    fig = go.Figure([go.Bar(x=words, y=freqs, text=freqs, textposition='auto',
                            marker_color='#4ECDC4', width=0.8)])
    fig.update_layout(title=f"Top {top_n} Kata Paling Sering Muncul",
                      xaxis_title="Kata", yaxis_title="Frekuensi",
                      title_x=0.5, height=450)
    return fig

def plot_confusion_matrix(y_test, predictions, model_name):
    cm = confusion_matrix(y_test, predictions, labels=['Negative', 'Neutral', 'Positive'])
    fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Negative', 'Neutral', 'Positive'],
                    y=['Negative', 'Neutral', 'Positive'])
    fig.update_layout(title=f"Confusion Matrix - {model_name}", title_x=0.5)
    return fig

def plot_data_split(train_size, test_size):
    fig = go.Figure([go.Bar(x=['Data Training', 'Data Uji'], y=[train_size, test_size],
                            text=[train_size, test_size], textposition='auto',
                            marker_color=['#66B3FF', '#FF9999'], width=0.8)])
    fig.update_layout(title="Perbandingan Data Training dan Uji",
                      yaxis_title="Jumlah Data", title_x=0.5, height=400)
    return fig

def plot_processing_counts(counts_dict):
    steps = list(counts_dict.keys())
    counts = list(counts_dict.values())
    fig = go.Figure([go.Bar(x=steps, y=counts, text=counts, textposition='auto',
                            marker_color='#66B3FF', width=0.8)])
    fig.update_layout(
        title="Jumlah Data Setelah Setiap Tahap Pemrosesan",
        xaxis_title="Tahap Pemrosesan",
        yaxis_title="Jumlah Data",
        title_x=0.5,
        height=400
    )
    return fig

def plot_accuracy_comparison(results):
    fig = go.Figure([go.Bar(x=list(results.keys()), y=[results[name]['acc'] for name in results],
                            text=[f"{results[name]['acc']:.2f}" for name in results],
                            textposition='auto', marker_color=['#FF6B6B', '#4ECDC4'], width=0.8)])
    fig.update_layout(title="Perbandingan Akurasi Model", yaxis_title="Akurasi",
                      yaxis_range=[0, 1], title_x=0.5, height=400)
    return fig