import streamlit as st
import pandas as pd
from utils.utils import *
from models.models import prepare_data, train_and_evaluate_models

# Konfigurasi Streamlit
st.set_page_config(layout="wide", page_title="Sentiment Analysis Dashboard")

def main():
    st.title("Sentiment Analysis Dashboard")
    st.markdown("Unggah file CSV hasil preprocessing untuk analisis sentimen.")

    # File uploader for preprocessed CSV
    uploaded_file = st.file_uploader("Unggah file CSV hasil preprocessing", type=["csv"], key="preprocessed_data_uploader")

    if uploaded_file is None:
        st.info("Silakan unggah file CSV hasil preprocessing untuk memulai analisis.")
        return

    with st.spinner("Memproses analisis..."):
        # Load preprocessed data
        processed_data = load_preprocessed_data(uploaded_file)
        if processed_data is None:
            return

        # Verify required columns
        required_columns = ['review_text', 'stemmed', 'Text_for_Model', 'Sentiment']
        missing_columns = [col for col in required_columns if col not in processed_data.columns]
        if missing_columns:
            st.error(f"Kolom berikut tidak ditemukan di dataset: {', '.join(missing_columns)}")
            return

        # --- Visualization Section ---
        st.subheader("📈 Visualisasi Hasil Analisis")

        # Data Overview - Total Data Only
        st.write("### Ringkasan Data")
        st.write("**Total Data**")
        st.write(f"{len(processed_data)} Ulasan")

        # Sentiment Table
        st.write("### Sentimen")
        st.write("Tabel ini menampilkan teks asli, kata-kata yang telah distem, dan label sentimen.")
        st.dataframe(processed_data[['review_text', 'stemmed', 'Sentiment']])
        st.write("Distribusi Sentimen:", processed_data['Sentiment'].value_counts())

        # Check for sufficient sentiment classes
        unique_classes = processed_data['Sentiment'].nunique()
        if unique_classes <= 1:
            st.error(f"Hanya ditemukan {unique_classes} kelas sentimen: {processed_data['Sentiment'].unique()}. "
                     "Model tidak dapat dilatih karena memerlukan minimal 2 kelas.")
            return

        # Word Clouds
        st.write("### Visualisasi Preprocessing (Word Clouds)")
        col1, col2 = st.columns(2)

        with col1:
            tabs = st.tabs(["Semua Data", "Positif", "Netral", "Negatif"])
            with tabs[0]:
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.imshow(generate_wordcloud(processed_data['stemmed'], "Semua Data", 'viridis'), interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)

            for tab, sentiment, color in zip(tabs[1:], ['Positive', 'Neutral', 'Negative'], ['Greens', 'Greys', 'Reds']):
                with tab:
                    sentiment_data = processed_data[processed_data['Sentiment'] == sentiment]['stemmed']
                    if len(sentiment_data) > 0:
                        fig, ax = plt.subplots(figsize=(6, 3))
                        ax.imshow(generate_wordcloud(sentiment_data, f"{sentiment}", color), interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)

        with col2:
            st.write("#### Distribusi Sentimen")
            sentiment_counts = processed_data['Sentiment'].value_counts()
            colors = {'Positive': '#00FF00', 'Negative': '#FF0000', 'Neutral': '#808080'}
            color_list = [colors.get(sentiment, '#808080') for sentiment in sentiment_counts.index]
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Distribusi Sentimen",
                hole=0.3,
                color_discrete_sequence=color_list
            )
            st.plotly_chart(fig, use_container_width=True)

        # Top 10 Words
        st.write("### Top 10 Kata Paling Sering Muncul")
        st.plotly_chart(generate_frequency_chart(processed_data['stemmed']), use_container_width=True)

        # Model Training and Evaluation
        st.write("### Hasil Model")
        X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = prepare_data(
            processed_data['Text_for_Model'], processed_data['Sentiment']
        )
        results = train_and_evaluate_models(X_train_tfidf, X_test_tfidf, y_train, y_test)

        col_model1, col_model2 = st.columns(2)
        with col_model1:
            st.plotly_chart(plot_data_split(X_train_tfidf.shape[0], X_test_tfidf.shape[0]), use_container_width=True)
        with col_model2:
            st.plotly_chart(plot_accuracy_comparison(results), use_container_width=True)

        for name in results:
            st.write(f"#### {name}")
            col_eval1, col_eval2 = st.columns(2)
            with col_eval1:
                st.plotly_chart(plot_confusion_matrix(y_test, results[name]['pred'], name), use_container_width=True)
            with col_eval2:
                report_df = pd.DataFrame(results[name]['report']).T.round(2)
                st.write("Laporan Klasifikasi:")
                st.dataframe(report_df.style.highlight_max(axis=0, subset=['precision', 'recall', 'f1-score']))

if __name__ == "__main__":
    main()