import streamlit as st

def guide():
    st.set_page_config(
        page_title="Panduan Pengguna - Analisis Sentimen",
        layout="wide"
    )

    st.title("🚀 Panduan Pengguna Aplikasi Analisis Sentimen")

    st.markdown(
        """
        Selamat datang di aplikasi Analisis Sentimen.  
        Panduan ini menunjukkan langkah-langkah utama proses mulai dari pengambilan data hingga visualisasi hasil.  
        Perhatikan diagram sebelah kiri dan deskripsi langkah prosedur di sebelah kanan.
        """
    )

    col1, col2 = st.columns([1, 1])

    graphviz_code = """
    digraph {
        rankdir=TB;  // Top to Bottom (default)
        node [shape=box, style=filled, color="#FF4B4B", fontname="Arial", fontsize=14, fontcolor=white, width=4.7, height=0.5];
        edge [color="#FF4B4B", penwidth=2];

        Scraping [label="Scraping Data Halaman scraping"];
        Preprocessing [label="Preprocessing Data Pada Halaman Analisis"];
        SentimentAnalysis [label="Analisis Sentimen"];
        Visualization [label="Visualisasi Hasil"];

        Scraping -> Preprocessing;
        Preprocessing -> SentimentAnalysis;
        SentimentAnalysis -> Visualization;
    }
    """

    with col1:
        st.subheader("Petunjuk Penggunaan")
        st.graphviz_chart(graphviz_code, use_container_width=True)

    with col2:
        st.subheader("Petunjuk Penggunaan")

        st.markdown("### 1. Scraping Data  Halaman scraping")
        st.write(
            """
            - Buka Browser kemudian cari google play store.
            - Masuk ke bagian aplikasi yang ingin di scraping datanya lalu salin url contoh "https://play.google.com/store/apps/details?id=id.co.bitcoin".
            - Masukan jumlah data yang diinginkan lalu unduh dataset berupa file csv tersebut.
            """
        )

        st.markdown("### 2. Preprocessing Data")
        st.write(
            """
            - Pergi kehalaman analisis.
            - Unggah file dataset file berupa csv atau jika belum mempunyai dataset bisa menuju halaman scraping dahulu.
            """
        )

        st.markdown("### 3. Analisis Sentimen")
        st.write(
            """
            - Menerapkan model untuk menentukan sentimen teks: positif, negatif, atau netral.
            - Model dapat menggunakan metode supervised learning dan naive bayes classifier.
            - Menghasilkan label sentimen dan skor kepercayaan untuk setiap data.
            """
        )

        st.markdown("### 4. Visualisasi Hasil")
        st.write(
            """
            - Menampilkan hasil analisis dalam bentuk grafik atau dashboard interaktif.
            - Memudahkan pemahaman distribusi sentimen.
            - Membantu pengambilan keputusan berbasis hasil analisis.
            """
        )

    st.markdown("---")
    st.info("📌 Gunakan menu utama aplikasi untuk mulai proses analisis sentimen Anda.")

if __name__ == "__main__":
    guide()
