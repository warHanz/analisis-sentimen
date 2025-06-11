import streamlit as st
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="CryptoSentiment Analyzer",
    page_icon="📊",
    layout="wide",
)

# Function to load and apply CSS from file
def load_css(css_file):
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.warning(f"CSS file not found: {css_file}")

def add_logo_to_sidebar():
    logo_path = os.path.join("assets", "image", "logo.png")  # Ganti dengan path logo Anda
    if os.path.exists(logo_path):
        logo = Image.open(logo_path)
        st.sidebar.image(logo, use_column_width=True)
    else:
        st.sidebar.markdown("Logo tidak ditemukan.")

# Hero section with full-width image and overlay text
def hero_section():
    image_path = os.path.join("assets", "image", "hero.jpg")
    
    # Create hero section container
    st.markdown('<div class="hero-section">', unsafe_allow_html=True)
    
    if os.path.exists(image_path):
        img = Image.open(image_path)
        st.image(img, use_container_width=False)
    else:
        # Fallback to a colored div if image doesn't exist
        st.markdown("""
        <div style="width:100%; height:30px; background:linear-gradient(135deg, #18d26e, #0a6e3a); margin-top:-80px;"></div>
        """, unsafe_allow_html=True)

    
    # Add extra spacing after hero section
    st.markdown('<div style="margin-top: 0px;"></div>', unsafe_allow_html=True)

# Description section
def description_section():
    # Add some space after hero section
    st.write("")
    st.write("")
    
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Tentang Aplikasi</h2>', unsafe_allow_html=True)
    st.markdown("""
     <p>
                CryptoSentiment Analyzer adalah platform analisis sentimen yang dirancang khusus 
                untuk menganalisis ulasan dan review aplikasi exchange crypto dari platform 
                Google Play Store. Aplikasi ini membantu investor, trader, dan pengembang 
                untuk memahami persepsi publik terhadap platform trading crypto favorit mereka.
                Dengan memanfaatkan teknologi Natural Language Processing (NLP) dan Machine Learning, 
                untuk memberikan insights mendalam tentang sentimen pengguna, tren kepuasan, dan aspek-aspek 
                yang bisa diperhatikan oleh komunitas crypto trading.
            </p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Features section
def features_section():
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Fitur Utama</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h3 class="feature-title">Sentimen Analisis</h3>
            <p class="feature-desc">Ketahui Sentimen ulasan pengguna terhadap aplikasi.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <h3 class="feature-title">Sentimen Text</h3>
            <p class="feature-desc">Mengetahui Hasil sentimen Positif, netral dan Negatif.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📈</div>
            <h3 class="feature-title">Visualisasi Dashboard</h3>
            <p class="feature-desc">Visualisasi hasil data mengenai sentimen ulasan aplikasi.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Call to action section
def cta_section():
    st.markdown('<div class="content-section" style="text-align: center;">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title" style="display: inline-block; margin-bottom: 1.5rem;">Mulai Analisis Anda</h2>', unsafe_allow_html=True)
    st.markdown("""
    <p class="section-text" style="margin-bottom: 2rem;">
        Siap untuk memulai? Eksplorasi sentimen aplikasi exchange crypto sekarang juga.
    </p>
    """, unsafe_allow_html=True)
    # Use Streamlit's button for navigation, keep the default button style
    if st.button("Lihat Analisis Sentimen", key="cta_button"):
        st.switch_page("pages/4_📖_Analisis.py")
    st.markdown('</div>', unsafe_allow_html=True)

# Main app
def home():
    # Load CSS from file
    css_file = os.path.join("assets", "style.css")
    load_css(css_file)
    
    hero_section()
    description_section()
    features_section()
    cta_section()

if __name__ == "__main__":
    home()