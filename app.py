import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))

import streamlit as st

# Konfigurasi Utama Halaman (Harus dipanggil paling pertama)
st.set_page_config(
    page_title="ThermaWatch | Home",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def render_home():
    # Menampilkan Logo atau Banner
    st.markdown("<h1 style='text-align: center; color: #f97316;'>🌍 ThermaWatch</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #94a3b8;'>Sistem Monitoring Geothermal </h4>", unsafe_allow_html=True)
    
    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Selamat Datang di Aplikasi ThermaWatch!
        Aplikasi ini dirancang untuk memantau, menganalisis, dan memprediksi anomali suhu permukaan serta kualitas lingkungan dengan memanfaatkan data citra satelit dari **Google Earth Engine (GEE)**.
        
        #### 🧭 Navigasi Menu
        Silakan gunakan menu di *sidebar* sebelah kiri untuk mengakses berbagai fitur:
        
        - **📊 Dashboard**: Pantau kondisi suhu aktual, persebaran spasial pada peta, dan tren historis.
        - **🍃 Environmental Analytics**: Analisis mendalam mengenai metrik lingkungan lainnya seperti *Soil Moisture* dan indeks vegetasi (NDVI).
        - **🤖 Model Information**: Pelajari arsitektur prediksi *Machine Learning* (ANFIS-LSTM) yang digunakan oleh sistem ini.
        - **🔬 Simulation**: Lakukan simulasi kondisi suhu berdasarkan input parameter lingkungan buatan.
        """)

    with col2:
        st.info("""
        **💡 Informasi Sistem**
        - **Data Source**: Google Earth Engine (GEE)
        - **Storage**: Google Sheets (Sinkronisasi harian)
        - **Model**: Dual-Branch ANFIS-LSTM
        - **Status**: 🟢 Aktif & Berjalan
        """)

    st.divider()
    st.markdown("<p style='text-align: center; font-size: 0.8rem; color: gray;'>© 2026 ThermaWatch Project | Soft Computing</p>", unsafe_allow_html=True)

# Membuat dictionary navigasi menggunakan st.Page
pages = {
    "Beranda": [
        st.Page(render_home, title="Home", icon="🏠", default=True),
    ],
    "Menu Utama": [
        st.Page("frontend/pages/1_Dashboard.py", title="Dashboard", icon="📊"),
        st.Page("frontend/pages/2_Environmental_Analytics.py", title="Environmental Analytics", icon="🍃"),
        st.Page("frontend/pages/3_Model_Information.py", title="Model Information", icon="🤖"),
        st.Page("frontend/pages/4_Simulation.py", title="Simulation", icon="🔬"),
    ]
}

# Terapkan navigasi multi-page native dari Streamlit (v1.36+)
pg = st.navigation(pages)
pg.run()

