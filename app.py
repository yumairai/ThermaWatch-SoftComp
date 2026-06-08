import streamlit as st

# Konfigurasi Utama Halaman (Harus dipanggil paling pertama)
st.set_page_config(
    page_title="ThermaWatch | Home",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Menampilkan Logo atau Banner (Opsional, gunakan emoji atau gambar)
    st.markdown("<h1 style='text-align: center; color: #f97316;'>🌍 ThermaWatch - SoftComp</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #94a3b8;'>Sistem Pemantauan Suhu & Kualitas Lingkungan Berbasis Geospasial</h4>", unsafe_allow_html=True)
    
    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Selamat Datang di Aplikasi ThermaWatch!
        Aplikasi ini dirancang untuk memantau, menganalisis, dan memprediksi anomali suhu permukaan serta kualitas lingkungan dengan memanfaatkan data citra satelit dari **Google Earth Engine (GEE)**.
        
        #### 🧭 Navigasi Menu
        Silakan gunakan menu di *sidebar* sebelah kiri untuk mengakses berbagai fitur:
        
        - **📊 1. Dashboard**: Pantau kondisi suhu aktual, persebaran spasial pada peta, dan tren historis.
        - **🍃 2. Environmental Analytics**: Analisis mendalam mengenai metrik lingkungan lainnya seperti *Soil Moisture* dan indeks vegetasi (NDVI).
        - **🤖 3. Model Information**: Pelajari arsitektur prediksi *Machine Learning* (ANFIS-LSTM) yang digunakan oleh sistem ini.
        - **🔬 4. Simulation**: Lakukan simulasi kondisi suhu berdasarkan input parameter lingkungan buatan.
        """)

    with col2:
        st.info("""
        **💡 Informasi Sistem**
        - **Data Source**: Google Earth Engine (GEE)
        - **Storage**: Google Sheets (Sinkronisasi harian)
        - **Model**: Dual-Branch ANFIS-LSTM
        - **Status**: 🟢 Aktif & Berjalan
        """)
        
        # Tambahkan ilustrasi sederhana atau gambar jika ada di folder assets/
        # st.image("assets/logo.png", use_column_width=True)

    st.divider()
    st.markdown("<p style='text-align: center; font-size: 0.8rem; color: gray;'>© 2026 ThermaWatch Project | Soft Computing</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
