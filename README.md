# 🌡️ ThermaWatch (Dual-Branch ANFIS-LSTM LST Prediction)

**ThermaWatch** adalah aplikasi berbasis web interaktif untuk memproyeksikan **Suhu Permukaan Tanah (Land Surface Temperature / LST)** menggunakan arsitektur kecerdasan buatan **Dual-Branch ANFIS-LSTM**. Aplikasi ini dirancang untuk memantau, menganalisis, dan mensimulasikan anomali suhu di 15 Kabupaten/Kota Jawa Barat dengan memanfaatkan data historis satelit (ERA5 & MODIS) serta variabel lingkungan meteorologis.

---

## 🚀 Fitur Utama

### 1. 📊 Interactive Dashboard & GIS Map
- **Visualisasi Anomali Real-time**: Peta GIS Kabupaten/Kota Jawa Barat interaktif dengan sinkronisasi hover antartabel dan peta.
- **KPI Real-time**: Ringkasan kondisi suhu rata-rata, tingkat kelembaban, dan indeks vegetasi (NDVI).
- **Dual-Model Support**: Memungkinkan pengguna untuk beralih antara data sensor **ERA5** dan **MODIS**.

### 2. 📉 Environmental Analytics
- **Heatmap Korelasi**: Analisis hubungan antar-variabel meteorologi (LST, Soil Moisture, NDVI, Elevasi).
- **Distribusi Variabel**: Histogram interaktif dengan garis rata-rata untuk melihat pola data.
- **Statistik Deskriptif**: Tabel ringkasan statistik yang lengkap untuk analisis mendalam.

### 3. 🤖 Model Information & Architecture
- **Diagram Aliran Paralel**: Visualisasi interaktif arsitektur model Dual-Branch ANFIS-LSTM yang menunjukkan aliran *Forward Pass* dan proses *Update Weight* secara paralel selama *Backward Pass*.
- **Metrik Evaluasi**: Metrik performa uji komparatif (RMSE, MAE, R²) untuk berbagai horizon prediksi (H+1, H+3, H+7).
- **Detail Dataset**: Informasi lengkap dataset latih (2014-2026) yang terdiri dari ~86k sampel MODIS dan ~84k sampel ERA5.

### 4. 🔮 Simulation & Explainable AI (XAI)
- **What-If Analysis**: Pengguna dapat memanipulasi parameter lingkungan secara manual (LST, Kelembaban Tanah, NDVI, Elevasi, Bulan) untuk melihat proyeksi suhu mendatang.
- **Explainable AI (XAI)**: Panel dropdown penjelasan faktor prediksi yang menjabarkan kontribusi logika meteorologis di balik proyeksi model.

---

## 🛠️ Struktur Proyek

```directory
├── backend/
│   ├── config/              # Konfigurasi baseline rata-rata historis
│   ├── models/              # Model terlatih (StandardScaler & Weights)
│   └── services/            # Logika model prediction & Google Sheets integration
├── data/
│   └── geojson/             # GeoJSON batas administratif kabupaten Jawa Barat
├── frontend/
│   └── pages/               # Halaman-halaman antarmuka Streamlit:
│       ├── 1_Dashboard.py
│       ├── 2_Environmental_Analytics.py
│       ├── 3_Model_Information.py
│       └── 4_Simulation.py
├── app.py                   # Entrypoint navigasi utama aplikasi
├── requirements.txt         # Ketergantungan pustaka Python
└── README.md                # Dokumentasi proyek
```

---

## 💻 Cara Instalasi & Menjalankan Aplikasi

### Prerequisites
Pastikan Anda sudah menginstal **Python 3.10** atau versi yang lebih baru di sistem Anda.

### 1. Klon Repositori
```bash
git clone https://github.com/username/ThermaWatch-SoftComp.git
cd ThermaWatch-SoftComp
```

### 2. Pasang Ketergantungan Pustaka (Dependencies)
Gunakan pip untuk menginstal semua library yang dibutuhkan:
```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi Streamlit
Jalankan server pengembangan lokal dengan perintah:
```bash
streamlit run app.py
```

Setelah dijalankan, buka browser Anda dan akses aplikasi di URL default: `http://localhost:8501`.

---

## 🧠 Desain Arsitektur Model AI

Model utama menggabungkan keunggulan logika fuzzy dan memori jangka panjang:
- **Temporal Branch (LSTM)**: Memproses data time-series suhu historis (LST 14 hari terakhir) untuk menangkap tren musiman dan tren termal jangka pendek.
- **Environmental Branch (ANFIS)**: Memproses fitur statis & spasial (Soil Moisture, NDVI, Elevasi, Informasi Bulan) menggunakan *Adaptive Neuro-Fuzzy Inference System* untuk memahami interaksi non-linear parameter lingkungan.
- **Feature Fusion**: Menggabungkan representasi temporal dan lingkungan menggunakan Layer Normalization sebelum diteruskan ke Shared MLP dan output prediktor multi-horizon (H+1, H+3, H+7).
