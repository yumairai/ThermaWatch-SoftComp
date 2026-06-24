# ThermaWatch (Dual-Branch ANFIS-LSTM LST Prediction)

ThermaWatch adalah aplikasi berbasis web interaktif untuk memproyeksikan Suhu Permukaan Tanah (Land Surface Temperature / LST) menggunakan arsitektur kecerdasan buatan Dual-Branch ANFIS-LSTM. Aplikasi ini dirancang untuk memantau, menganalisis, dan mensimulasikan anomali suhu di 15 Kabupaten/Kota Jawa Barat dengan memanfaatkan data historis satelit (ERA5 & MODIS) serta variabel lingkungan meteorologis.

---

## Apa yang Bisa Anda Lakukan dan Lihat di Aplikasi Ini?

### 1. Dashboard Interaktif & Peta Pemantauan Suhu
- Memantau Peta Sebaran Panas: Anda akan melihat peta interaktif Provinsi Jawa Barat. Titik atau warna pada peta menunjukkan lokasi anomali suhu (misalnya area yang memanas secara ekstrem).
- Memeriksa Indikator Utama (KPI): Di layar utama, Anda akan langsung disajikan panel ringkasan kondisi suhu rata-rata terkini, tingkat kelembaban tanah, dan indeks vegetasi.
- Memilih Sumber Data Satelit: Anda bebas mengganti mode pembacaan data sensor dari satelit ERA5 ke MODIS (atau sebaliknya) untuk membandingkan informasi.

### 2. Analisis Lingkungan (Environmental Analytics)
- Membaca Pola Hubungan Faktor Alam: Anda bisa melihat grafik korelasi (Heatmap) yang memperjelas seberapa kuat pengaruh kepadatan vegetasi, elevasi, dan kelembaban tanah terhadap peningkatan suhu permukaan.
- Melihat Sebaran Data Harian: Tersedia grafik histogram interaktif untuk meninjau distribusi dan tren data meteorologi dari waktu ke waktu.

### 3. Informasi Arsitektur Model AI
- Mempelajari Cara Kerja AI: Terdapat diagram interaktif yang memperlihatkan dengan jelas bagaimana sistem AI memproses data waktu (Temporal) dan data alam (Lingkungan) secara bersamaan.
- Memeriksa Tingkat Akurasi Prediksi: Anda bisa melihat tabel nilai akurasi (seperti RMSE dan MAE) untuk menilai seberapa presisi prediksi AI dalam memproyeksikan suhu 1, 3, hingga 7 hari ke depan.

### 4. Simulasi Mandiri & Penjelasan Logika AI
- Melakukan Eksperimen Prediksi Sendiri (What-If): Anda dapat bereksperimen dengan memasukkan angka buatan sendiri (misalnya mengubah kelembaban menjadi sangat kering atau menaikkan suhu awal). Sistem akan langsung memproyeksikan suhu di masa depan.
- Membaca Alasan Prediksi (Explainable AI): Setelah prediksi muncul, aplikasi akan menyajikan penjelasan naratif dan logis mengenai mengapa suhu tersebut diprediksi naik atau turun (contoh: prediksi suhu menjadi tinggi karena sistem mendeteksi ketiadaan pohon penyejuk dan kelembaban yang sangat rendah).

---

## Struktur Proyek

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

## Panduan Penggunaan & Instalasi

### Prerequisites
Pastikan Anda sudah menginstal Python 3.10 atau versi yang lebih baru di sistem Anda.

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

Setelah dijalankan, buka browser Anda dan akses aplikasi di URL default: http://localhost:8501.

---

## Desain Arsitektur Model AI

Model utama menggabungkan keunggulan logika fuzzy dan memori jangka panjang:
- Temporal Branch (LSTM): Memproses data time-series suhu historis (LST 14 hari terakhir) untuk menangkap tren musiman dan tren termal jangka pendek.
- Environmental Branch (ANFIS): Memproses fitur statis & spasial (Soil Moisture, NDVI, Elevasi, Informasi Bulan) menggunakan Adaptive Neuro-Fuzzy Inference System untuk memahami interaksi non-linear parameter lingkungan.
- Feature Fusion: Menggabungkan representasi temporal dan lingkungan menggunakan Layer Normalization sebelum diteruskan ke Shared MLP dan output prediktor multi-horizon (H+1, H+3, H+7).
