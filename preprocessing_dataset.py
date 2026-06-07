import pandas as pd
import numpy as np
import os

print("=== MEMULAI PREPROCESSING DATA THERMAWATCH (BASELINE: ERA5-LAND) ===\n")

# =========================================================================
# 1. LOAD DATASET DAN BASELINE ELEVASI
# =========================================================================
print("1. Memuat file CSV...")
df_era_lst = pd.read_csv('raw_data/ERA5_LST_Jabar_Daily_Clean.csv')
df_landsat = pd.read_csv('raw_data/Landsat8_Jabar_Satelit_Lintasan_Only.csv') 
df_soil = pd.read_csv('raw_data/GLDAS_SoilMoisture_Jabar_Daily.csv')
df_lama = pd.read_csv('raw_data/Fitur_Lengkap_Geothermal_Jabar_2014_2026.csv')

file_ndvi = 'raw_data/MODIS_NDVI_Jabar_8Day.csv'
file_rain = 'raw_data/Data_Hujan_GPM_Jabar_Harian_2025-14.csv'

df_ndvi = pd.read_csv(file_ndvi) if os.path.exists(file_ndvi) else pd.DataFrame(columns=['date', 'Kabupaten'])
df_rain = pd.read_csv(file_rain) if os.path.exists(file_rain) else pd.DataFrame(columns=['date', 'Kabupaten'])

# =========================================================================
# 2. STANDARISASI STRING WILAYAH KABUPATEN
# =========================================================================
def clean_kab(name):
    """Menghapus prefiks wilayah dan menstandarkan string menjadi huruf kecil."""
    return str(name).lower().strip().replace('kabupaten ', '').replace('kota ', '')

df_lama['Kabupaten_Clean'] = df_lama['Kabupaten'].apply(clean_kab)
elevasi_dict = df_lama.groupby('Kabupaten_Clean')['Elevation_m'].mean().to_dict()

# =========================================================================
# 3. CLEANSING FORMAT TANGGAL DAN SINKRONISASI DOY LANDSAT
# =========================================================================
def fix_doy_date(date_str):
    """Mengonversi format Day of Year (DOY) Landsat ke format standar YYYY-MM-DD."""
    if pd.isna(date_str) or 'system:index' in str(date_str) or 'date' in str(date_str):
        return np.nan
    
    date_str = str(date_str).strip()
    parts = date_str.split('-')
    
    if len(parts) == 3:
        try:
            if len(parts[2]) >= 3 or int(parts[2]) > 31:
                tahun = parts[0]
                doy = parts[2]
                return pd.to_datetime(f"{tahun}-{doy}", format="%Y-%j").strftime('%Y-%m-%d')
            else:
                return pd.to_datetime(date_str).strftime('%Y-%m-%d')
        except:
            return np.nan
    return date_str

print("2. Menyelaraskan nama wilayah dan memproses format penanggalan...")

# Memasukkan df_era_lst ke dalam loop pembersihan wilayah
for df in [df_ndvi, df_era_lst, df_landsat, df_soil, df_rain]:
    if not df.empty:
        df.drop(df[df['date'].astype(str).str.contains('system:index|date')].index, inplace=True, errors='ignore')
        df['Kabupaten'] = df['Kabupaten'].apply(clean_kab)

df_ndvi['date'] = df_ndvi['date'].apply(fix_doy_date)
df_era_lst['date'] = df_era_lst['date'].apply(fix_doy_date)
df_landsat['date'] = df_landsat['date'].apply(fix_doy_date)
df_soil['date'] = df_soil['date'].apply(fix_doy_date)

df_ndvi = df_ndvi.dropna(subset=['date'])
df_era_lst = df_era_lst.dropna(subset=['date'])
df_landsat = df_landsat.dropna(subset=['date'])
df_soil = df_soil.dropna(subset=['date'])
df_rain = df_rain.dropna(subset=['date']) if not df_rain.empty else df_rain

# =========================================================================
# 4. FILTERING OUTLIER EKSTREM LANDSAT & CLEANSING DATA ERA5 KOSONG (-9999)
# =========================================================================
# Filter outlier ekstrim Landsat tetap dipertahankan
invalid_landsat = (df_landsat['LST_Mean'] < 10) | (df_landsat['LST_Mean'] > 65)
df_landsat.loc[invalid_landsat, ['LST_Mean', 'LST_Max', 'LST_Percentile95']] = np.nan
if 'Max_Lon' in df_landsat.columns:
    df_landsat = df_landsat.drop(columns=['Max_Lon', 'Max_Lat'])

# Tambahan Pengaman: Jika ada koordinat atau nilai ERA5 bernilai -9999 (efek kecilnya kota), ubah ke NaN agar diisi bfill/ffill otomatis
for col in ['ERA5_LST_Mean', 'ERA5_LST_Max', 'ERA5_LST_Percentile95', 'ERA5_Max_Lon', 'ERA5_Max_Lat']:
    if col in df_era_lst.columns:
        df_era_lst[col] = df_era_lst[col].replace(-9999, np.nan)

# =========================================================================
# 5. MERGE DATASET MASTER (LEFT JOIN BERBASIS KALENDER HARIAN ERA5-LAND)
# =========================================================================
print("3. Menggabungkan multi-sensor (Memaksa Landsat & variabel lain mengikuti kalender harian ERA5)...")
# Mengunci data berbasis kalender kontinu penuh dari ERA5_LST
df_master = pd.merge(df_era_lst, df_landsat, on=['date', 'Kabupaten'], how='left', suffixes=('_era', '_landsat'))
if not df_ndvi.empty: df_master = pd.merge(df_master, df_ndvi, on=['date', 'Kabupaten'], how='left')
if not df_soil.empty: df_master = pd.merge(df_master, df_soil, on=['date', 'Kabupaten'], how='left')
if not df_rain.empty: df_master = pd.merge(df_master, df_rain, on=['date', 'Kabupaten'], how='left')

df_master['date'] = pd.to_datetime(df_master['date'], errors='coerce')
df_master = df_master.dropna(subset=['date'])

# =========================================================================
# 6. DEDUPLIKASI DATA HARIAN
# =========================================================================
print("4. Menciutkan data ganda menjadi satu baris unik per hari...")
metadata_cols = ['date', 'Kabupaten']
coord_cols = ['ERA5_Max_Lon', 'ERA5_Max_Lat'] # Diubah dari MODIS menjadi ERA5

mean_cols = [col for col in df_master.select_dtypes(include=[np.number]).columns if col not in coord_cols]
df_master_unique = df_master.groupby(metadata_cols)[mean_cols].mean().reset_index()
df_coords = df_master.groupby(metadata_cols)[coord_cols].first().reset_index()

df_master = pd.merge(df_master_unique, df_coords, on=metadata_cols, how='left')
df_master = df_master.sort_values(by=['Kabupaten', 'date']).reset_index(drop=True)

# =========================================================================
# 7. INTERPOLASI TEMPORAL (MENAMBAL JEDA LANDSAT BERBASIS KONTINUITAS ERA5)
# =========================================================================
print("5. Mengisi kekosongan data koordinat dan interpolasi linier variabel bolong...")
# Mengisi koordinat kosong ERA5 akibat limitasi geometri wilayah sempit
df_master['ERA5_Max_Lon'] = df_master.groupby('Kabupaten')['ERA5_Max_Lon'].ffill().bfill()
df_master['ERA5_Max_Lat'] = df_master.groupby('Kabupaten')['ERA5_Max_Lat'].ffill().bfill()

# Bersihkan kolom target lama agar tidak duplikat saat windowing ulang
cols_to_drop = ['Target_Anomali_H1', 'Target_Anomali_H3', 'Target_Anomali_H7']
df_master = df_master.drop(columns=cols_to_drop, errors='ignore')

# Mengisi seluruh data bolong (terutama data Landsat 16-harian) secara mulus mengikuti time-series ERA5
numeric_cols = df_master.select_dtypes(include=[np.number]).columns.tolist()
df_master[numeric_cols] = df_master.groupby('Kabupaten')[numeric_cols].transform(
    lambda x: x.interpolate(method='linear').ffill().bfill()
)

# =========================================================================
# 8. SUNTIK ELEVASI DAN KALKULASI ANOMALI TERMAL BERBASIS ERA5
# =========================================================================
print("6. Menyuntikkan nilai elevasi wilayah dan menghitung anomali termal berbasis ERA5...")
df_master['Elevation_m'] = df_master['Kabupaten'].map(elevasi_dict)
df_master['month'] = df_master['date'].dt.month

# Menghitung baseline historis bulanan dan nilai anomali berdasarkan fitur ERA5_LST_Mean
df_master['LST_Historical_Mean'] = df_master.groupby(['Kabupaten', 'month'])['ERA5_LST_Mean'].transform('mean')
df_master['LST_Anomaly'] = df_master['ERA5_LST_Mean'] - df_master['LST_Historical_Mean']

# =========================================================================
# 9. WINDOWING TARGET TIME-SERIES & CUT-OFF KALENDER
# =========================================================================
print("7. Menyusun target time-series windowing (H-1, H-3, H-7)...")
df_master['Target_Anomali_H1'] = df_master.groupby('Kabupaten')['LST_Anomaly'].shift(-1)
df_master['Target_Anomali_H3'] = df_master.groupby('Kabupaten')['LST_Anomaly'].shift(-3)
df_master['Target_Anomali_H7'] = df_master.groupby('Kabupaten')['LST_Anomaly'].shift(-7)

# Isi fallback nilai elevasi jika ada kabupaten baru yang belum terpetakan di kamus lama
df_master['Elevation_m'] = df_master['Elevation_m'].fillna(df_master['Elevation_m'].mean())

# --- PEMBATASAN STRATEGIS JANGKA PANJANG ---
# Disesuaikan dengan batas akhir data ekstraksi maksimum ERA5 kamu (Januari 2026)
df_master = df_master[(df_master['date'] >= '2014-01-01') & (df_master['date'] <= '2026-01-07')]

# Dropna ditaruh di paling akhir agar baris sisa pergeseran (shifiting) di akhir data dibersihkan otomatis
df_master = df_master.dropna(subset=['Target_Anomali_H1', 'Target_Anomali_H3', 'Target_Anomali_H7'])
# ----------------------------------------

# Ekspor hasil pengerjaan ke file master CSV baru
df_master.to_csv('Dataset_Master_ERA5_Ready_LSTM.csv', index=False)

print("\n=== PROSES PREPROCESSING RE-RUN SELESAI SEMPURNA ===")
print(f"📅 Rentang Data Aktual Baru: {df_master['date'].min().strftime('%Y-%m-%d')} s/d {df_master['date'].max().strftime('%Y-%m-%d')}")
print("File 'Dataset_Master_ERA5_Ready_LSTM.csv' siap dimasukkan ke training model AI ThermaWatch!")