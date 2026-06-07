import pandas as pd
import numpy as np

def calculate_features(df_window, elevasi_dict, historical_means):
    """
    Melakukan feature engineering pada data 30 hari terakhir:
    - Interpolasi linear data Landsat (LST_Mean) yang kosong
    - Mengisi data NDVI dan Soil Moisture dengan interpolasi/forward fill
    - Menghitung LST_Anomaly berdasarkan rata-rata historis bulanan
    """
    df = df_window.copy()
    df = df.sort_values(by='date').reset_index(drop=True)
    
    # Pastikan kolom date bertipe datetime
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    
    kabupaten = df['Kabupaten'].iloc[0]
    
    # 1. Tambah Elevasi
    elevation = elevasi_dict.get(kabupaten, 300.0)  # Default 300m jika tidak terdaftar
    df['Elevation_m'] = elevation

    # 2. Interpolasi data Landsat (LST_Mean) & data lainnya yang bolong (linear ffill bfill seperti preprocessing)
    numeric_cols = ['ERA5_LST_Mean', 'LST_Mean', 'SoilMoisture_Daily_Mean', 'NDVI_8Day_Mean']
    for col in numeric_cols:
        if col in df.columns:
            # Gunakan linear interpolation, ffill dan bfill untuk menambal data kosong harian
            df[col] = df[col].interpolate(method='linear').ffill().bfill()

    # 3. Hitung LST_Anomaly berbasis lookup month
    kab_means = historical_means.get(kabupaten, {})
    
    def get_historical_mean(row):
        month_str = str(int(row['month']))
        return kab_means.get(month_str, float(row['ERA5_LST_Mean']))  # fallback jika tidak ada

    df['LST_Historical_Mean'] = df.apply(get_historical_mean, axis=1)
    df['LST_Anomaly'] = df['ERA5_LST_Mean'] - df['LST_Historical_Mean']
    
    return df
