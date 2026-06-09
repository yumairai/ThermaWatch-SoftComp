import pandas as pd
import numpy as np

def calculate_features(df_window, elevasi_dict, historical_means, is_modis=False):
    """
    Melakukan feature engineering pada data 14/30 hari terakhir:
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

    # 2. Delta-scaling for LST variables using GFS_Temp if GFS is present
    if 'GFS_Temp' in df.columns:
        # Interpolate GFS_Temp first
        df['GFS_Temp'] = df['GFS_Temp'].interpolate(method='linear').ffill().bfill()
        
        # Tentukan kolom LST utama yang akan di-scaling
        lst_mean_col = 'MODIS_LST_Mean' if is_modis else 'ERA5_LST_Mean'
        lst_max_col = 'MODIS_LST_Max' if is_modis else 'ERA5_LST_Max'
        lst_p95_col = 'MODIS_LST_Percentile95' if is_modis else 'ERA5_LST_Percentile95'
        
        # Cari baris terakhir yang memiliki nilai LST valid sebelum di-interpolasi
        non_null_lst = df[df[lst_mean_col].notna()]
        if not non_null_lst.empty:
            anchor_idx = non_null_lst.index[-1]
            anchor_gfs = df.loc[anchor_idx, 'GFS_Temp']
            
            # Terapkan GFS delta untuk baris-baris setelah anchor
            for idx in df.index[anchor_idx + 1:]:
                delta = df.loc[idx, 'GFS_Temp'] - anchor_gfs
                for col in [lst_mean_col, lst_max_col, lst_p95_col]:
                    if col in df.columns and pd.isna(df.loc[idx, col]):
                        anchor_val = df.loc[anchor_idx, col]
                        if pd.notna(anchor_val) and anchor_val != -9999.0:
                            df.loc[idx, col] = anchor_val + delta

    # 3. Interpolate other missing data (linear ffill bfill like preprocessing)
    numeric_cols = [
        'ERA5_LST_Mean', 'ERA5_LST_Max', 'ERA5_LST_Percentile95', 
        'LST_Mean', 'LST_Max', 'LST_Percentile95', 
        'SoilMoisture_Daily_Mean', 'NDVI_8Day_Mean', 'Cloud_Cover_Percentage',
        'MODIS_LST_Mean', 'MODIS_LST_Max', 'MODIS_LST_Percentile95', 'MODIS_Data_Availability', 'MODIS_QC_Raw'
    ]
    for col in numeric_cols:
        if col in df.columns:
            # Linear interpolation, ffill and bfill as a fallback
            df[col] = df[col].interpolate(method='linear').ffill().bfill()

    # 3. Hitung LST_Anomaly berbasis lookup month
    kab_means = historical_means.get(kabupaten, {})
    
    # Tentukan kolom acuan utama untuk anomali
    base_col = 'MODIS_LST_Mean' if is_modis else 'ERA5_LST_Mean'
    
    def get_historical_mean(row):
        month_str = str(int(row['month']))
        return kab_means.get(month_str, float(row[base_col]))  # fallback jika tidak ada

    df['LST_Historical_Mean'] = df.apply(get_historical_mean, axis=1)
    df['LST_Anomaly'] = df[base_col] - df['LST_Historical_Mean']
    
    return df
