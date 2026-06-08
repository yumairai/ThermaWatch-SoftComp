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

    # 2. Delta-scaling for ERA5 LST variables using GFS_Temp if GFS is present
    if 'GFS_Temp' in df.columns:
        # Interpolate GFS_Temp first
        df['GFS_Temp'] = df['GFS_Temp'].interpolate(method='linear').ffill().bfill()
        
        # Find the last row with valid ERA5_LST_Mean before we interpolate
        non_null_era5 = df[df['ERA5_LST_Mean'].notna()]
        if not non_null_era5.empty:
            anchor_idx = non_null_era5.index[-1]
            anchor_gfs = df.loc[anchor_idx, 'GFS_Temp']
            
            # Apply GFS delta for rows after the anchor
            for idx in df.index[anchor_idx + 1:]:
                delta = df.loc[idx, 'GFS_Temp'] - anchor_gfs
                for col in ['ERA5_LST_Mean', 'ERA5_LST_Max', 'ERA5_LST_Percentile95']:
                    if col in df.columns and pd.isna(df.loc[idx, col]):
                        anchor_val = df.loc[anchor_idx, col]
                        if pd.notna(anchor_val) and anchor_val != -9999.0:
                            df.loc[idx, col] = anchor_val + delta

    # 3. Interpolate other missing data (linear ffill bfill like preprocessing)
    numeric_cols = [
        'ERA5_LST_Mean', 'ERA5_LST_Max', 'ERA5_LST_Percentile95', 
        'LST_Mean', 'LST_Max', 'LST_Percentile95', 
        'SoilMoisture_Daily_Mean', 'NDVI_8Day_Mean', 'Cloud_Cover_Percentage'
    ]
    for col in numeric_cols:
        if col in df.columns:
            # Linear interpolation, ffill and bfill as a fallback
            df[col] = df[col].interpolate(method='linear').ffill().bfill()

    # 3. Hitung LST_Anomaly berbasis lookup month
    kab_means = historical_means.get(kabupaten, {})
    
    def get_historical_mean(row):
        month_str = str(int(row['month']))
        return kab_means.get(month_str, float(row['ERA5_LST_Mean']))  # fallback jika tidak ada

    df['LST_Historical_Mean'] = df.apply(get_historical_mean, axis=1)
    df['LST_Anomaly'] = df['ERA5_LST_Mean'] - df['LST_Historical_Mean']
    
    return df
