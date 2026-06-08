import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import modul buatan kita
from pipeline.gee_extractor import GEEExtractor
from pipeline.predictor import PredictorPipeline
from pipeline.spreadsheet_writer import SpreadsheetWriter
from pipeline.feature_engineering import calculate_features

def run_daily_pipeline(target_date_str=None):
    print("==============================================================")
    print("       MEMULAI DAILY PIPELINE RUN - THERMAWATCH AI            ")
    print("==============================================================")

    # 1. Inisialisasi GEE Extractor terlebih dahulu untuk memeriksa tanggal satelit
    credentials_path = "config/credentials.json"
    if not os.path.exists(credentials_path) and os.path.exists("backend/config/credentials.json"):
        credentials_path = "backend/config/credentials.json"

    try:
        extractor = GEEExtractor(credentials_path)
    except Exception as e:
        print(f"[ERROR] Gagal inisialisasi GEE Extractor: {e}")
        return

    # 2. Tentukan tanggal target (Default: kemarin/H-1 secara kalender)
    if not target_date_str:
        yesterday = datetime.now() - timedelta(days=1)
        target_date_str = yesterday.strftime('%Y-%m-%d')
    
    print(f"[Pipeline] Tanggal target pemrosesan: {target_date_str}")
    target_date = pd.to_datetime(target_date_str)
    
    # 3. Load Database Lokal
    local_db_era5_path = 'data/Dataset_Master_ERA5_Ready_LSTM.csv'
    local_db_modis_path = 'data/Dataset_(Jan,2014-Mei,2026).csv'
    
    if not os.path.exists(local_db_era5_path):
        raise FileNotFoundError(f"Database lokal ERA5 '{local_db_era5_path}' tidak ditemukan.")
    if not os.path.exists(local_db_modis_path):
        raise FileNotFoundError(f"Database lokal MODIS '{local_db_modis_path}' tidak ditemukan.")
        
    print(f"[Pipeline] Membaca database histori lokal ERA5...")
    df_db_era5 = pd.read_csv(local_db_era5_path)
    df_db_era5['date'] = pd.to_datetime(df_db_era5['date'])
    df_db_era5['Kabupaten'] = df_db_era5['Kabupaten'].str.lower().str.strip()

    print(f"[Pipeline] Membaca database histori lokal MODIS...")
    df_db_modis = pd.read_csv(local_db_modis_path)
    df_db_modis['date'] = pd.to_datetime(df_db_modis['date'])
    df_db_modis['Kabupaten'] = df_db_modis['Kabupaten'].str.lower().str.strip()

    # Cek apakah tanggal target sudah ada di database lokal
    if target_date in df_db_era5['date'].values:
        print(f"[WARN] Tanggal {target_date_str} sudah ada di database lokal ERA5. Proses akan menimpa baris tersebut.")
        df_db_era5 = df_db_era5[df_db_era5['date'] != target_date]

    if target_date in df_db_modis['date'].values:
        print(f"[WARN] Tanggal {target_date_str} sudah ada di database lokal MODIS. Proses akan menimpa baris tersebut.")
        df_db_modis = df_db_modis[df_db_modis['date'] != target_date]

    # Tarik data cuaca, soil, ndvi, dan landsat untuk hari target
    df_era5 = extractor.extract_era5(target_date_str)
    if df_era5.empty:
        print("[ERROR] Ekstraksi ERA5 gagal atau data belum rilis di GEE. Menghentikan pipeline.")
        return
        
    df_soil = extractor.extract_soil_moisture(target_date_str)
    df_ndvi = extractor.extract_ndvi(target_date_str)
    df_landsat = extractor.extract_landsat8(target_date_str)
    df_modis = extractor.extract_modis(target_date_str)

    # Gabungkan data ekstraksi hari ini (satu baris per kabupaten)
    print("\n[Pipeline] Menggabungkan data satelit hari ini...")
    df_today = df_era5.copy()
    if not df_soil.empty:
        df_today = pd.merge(df_today, df_soil, on=['date', 'Kabupaten'], how='left')
    if not df_ndvi.empty:
        df_today = pd.merge(df_today, df_ndvi, on=['date', 'Kabupaten'], how='left')
    if not df_landsat.empty:
        df_today = pd.merge(df_today, df_landsat, on=['date', 'Kabupaten'], how='left')
    if not df_modis.empty:
        df_today = pd.merge(df_today, df_modis, on=['date', 'Kabupaten'], how='left')
        
    # Standarkan format data hari ini dan ubah ke numerik agar bisa di-interpolate
    df_today['date'] = pd.to_datetime(df_today['date'])
    df_today['Kabupaten'] = df_today['Kabupaten'].str.lower().str.strip()
    df_today['month'] = df_today['date'].dt.month

    # Konversi kolom-kolom satelit ke tipe float
    numeric_cols = [
        'ERA5_LST_Mean', 'ERA5_LST_Max', 'ERA5_LST_Percentile95', 
        'SoilMoisture_Daily_Mean', 'NDVI_8Day_Mean', 
        'LST_Mean', 'LST_Max', 'LST_Percentile95', 'Cloud_Cover_Percentage',
        'MODIS_LST_Mean', 'MODIS_LST_Max', 'MODIS_LST_Percentile95', 'MODIS_Data_Availability', 'MODIS_QC_Raw'
    ]
    for col in numeric_cols:
        if col in df_today.columns:
            df_today[col] = pd.to_numeric(df_today[col], errors='coerce')

    # Filter Awan MODIS: Jika data_availability < 20% (tutupan awan > 80%), set ke NaN agar di-interpolate
    if 'MODIS_Data_Availability' in df_today.columns:
        cloudy_mask = df_today['MODIS_Data_Availability'] < 0.20
        df_today.loc[cloudy_mask, ['MODIS_LST_Mean', 'MODIS_LST_Max', 'MODIS_LST_Percentile95']] = np.nan

    # Hubungkan koordinat spasial MODIS dengan ERA5 sebagai default
    if 'MODIS_Max_Lon' not in df_today.columns and 'ERA5_Max_Lon' in df_today.columns:
        df_today['MODIS_Max_Lon'] = df_today['ERA5_Max_Lon']
    if 'MODIS_Max_Lat' not in df_today.columns and 'ERA5_Max_Lat' in df_today.columns:
        df_today['MODIS_Max_Lat'] = df_today['ERA5_Max_Lat']

    # Deduplikasi poligon parsial kabupaten (merata-ratakan nilai LST/Soil, dan ambil koordinat pertama)
    coord_cols = ['ERA5_Max_Lon', 'ERA5_Max_Lat', 'MODIS_Max_Lon', 'MODIS_Max_Lat']
    mean_cols = [c for c in df_today.columns if c not in coord_cols and c not in ['date', 'Kabupaten']]
    
    df_today_unique = df_today.groupby(['date', 'Kabupaten'])[mean_cols].mean().reset_index()
    df_today_coords = df_today.groupby(['date', 'Kabupaten'])[coord_cols].first().reset_index()
    df_today = pd.merge(df_today_unique, df_today_coords, on=['date', 'Kabupaten'], how='left')

    # 4. Extract NOAA GFS daily temperature for delta-scaling (Range: target_date - 6 days to target_date)
    gfs_start = (target_date - timedelta(days=6)).strftime('%Y-%m-%d')
    gfs_end = target_date_str
    print(f"\n[Pipeline] Mengekstrak data GFS untuk delta-scaling dari {gfs_start} s/d {gfs_end}...")
    try:
        df_gfs_all = extractor.extract_gfs_range(gfs_start, gfs_end)
        if not df_gfs_all.empty:
            df_gfs_all['date'] = pd.to_datetime(df_gfs_all['date'])
            df_gfs_all['Kabupaten'] = df_gfs_all['Kabupaten'].str.lower().str.strip()
            # Deduplicate GFS data by averaging over the same date and kabupaten (e.g. Kabupaten/Kota Bandung)
            df_gfs_all = df_gfs_all.groupby(['date', 'Kabupaten'])['GFS_Temp'].mean().reset_index()
    except Exception as e:
        print(f"[Pipeline] [WARN] Gagal mengekstrak GFS: {e}. Delta-scaling dilewati.")
        df_gfs_all = pd.DataFrame()

    # 5. Inisialisasi Model Predictor untuk ERA5 dan MODIS
    print("\n[Pipeline] Menginisialisasi Model Predictor ERA5 dan MODIS...")
    predictor_era5 = PredictorPipeline(
        model_path="model/best_model.pt",
        scaler_path="model/scalers.pkl",
        baselines_path="config/baselines.json"
    )
    predictor_modis = PredictorPipeline(
        model_path="model/best_model_modis.pt",
        scaler_path="model/scalers_modis.pkl",
        baselines_path="config/baselines.json"
    )

    # List untuk menampung hasil final yang akan ditulis ke Sheets & CSV
    final_outputs_era5 = []
    final_outputs_modis = []

    # 6. Iterasi Per Kabupaten untuk Predict & Update
    print("\n[Pipeline] Memulai inferensi AI per kabupaten untuk ERA5 & MODIS...")
    kabupaten_list = df_today['Kabupaten'].unique()
    
    for kab in kabupaten_list:
        row_today = df_today[df_today['Kabupaten'] == kab].copy()
        if row_today.empty:
            continue
            
        # ----------------------------------------------------
        # PROSES INFERENSI ERA5
        # ----------------------------------------------------
        try:
            df_hist_era5 = df_db_era5[(df_db_era5['Kabupaten'] == kab) & (df_db_era5['date'] < target_date)].sort_values(by='date').tail(13)
            df_14_days_era5 = pd.concat([df_hist_era5, row_today], ignore_index=True)
            if len(df_14_days_era5) < 14:
                shortage = 14 - len(df_14_days_era5)
                dummy_rows = pd.concat([df_14_days_era5.iloc[[0]]] * shortage, ignore_index=True)
                df_14_days_era5 = pd.concat([dummy_rows, df_14_days_era5], ignore_index=True)
                df_14_days_era5['date'] = [target_date - timedelta(days=i) for i in range(13, -1, -1)]

            if not df_gfs_all.empty:
                if 'GFS_Temp' in df_14_days_era5.columns:
                    df_14_days_era5 = df_14_days_era5.drop(columns=['GFS_Temp'])
                df_14_days_era5 = pd.merge(df_14_days_era5, df_gfs_all, on=['date', 'Kabupaten'], how='left')

            df_processed_era5 = calculate_features(df_14_days_era5, predictor_era5.elevasi_dict, predictor_era5.historical_means, is_modis=False)
            prediction_res_era5 = predictor_era5.model_service.predict(
                df_processed_era5[['ERA5_LST_Mean', 'LST_Mean', 'LST_Anomaly']], 
                {
                    'Elevation_m': float(df_processed_era5['Elevation_m'].iloc[-1]),
                    'NDVI_8Day_Mean': float(df_processed_era5['NDVI_8Day_Mean'].iloc[-1]),
                    'SoilMoisture_Daily_Mean': float(df_processed_era5['SoilMoisture_Daily_Mean'].iloc[-1])
                }
            )
            
            target_row_mask_era5 = df_processed_era5['date'] == target_date
            latest_row_era5 = df_processed_era5[target_row_mask_era5].iloc[0] if target_row_mask_era5.any() else df_processed_era5.iloc[-1]
            lat_val = latest_row_era5['ERA5_Max_Lat']
            lon_val = latest_row_era5['ERA5_Max_Lon']
            
            if pd.isna(lat_val) or lat_val < -90 or lat_val == -9999.0:
                valid_hist_lat = df_hist_era5[df_hist_era5['ERA5_Max_Lat'].notna() & (df_hist_era5['ERA5_Max_Lat'] > -90) & (df_hist_era5['ERA5_Max_Lat'] != -9999.0)]
                lat_val = valid_hist_lat['ERA5_Max_Lat'].iloc[-1] if not valid_hist_lat.empty else -7.0
            if pd.isna(lon_val) or lon_val < 0 or lon_val == -9999.0:
                valid_hist_lon = df_hist_era5[df_hist_era5['ERA5_Max_Lon'].notna() & (df_hist_era5['ERA5_Max_Lon'] > 0) & (df_hist_era5['ERA5_Max_Lon'] != -9999.0)]
                lon_val = valid_hist_lon['ERA5_Max_Lon'].iloc[-1] if not valid_hist_lon.empty else 107.6

            kab_result_era5 = {
                "date": target_date_str,
                "Kabupaten": kab,
                "LST_Mean": latest_row_era5['LST_Mean'],
                "LST_Max": latest_row_era5.get('LST_Max', None),
                "LST_Percentile95": latest_row_era5.get('LST_Percentile95', None),
                "Cloud_Cover_Percentage": latest_row_era5.get('Cloud_Cover_Percentage', None),
                "ERA5_LST_Mean": latest_row_era5['ERA5_LST_Mean'],
                "ERA5_LST_Max": latest_row_era5.get('ERA5_LST_Max', None),
                "ERA5_LST_Percentile95": latest_row_era5.get('ERA5_LST_Percentile95', None),
                "SoilMoisture_Daily_Mean": latest_row_era5['SoilMoisture_Daily_Mean'],
                "Elevation_m": latest_row_era5['Elevation_m'],
                "NDVI_8Day_Mean": latest_row_era5['NDVI_8Day_Mean'],
                "month": int(latest_row_era5['month']),
                "LST_Historical_Mean": latest_row_era5.get('LST_Historical_Mean', None),
                "LST_Anomaly": latest_row_era5['LST_Anomaly'],
                "prediction": prediction_res_era5["prediction"],
                "ERA5_Max_Lat": float(lat_val),
                "ERA5_Max_Lon": float(lon_val)
            }
            final_outputs_era5.append(kab_result_era5)
        except Exception as e:
            print(f"[ERROR] Gagal memprediksi ERA5 untuk kabupaten '{kab}': {e}")

        # ----------------------------------------------------
        # PROSES INFERENSI MODIS
        # ----------------------------------------------------
        try:
            df_hist_modis = df_db_modis[(df_db_modis['Kabupaten'] == kab) & (df_db_modis['date'] < target_date)].sort_values(by='date').tail(13)
            df_14_days_modis = pd.concat([df_hist_modis, row_today], ignore_index=True)
            if len(df_14_days_modis) < 14:
                shortage = 14 - len(df_14_days_modis)
                dummy_rows = pd.concat([df_14_days_modis.iloc[[0]]] * shortage, ignore_index=True)
                df_14_days_modis = pd.concat([dummy_rows, df_14_days_modis], ignore_index=True)
                df_14_days_modis['date'] = [target_date - timedelta(days=i) for i in range(13, -1, -1)]

            if not df_gfs_all.empty:
                if 'GFS_Temp' in df_14_days_modis.columns:
                    df_14_days_modis = df_14_days_modis.drop(columns=['GFS_Temp'])
                df_14_days_modis = pd.merge(df_14_days_modis, df_gfs_all, on=['date', 'Kabupaten'], how='left')

            df_processed_modis = calculate_features(df_14_days_modis, predictor_modis.elevasi_dict, predictor_modis.historical_means, is_modis=True)
            prediction_res_modis = predictor_modis.model_service.predict(
                df_processed_modis[['MODIS_LST_Mean', 'LST_Mean', 'LST_Anomaly']], 
                {
                    'Elevation_m': float(df_processed_modis['Elevation_m'].iloc[-1]),
                    'NDVI_8Day_Mean': float(df_processed_modis['NDVI_8Day_Mean'].iloc[-1]),
                    'SoilMoisture_Daily_Mean': float(df_processed_modis['SoilMoisture_Daily_Mean'].iloc[-1])
                }
            )
            
            target_row_mask_modis = df_processed_modis['date'] == target_date
            latest_row_modis = df_processed_modis[target_row_mask_modis].iloc[0] if target_row_mask_modis.any() else df_processed_modis.iloc[-1]
            lat_val_modis = latest_row_modis['MODIS_Max_Lat']
            lon_val_modis = latest_row_modis['MODIS_Max_Lon']
            
            if pd.isna(lat_val_modis) or lat_val_modis < -90 or lat_val_modis == -9999.0:
                valid_hist_lat = df_hist_modis[df_hist_modis['MODIS_Max_Lat'].notna() & (df_hist_modis['MODIS_Max_Lat'] > -90) & (df_hist_modis['MODIS_Max_Lat'] != -9999.0)]
                lat_val_modis = valid_hist_lat['MODIS_Max_Lat'].iloc[-1] if not valid_hist_lat.empty else -7.0
            if pd.isna(lon_val_modis) or lon_val_modis < 0 or lon_val_modis == -9999.0:
                valid_hist_lon = df_hist_modis[df_hist_modis['MODIS_Max_Lon'].notna() & (df_hist_modis['MODIS_Max_Lon'] > 0) & (df_hist_modis['MODIS_Max_Lon'] != -9999.0)]
                lon_val_modis = valid_hist_lon['MODIS_Max_Lon'].iloc[-1] if not valid_hist_lon.empty else 107.6

            kab_result_modis = {
                "date": target_date_str,
                "Kabupaten": kab,
                "LST_Mean": latest_row_modis['LST_Mean'],
                "LST_Max": latest_row_modis.get('LST_Max', None),
                "LST_Percentile95": latest_row_modis.get('LST_Percentile95', None),
                "Cloud_Cover_Percentage": latest_row_modis.get('Cloud_Cover_Percentage', None),
                "MODIS_LST_Mean": latest_row_modis['MODIS_LST_Mean'],
                "MODIS_LST_Max": latest_row_modis.get('MODIS_LST_Max', None),
                "MODIS_LST_Percentile95": latest_row_modis.get('MODIS_LST_Percentile95', None),
                "MODIS_Data_Availability": latest_row_modis.get('MODIS_Data_Availability', 1.0),
                "MODIS_QC_Raw": latest_row_modis.get('MODIS_QC_Raw', 0),
                "SoilMoisture_Daily_Mean": latest_row_modis['SoilMoisture_Daily_Mean'],
                "Elevation_m": latest_row_modis['Elevation_m'],
                "NDVI_8Day_Mean": latest_row_modis['NDVI_8Day_Mean'],
                "month": int(latest_row_modis['month']),
                "LST_Historical_Mean": latest_row_modis.get('LST_Historical_Mean', None),
                "LST_Anomaly": latest_row_modis['LST_Anomaly'],
                "prediction": prediction_res_modis["prediction"],
                "MODIS_Max_Lat": float(lat_val_modis),
                "MODIS_Max_Lon": float(lon_val_modis)
            }
            final_outputs_modis.append(kab_result_modis)
        except Exception as e:
            print(f"[ERROR] Gagal memprediksi MODIS untuk kabupaten '{kab}': {e}")

    # 7. Tulis ke Google Sheets (Masing-masing sheet)
    writer = SpreadsheetWriter(spreadsheet_name="thermawatch-data", credentials_path=credentials_path)
    
    if final_outputs_era5:
        try:
            print("\n[Pipeline] Menulis prediksi ERA5 ke Google Sheet 'daily_data'...")
            writer.write_predictions(final_outputs_era5, sheet_name="daily_data", is_modis=False)
        except Exception as e:
            print(f"[ERROR] Gagal menulis prediksi ERA5 ke Google Sheets: {e}")
            
    if final_outputs_modis:
        try:
            print("\n[Pipeline] Menulis prediksi MODIS ke Google Sheet 'daily_data_modis'...")
            writer.write_predictions(final_outputs_modis, sheet_name="daily_data_modis", is_modis=True)
        except Exception as e:
            print(f"[ERROR] Gagal menulis prediksi MODIS ke Google Sheets: {e}")

    # 8. Update Database Master Lokal (CSV) ERA5 & MODIS
    if final_outputs_era5:
        print("\n[Pipeline] Mengupdate file database lokal ERA5...")
        rows_local_era5 = []
        for item in final_outputs_era5:
            pred = item.get("prediction", {})
            h1 = pred.get("H1", {}).get("anomaly_temp", None)
            h3 = pred.get("H3", {}).get("anomaly_temp", None)
            h7 = pred.get("H7", {}).get("anomaly_temp", None)
            
            row = {
                "date": item["date"],
                "Kabupaten": item["Kabupaten"],
                "LST_Mean": item["LST_Mean"],
                "LST_Max": item.get("LST_Max", None),
                "LST_Percentile95": item.get("LST_Percentile95", None),
                "Cloud_Cover_Percentage": item.get("Cloud_Cover_Percentage", None),
                "ERA5_LST_Mean": item["ERA5_LST_Mean"],
                "ERA5_LST_Max": item.get("ERA5_LST_Max", None),
                "ERA5_LST_Percentile95": item.get("ERA5_LST_Percentile95", None),
                "ERA5_Cloud_Cover_Percentage": item.get("ERA5_Cloud_Cover_Percentage", 0),
                "SoilMoisture_Daily_Mean": item["SoilMoisture_Daily_Mean"],
                "NDVI_8Day_Mean": item["NDVI_8Day_Mean"],
                "Elevation_m": item["Elevation_m"],
                "month": item["month"],
                "LST_Historical_Mean": item.get("LST_Historical_Mean", None),
                "LST_Anomaly": item["LST_Anomaly"],
                "Target_Anomali_H1": h1,
                "Target_Anomali_H3": h3,
                "Target_Anomali_H7": h7,
                "ERA5_Max_Lat": item["ERA5_Max_Lat"],
                "ERA5_Max_Lon": item["ERA5_Max_Lon"]
            }
            rows_local_era5.append(row)
            
        df_new_era5 = pd.DataFrame(rows_local_era5)
        df_new_era5['date'] = pd.to_datetime(df_new_era5['date'])
        df_db_updated_era5 = pd.concat([df_db_era5, df_new_era5], ignore_index=True)
        df_db_updated_era5 = df_db_updated_era5.sort_values(by=['Kabupaten', 'date']).reset_index(drop=True)
        df_db_updated_era5.to_csv(local_db_era5_path, index=False)
        print(f"[Pipeline] Database lokal ERA5 '{local_db_era5_path}' berhasil diperbarui.")

    if final_outputs_modis:
        print("\n[Pipeline] Mengupdate file database lokal MODIS...")
        rows_local_modis = []
        for item in final_outputs_modis:
            pred = item.get("prediction", {})
            h1 = pred.get("H1", {}).get("anomaly_temp", None)
            h3 = pred.get("H3", {}).get("anomaly_temp", None)
            h7 = pred.get("H7", {}).get("anomaly_temp", None)
            
            row = {
                "date": item["date"],
                "Kabupaten": item["Kabupaten"],
                "MODIS_LST_Mean": item["MODIS_LST_Mean"],
                "MODIS_LST_Max": item.get("MODIS_LST_Max", None),
                "MODIS_LST_Percentile95": item.get("MODIS_LST_Percentile95", None),
                "MODIS_Data_Availability": item.get("MODIS_Data_Availability", 1.0),
                "MODIS_QC_Raw": item.get("MODIS_QC_Raw", 0),
                "LST_Mean": item["LST_Mean"],
                "LST_Max": item.get("LST_Max", None),
                "LST_Percentile95": item.get("LST_Percentile95", None),
                "Cloud_Cover_Percentage": item.get("Cloud_Cover_Percentage", None),
                "SoilMoisture_Daily_Mean": item["SoilMoisture_Daily_Mean"],
                "NDVI_8Day_Mean": item["NDVI_8Day_Mean"],
                "Elevation_m": item["Elevation_m"],
                "month": item["month"],
                "LST_Historical_Mean": item.get("LST_Historical_Mean", None),
                "LST_Anomaly": item["LST_Anomaly"],
                "Target_Anomali_H1": h1,
                "Target_Anomali_H3": h3,
                "Target_Anomali_H7": h7,
                "MODIS_Max_Lat": item["MODIS_Max_Lat"],
                "MODIS_Max_Lon": item["MODIS_Max_Lon"]
            }
            rows_local_modis.append(row)
            
        df_new_modis = pd.DataFrame(rows_local_modis)
        df_new_modis['date'] = pd.to_datetime(df_new_modis['date'])
        df_db_updated_modis = pd.concat([df_db_modis, df_new_modis], ignore_index=True)
        df_db_updated_modis = df_db_updated_modis.sort_values(by=['Kabupaten', 'date']).reset_index(drop=True)
        df_db_updated_modis.to_csv(local_db_modis_path, index=False)
        print(f"[Pipeline] Database lokal MODIS '{local_db_modis_path}' berhasil diperbarui.")

    print("==============================================================")
    print("             DAILY PIPELINE RUN SELESAI DENGAN SUKSES         ")
    print("==============================================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ThermaWatch Daily Pipeline.')
    parser.add_argument('--date', type=str, help='Target date (YYYY-MM-DD), default is yesterday.')
    args = parser.parse_args()
    
    if args.date:
        run_daily_pipeline(args.date)
    else:
        # Menjalankan sliding window healing untuk 15 hari terakhir (H-15 s/d H-1) secara berurutan
        print("\n==============================================================")
        print("    MEMULAI SLIDING WINDOW HEALING & PREDIKSI (H-15 s/d H-1)    ")
        print("==============================================================")
        
        today = datetime.now()
        dates_to_run = []
        for i in range(15, 0, -1):
            d_str = (today - timedelta(days=i)).strftime('%Y-%m-%d')
            dates_to_run.append(d_str)
            
        print(f"[Scheduler] Rentang tanggal yang akan diproses: {dates_to_run}")
        
        for d_str in dates_to_run:
            print(f"\n[Scheduler] >>> MEMULAI PROSES TANGGAL: {d_str} <<<")
            try:
                run_daily_pipeline(d_str)
            except Exception as e:
                print(f"[Scheduler] [ERROR] Gagal memproses tanggal {d_str}: {e}")
        
        print("\n==============================================================")
        print("         SELURUH SLIDING WINDOW RUN SELESAI DENGAN SUKSES       ")
        print("==============================================================")


