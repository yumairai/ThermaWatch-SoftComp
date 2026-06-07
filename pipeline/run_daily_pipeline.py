import os
import argparse
import pandas as pd
from datetime import datetime, timedelta

# Import modul buatan kita
from pipeline.gee_extractor import GEEExtractor
from pipeline.predictor import PredictorPipeline
from pipeline.spreadsheet_writer import SpreadsheetWriter

def run_daily_pipeline(target_date_str=None):
    print("==============================================================")
    print("       MEMULAI DAILY PIPELINE RUN - THERMAWATCH AI            ")
    print("==============================================================")

    # 1. Tentukan tanggal target (Default: kemarin/H-1 karena latensi rilis data satelit)
    if not target_date_str:
        yesterday = datetime.now() - timedelta(days=1)
        target_date_str = yesterday.strftime('%Y-%m-%d')
    
    print(f"[Pipeline] Tanggal target pemrosesan: {target_date_str}")
    target_date = pd.to_datetime(target_date_str)
    
    # 2. Load Database Lokal
    local_db_path = 'Dataset_Master_ERA5_Ready_LSTM.csv'
    if not os.path.exists(local_db_path):
        raise FileNotFoundError(f"Database lokal '{local_db_path}' tidak ditemukan. Harap siapkan database master terlebih dahulu.")
        
    print(f"[Pipeline] Membaca database histori lokal: '{local_db_path}'...")
    df_db = pd.read_csv(local_db_path)
    df_db['date'] = pd.to_datetime(df_db['date'])
    df_db['Kabupaten'] = df_db['Kabupaten'].str.lower().str.strip()

    # Cek apakah tanggal target sudah ada di database lokal
    if target_date in df_db['date'].values:
        print(f"[WARN] Tanggal {target_date_str} sudah ada di database lokal. Proses akan menimpa baris tersebut.")
        # Hapus baris lama tersebut untuk mencegah duplikasi
        df_db = df_db[df_db['date'] != target_date]

    # 3. Ekstraksi Data dari GEE
    try:
        extractor = GEEExtractor("config/credentials.json")
    except Exception as e:
        print(f"[ERROR] Gagal inisialisasi GEE Extractor: {e}")
        return

    # Tarik data cuaca, soil, ndvi, dan landsat untuk hari target
    df_era5 = extractor.extract_era5(target_date_str)
    if df_era5.empty:
        print("[ERROR] Ekstraksi ERA5 gagal atau data belum rilis di GEE. Menghentikan pipeline.")
        return
        
    df_soil = extractor.extract_soil_moisture(target_date_str)
    df_ndvi = extractor.extract_ndvi(target_date_str)
    df_landsat = extractor.extract_landsat8(target_date_str)

    # Gabungkan data ekstraksi hari ini (satu baris per kabupaten)
    print("\n[Pipeline] Menggabungkan data satelit hari ini...")
    df_today = df_era5
    if not df_soil.empty:
        df_today = pd.merge(df_today, df_soil, on=['date', 'Kabupaten'], how='left')
    if not df_ndvi.empty:
        df_today = pd.merge(df_today, df_ndvi, on=['date', 'Kabupaten'], how='left')
    if not df_landsat.empty:
        df_today = pd.merge(df_today, df_landsat, on=['date', 'Kabupaten'], how='left')
        
    # Standarkan format data hari ini
    df_today['date'] = pd.to_datetime(df_today['date'])
    df_today['Kabupaten'] = df_today['Kabupaten'].str.lower().str.strip()
    df_today['month'] = df_today['date'].dt.month

    # 4. Inisialisasi Model Predictor
    predictor = PredictorPipeline(
        model_path="best_model.pt",
        scaler_path="scalers.pkl",
        baselines_path="config/baselines.json"
    )

    # List untuk menampung hasil final yang akan ditulis ke Sheets & CSV
    final_pipeline_outputs = []

    # 5. Iterasi Per Kabupaten untuk Predict & Update
    print("\n[Pipeline] Memulai inferensi AI per kabupaten...")
    kabupaten_list = df_today['Kabupaten'].unique()
    
    for kab in kabupaten_list:
        # Ambil data hari ini untuk kabupaten ini
        row_today = df_today[df_today['Kabupaten'] == kab].copy()
        if row_today.empty:
            continue
            
        # Ambil histori 29 hari ke belakang dari database lokal untuk kabupaten ini
        df_hist = df_db[df_db['Kabupaten'] == kab].sort_values(by='date').tail(29)
        
        if len(df_hist) < 29:
            print(f"[WARN] Histori kabupaten '{kab}' kurang dari 29 hari. Menggunakan data yang ada.")
        
        # Gabungkan histori 29 hari dengan data 1 hari terbaru
        df_30_days = pd.concat([df_hist, row_today], ignore_index=True)
        
        # Jika total data masih kurang dari 30 hari (misal kabupaten baru), ffill/bfill agar tidak error
        if len(df_30_days) < 30:
            shortage = 30 - len(df_30_days)
            dummy_rows = pd.concat([df_30_days.iloc[[0]]] * shortage, ignore_index=True)
            df_30_days = pd.concat([dummy_rows, df_30_days], ignore_index=True)
            df_30_days['date'] = [target_date - timedelta(days=i) for i in range(29, -1, -1)]

        # Jalankan kalkulasi fitur dan prediksi model ANFIS-LSTM
        try:
            prediction_res = predictor.predict_kabupaten(df_30_days)
            
            # Gabungkan input hari ini dengan output prediksi untuk disimpan
            kab_result = {
                "date": target_date_str,
                "Kabupaten": kab,
                "LST_Mean": row_today['LST_Mean'].values[0] if 'LST_Mean' in row_today.columns else None,
                "ERA5_LST_Mean": row_today['ERA5_LST_Mean'].values[0],
                "SoilMoisture_Daily_Mean": row_today['SoilMoisture_Daily_Mean'].values[0] if 'SoilMoisture_Daily_Mean' in row_today.columns else None,
                "NDVI_8Day_Mean": row_today['NDVI_8Day_Mean'].values[0] if 'NDVI_8Day_Mean' in row_today.columns else None,
                "Elevation_m": predictor.elevasi_dict.get(kab, 300.0),
                "month": int(row_today['month'].values[0]),
                # LST_Anomaly dihitung di predictor
                "LST_Anomaly": df_30_days['LST_Anomaly'].iloc[-1] if 'LST_Anomaly' in df_30_days.columns else 0.0,
                "prediction": prediction_res["prediction"],
                "ERA5_Max_Lat": row_today['ERA5_Max_Lat'].values[0],
                "ERA5_Max_Lon": row_today['ERA5_Max_Lon'].values[0]
            }
            final_pipeline_outputs.append(kab_result)
        except Exception as e:
            print(f"[ERROR] Gagal memprediksi kabupaten '{kab}': {e}")

    if not final_pipeline_outputs:
        print("[ERROR] Tidak ada prediksi yang berhasil dihasilkan.")
        return

    # 6. Tulis ke Google Sheets
    try:
        writer = SpreadsheetWriter(spreadsheet_name="thermawatch-data", credentials_path="config/credentials.json")
        writer.write_predictions(final_pipeline_outputs, sheet_name="daily_data")
    except Exception as e:
        print(f"[ERROR] Gagal menulis ke Google Sheets: {e}")
        # Lanjut saja karena setidaknya kita bisa update file lokal

    # 7. Update Database Master Lokal (CSV) untuk besok hari
    print("\n[Pipeline] Mengupdate file database lokal...")
    # Ubah hasil hari ini ke bentuk flat DataFrame untuk di-append ke CSV
    rows_local = []
    for item in final_pipeline_outputs:
        pred = item.get("prediction", {})
        h1 = pred.get("H1", {}).get("anomaly_temp", None)
        h3 = pred.get("H3", {}).get("anomaly_temp", None)
        h7 = pred.get("H7", {}).get("anomaly_temp", None)
        
        row = {
            "date": item["date"],
            "Kabupaten": item["Kabupaten"],
            "LST_Mean": item["LST_Mean"],
            "ERA5_LST_Mean": item["ERA5_LST_Mean"],
            "SoilMoisture_Daily_Mean": item["SoilMoisture_Daily_Mean"],
            "NDVI_8Day_Mean": item["NDVI_8Day_Mean"],
            "Elevation_m": item["Elevation_m"],
            "month": item["month"],
            "LST_Anomaly": item["LST_Anomaly"],
            "Target_Anomali_H1": h1, # Simpan prediksi H1 sebagai target sementara
            "Target_Anomali_H3": h3,
            "Target_Anomali_H7": h7,
            "ERA5_Max_Lat": item["ERA5_Max_Lat"],
            "ERA5_Max_Lon": item["ERA5_Max_Lon"]
        }
        rows_local.append(row)
        
    df_new_rows = pd.DataFrame(rows_local)
    df_new_rows['date'] = pd.to_datetime(df_new_rows['date'])
    
    # Gabungkan dengan DB lama dan urutkan
    df_db_updated = pd.concat([df_db, df_new_rows], ignore_index=True)
    df_db_updated = df_db_updated.sort_values(by=['Kabupaten', 'date']).reset_index(drop=True)
    
    # Simpan kembali ke file CSV
    df_db_updated.to_csv(local_db_path, index=False)
    print(f"[Pipeline] Database lokal '{local_db_path}' berhasil diperbarui.")
    print("==============================================================")
    print("             DAILY PIPELINE RUN SELESAI DENGAN SUKSES         ")
    print("==============================================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ThermaWatch Daily Pipeline.')
    parser.add_argument('--date', type=str, help='Target date (YYYY-MM-DD), default is yesterday.')
    args = parser.parse_args()
    
    run_daily_pipeline(args.date)
