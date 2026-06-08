import os
import json
import pandas as pd
import numpy as np
from services.model_service import ModelService
from pipeline.feature_engineering import calculate_features

class PredictorPipeline:
    def __init__(self, model_path="model/best_model.pt", scaler_path="model/scalers.pkl", baselines_path="config/baselines.json"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.baselines_path = baselines_path
        
        self.elevasi_dict = {}
        self.historical_means = {}  # Format: {kabupaten: {month_1: mean, month_2: mean, ...}}
        
        # Load model service
        self.model_service = ModelService(model_path=self.model_path, scaler_path=self.scaler_path)
        
        # Load baseline lookup tables
        self.load_baselines()

    def load_baselines(self):
        """Memuat data elevasi dan rata-rata historis bulanan per kabupaten."""
        resolved_path = self.baselines_path
        if not os.path.exists(resolved_path) and os.path.exists(os.path.join("backend", resolved_path)):
            resolved_path = os.path.join("backend", resolved_path)

        if not os.path.exists(resolved_path):
            # Jika file baselines belum dibuat, buat default kosong/nanti diisi oleh script pembuat
            print(f"[WARN] File baselines tidak ditemukan di {self.baselines_path} atau {resolved_path}. Harap jalankan script pembuat baseline.")
            return

        with open(resolved_path, 'r') as f:
            data = json.load(f)
            self.elevasi_dict = data.get("elevasi", {})
            self.historical_means = data.get("historical_means", {})
            print("[Predictor] Berhasil memuat lookup table elevasi dan mean historis.")

    def predict_kabupaten(self, df_14_days):
        """
        Menjalankan prediksi untuk 1 Kabupaten.
        df_14_days: DataFrame berisi 14 hari terakhir data satu kabupaten.
        """
        if len(df_14_days) != 14:
            raise ValueError(f"Data input harus tepat berdurasi 14 hari. Ditemukan: {len(df_14_days)} baris.")
        
        # Feature Engineering
        df_processed = calculate_features(df_14_days, self.elevasi_dict, self.historical_means, is_modis=self.model_service.is_modis)
        
        # Siapkan window data dinamis (14 baris)
        main_lst_col = 'MODIS_LST_Mean' if self.model_service.is_modis else 'ERA5_LST_Mean'
        df_window = df_processed[[main_lst_col, 'LST_Mean', 'LST_Anomaly']]
        
        # Siapkan fitur statis terasosiasi hari terakhir (baris ke-14)
        latest_row = df_processed.iloc[-1]
        df_static = {
            'Elevation_m': float(latest_row['Elevation_m']),
            'NDVI_8Day_Mean': float(latest_row['NDVI_8Day_Mean']),
            'SoilMoisture_Daily_Mean': float(latest_row['SoilMoisture_Daily_Mean'])
        }

        # Jalankan Model Inference
        results = self.model_service.predict(df_window, df_static)
        
        # Tambahkan metadata
        results["date"] = latest_row['date'].strftime('%Y-%m-%d')
        results["kabupaten"] = latest_row['Kabupaten']
        
        return results
