import pandas as pd
import numpy as np
from services.sheets_service import SheetsService
from utils.status_classifier import classify_status

class SpreadsheetWriter:
    def __init__(self, spreadsheet_name="thermawatch-data", credentials_path="config/credentials.json"):
        self.spreadsheet_name = spreadsheet_name
        self.sheets_service = SheetsService(credentials_path=credentials_path)
        
        # Definisi skema kolom baru yang rapi dan lengkap
        self.headers = [
            "date", "kabupaten", "lst_mean", "lst_max", "lst_p95", "cloud_cover",
            "era5_lst_mean", "era5_lst_max", "era5_lst_p95", "soil_moisture",
            "elevation", "ndvi", "month", "lst_historical_mean", "lst_anomaly",
            "prediksi_anomali_h1", "prediksi_anomali_h3", "prediksi_anomali_h7",
            "prediksi_suhu_h1", "prediksi_suhu_h3", "prediksi_suhu_h7",
            "status_h1", "status_h3", "status_h7", "lat", "lon"
        ]

    def write_predictions(self, final_results, sheet_name="daily_data"):
        """
        Format dan tulis data hasil gabungan ekstraksi & prediksi ke Google Sheets dengan skema lengkap.
        """
        rows_to_append = []
        
        for item in final_results:
            # 1. Ambil nilai prediksi anomali dari model
            pred = item.get("prediction", {})
            anom_h1 = pred.get("H1", {}).get("anomaly_temp", None)
            anom_h3 = pred.get("H3", {}).get("anomaly_temp", None)
            anom_h7 = pred.get("H7", {}).get("anomaly_temp", None)
            
            # 2. Ambil nilai suhu normal bulanan historis
            hist_mean = item.get("LST_Historical_Mean", None)
            if hist_mean is None:
                # Fallback ambil dari ERA5_LST_Mean - LST_Anomaly jika tidak ada
                lst_now = item.get("ERA5_LST_Mean", None)
                lst_anom = item.get("LST_Anomaly", None)
                if lst_now is not None and lst_anom is not None:
                    hist_mean = lst_now - lst_anom
            
            # 3. Hitung prediksi suhu riil (Suhu Riil = Anomali Prediksi + Rerata Historis)
            suhu_h1 = None
            suhu_h3 = None
            suhu_h7 = None
            
            if hist_mean is not None:
                if anom_h1 is not None: suhu_h1 = float(anom_h1) + float(hist_mean)
                if anom_h3 is not None: suhu_h3 = float(anom_h3) + float(hist_mean)
                if anom_h7 is not None: suhu_h7 = float(anom_h7) + float(hist_mean)
            
            # 4. Klasifikasi status kerawanan berdasarkan prediksi anomali
            status_h1 = classify_status(anom_h1) if anom_h1 is not None else "NaN"
            status_h3 = classify_status(anom_h3) if anom_h3 is not None else "NaN"
            status_h7 = classify_status(anom_h7) if anom_h7 is not None else "NaN"
            
            # 5. Susun mentah baris data sesuai urutan header
            raw_row = [
                item.get("date"),
                item.get("Kabupaten"),
                item.get("LST_Mean") if pd.notna(item.get("LST_Mean")) else "NaN",
                item.get("LST_Max") if pd.notna(item.get("LST_Max")) else "NaN",
                item.get("LST_Percentile95") if pd.notna(item.get("LST_Percentile95")) else "NaN",
                item.get("Cloud_Cover_Percentage") if pd.notna(item.get("Cloud_Cover_Percentage")) else "NaN",
                item.get("ERA5_LST_Mean") if pd.notna(item.get("ERA5_LST_Mean")) else "NaN",
                item.get("ERA5_LST_Max") if pd.notna(item.get("ERA5_LST_Max")) else "NaN",
                item.get("ERA5_LST_Percentile95") if pd.notna(item.get("ERA5_LST_Percentile95")) else "NaN",
                item.get("SoilMoisture_Daily_Mean") if pd.notna(item.get("SoilMoisture_Daily_Mean")) else "NaN",
                item.get("Elevation_m") if pd.notna(item.get("Elevation_m")) else "NaN",
                item.get("NDVI_8Day_Mean") if pd.notna(item.get("NDVI_8Day_Mean")) else "NaN",
                item.get("month") if pd.notna(item.get("month")) else "NaN",
                hist_mean if hist_mean is not None else "NaN",
                item.get("LST_Anomaly") if pd.notna(item.get("LST_Anomaly")) else "NaN",
                anom_h1 if anom_h1 is not None else "NaN",
                anom_h3 if anom_h3 is not None else "NaN",
                anom_h7 if anom_h7 is not None else "NaN",
                suhu_h1 if suhu_h1 is not None else "NaN",
                suhu_h3 if suhu_h3 is not None else "NaN",
                suhu_h7 if suhu_h7 is not None else "NaN",
                status_h1,
                status_h3,
                status_h7,
                item.get("ERA5_Max_Lat") if pd.notna(item.get("ERA5_Max_Lat")) else "NaN",
                item.get("ERA5_Max_Lon") if pd.notna(item.get("ERA5_Max_Lon")) else "NaN"
            ]
            
            # 6. Konversi ke native Python types agar JSON serializable
            cleaned_row = []
            for val in raw_row:
                if pd.isna(val) or val is None:
                    cleaned_row.append("NaN")
                elif isinstance(val, (int, np.integer)):
                    cleaned_row.append(int(val))
                elif isinstance(val, (float, np.floating)):
                    cleaned_row.append(float(val))
                else:
                    cleaned_row.append(str(val))
                    
            rows_to_append.append(cleaned_row)
            
        if rows_to_append:
            # Hapus data duplikat untuk tanggal yang sama di Google Sheets jika ada
            try:
                sheet = self.sheets_service.get_worksheet(self.spreadsheet_name, sheet_name)
                # Ambil seluruh isi kolom pertama (tanggal)
                col_dates = sheet.col_values(1)
                
                # Himpunan tanggal yang akan ditulis baru
                dates_to_delete = set(item.get("date") for item in final_results if item.get("date"))
                
                # Temukan index baris (1-indexed) yang akan didelete
                rows_to_delete = []
                for idx, val in enumerate(col_dates):
                    if val in dates_to_delete:
                        rows_to_delete.append(idx + 1)
                
                # Delete rows dalam reverse order secara berkelompok agar efisien
                if rows_to_delete:
                    print(f"[Writer] Mendeteksi data lama untuk tanggal {list(dates_to_delete)}. Menghapus {len(rows_to_delete)} baris lama...")
                    # Sort reverse
                    rows_to_delete = sorted(rows_to_delete, reverse=True)
                    ranges = []
                    start = rows_to_delete[0]
                    end = rows_to_delete[0]
                    for r in rows_to_delete[1:]:
                        if r == start - 1:
                            start = r
                        else:
                            ranges.append((start, end))
                            start = r
                            end = r
                    ranges.append((start, end))
                    
                    for s, e in ranges:
                        sheet.delete_rows(s, e)
                    print("[Writer] Pembersihan data lama selesai.")
            except Exception as e:
                print(f"[Writer] [WARN] Gagal membersihkan baris duplikat di Google Sheets: {e}")

            print(f"[Writer] Menyimpan {len(rows_to_append)} baris data baru ke Google Sheet...")
            self.sheets_service.append_rows(
                spreadsheet_name=self.spreadsheet_name,
                rows_data=rows_to_append,
                sheet_name=sheet_name
            )
            print("[Writer] Penulisan data ke Google Sheets sukses!")
        else:
            print("[Writer] Tidak ada data yang ditulis.")
