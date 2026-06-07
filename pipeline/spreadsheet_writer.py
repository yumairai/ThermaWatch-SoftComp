import pandas as pd
from services.sheets_service import SheetsService
from utils.status_classifier import classify_status

class SpreadsheetWriter:
    def __init__(self, spreadsheet_name="thermawatch-data", credentials_path="config/credentials.json"):
        self.spreadsheet_name = spreadsheet_name
        self.sheets_service = SheetsService(credentials_path=credentials_path)

    def write_predictions(self, final_results, sheet_name="daily_data"):
        """
        Format dan tulis data hasil gabungan ekstraksi & prediksi ke Google Sheets.
        
        final_results: list of dict, masing-masing berisi gabungan data GEE + Prediksi AI
        """
        rows_to_append = []
        
        for item in final_results:
            # Mengambil hasil prediksi
            pred = item.get("prediction", {})
            h1_val = pred.get("H1", {}).get("anomaly_temp", None)
            h3_val = pred.get("H3", {}).get("anomaly_temp", None)
            h7_val = pred.get("H7", {}).get("anomaly_temp", None)
            
            # Klasifikasi status kerawanan
            status_h1 = classify_status(h1_val) if h1_val is not None else "NaN"
            status_h3 = classify_status(h3_val) if h3_val is not None else "NaN"
            status_h7 = classify_status(h7_val) if h7_val is not None else "NaN"
            
            # Sesuaikan dengan schema tabel Google Sheets:
            # date | kabupaten | lst_mean | modis_lst_mean | soil_moisture | elevation | ndvi | month | anomaly | h1 | h3 | h7 | status_h1 | status_h3 | status_h7 | lat | lon
            row = [
                item.get("date"),
                item.get("Kabupaten"),
                item.get("LST_Mean") if pd.notna(item.get("LST_Mean")) else "NaN",
                item.get("ERA5_LST_Mean") if pd.notna(item.get("ERA5_LST_Mean")) else "NaN",
                item.get("SoilMoisture_Daily_Mean") if pd.notna(item.get("SoilMoisture_Daily_Mean")) else "NaN",
                item.get("Elevation_m") if pd.notna(item.get("Elevation_m")) else "NaN",
                item.get("NDVI_8Day_Mean") if pd.notna(item.get("NDVI_8Day_Mean")) else "NaN",
                int(item.get("month")) if pd.notna(item.get("month")) else "NaN",
                item.get("LST_Anomaly") if pd.notna(item.get("LST_Anomaly")) else "NaN",
                h1_val if h1_val is not None else "NaN",
                h3_val if h3_val is not None else "NaN",
                h7_val if h7_val is not None else "NaN",
                status_h1,
                status_h3,
                status_h7,
                item.get("ERA5_Max_Lat") if pd.notna(item.get("ERA5_Max_Lat")) else "NaN",
                item.get("ERA5_Max_Lon") if pd.notna(item.get("ERA5_Max_Lon")) else "NaN"
            ]
            rows_to_append.append(row)
            
        if rows_to_append:
            print(f"[Writer] Menyimpan {len(rows_to_append)} baris data ke Google Sheet...")
            self.sheets_service.append_rows(
                spreadsheet_name=self.spreadsheet_name,
                rows_data=rows_to_append,
                sheet_name=sheet_name
            )
            print("[Writer] Penulisan data ke Google Sheets sukses!")
        else:
            print("[Writer] Tidak ada data yang ditulis.")
