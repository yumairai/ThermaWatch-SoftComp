import os
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd

class SheetsService:
    def __init__(self, credentials_path="config/credentials.json"):
        self.credentials_path = credentials_path
        self.scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        self.client = None
        self.authenticate()

    def authenticate(self):
        """Autentikasi ke Google Sheets menggunakan file service account credentials."""
        resolved_path = self.credentials_path
        if not os.path.exists(resolved_path) and os.path.exists(os.path.join("backend", resolved_path)):
            resolved_path = os.path.join("backend", resolved_path)

        if not os.path.exists(resolved_path):
            raise FileNotFoundError(
                f"File credentials tidak ditemukan di {self.credentials_path} atau {resolved_path}. "
                f"Pastikan Anda telah meletakkan file credentials.json di folder config."
            )

        creds = Credentials.from_service_account_file(
            resolved_path, 
            scopes=self.scopes
        )
        self.client = gspread.authorize(creds)

    def get_worksheet(self, spreadsheet_name, sheet_name="daily_data"):
        """Membuka spreadsheet berdasarkan nama dan mengambil worksheet tertentu."""
        try:
            spreadsheet = self.client.open(spreadsheet_name)
            return spreadsheet.worksheet(sheet_name)
        except gspread.SpreadsheetNotFound:
            raise ValueError(
                f"Spreadsheet dengan nama '{spreadsheet_name}' tidak ditemukan. "
                f"Pastikan nama sudah benar dan file spreadsheet sudah di-share ke email Service Account."
            )
        except gspread.WorksheetNotFound:
            # Jika worksheet tidak ada, buat baru
            spreadsheet = self.client.open(spreadsheet_name)
            return spreadsheet.add_worksheet(title=sheet_name, rows="1000", cols="20")

    def read_data(self, spreadsheet_name, sheet_name="daily_data"):
        """Membaca seluruh data dari worksheet menjadi pandas DataFrame."""
        sheet = self.get_worksheet(spreadsheet_name, sheet_name)
        records = sheet.get_all_records()
        return pd.DataFrame(records)

    def append_row(self, spreadsheet_name, row_data, sheet_name="daily_data"):
        """Menambahkan satu baris data baru ke bagian paling bawah worksheet."""
        sheet = self.get_worksheet(spreadsheet_name, sheet_name)
        # Pastikan data berupa list
        if isinstance(row_data, pd.Series):
            row_to_append = row_data.tolist()
        elif isinstance(row_data, dict):
            # Jika berupa dict, pastikan urutan kolom sesuai dengan header di sheet
            headers = sheet.row_values(1)
            row_to_append = [row_data.get(header, "") for header in headers]
        else:
            row_to_append = list(row_data)
            
        sheet.append_row(row_to_append)

    def append_rows(self, spreadsheet_name, rows_data, sheet_name="daily_data"):
        """Menambahkan banyak baris sekaligus."""
        sheet = self.get_worksheet(spreadsheet_name, sheet_name)
        sheet.append_rows(rows_data)


# Fungsi Helper untuk Frontend Streamlit
import streamlit as st

@st.cache_data(ttl=3600)
def get_daily_data(spreadsheet_name="thermawatch-data"):
    """Mengambil data harian untuk dashboard frontend dengan caching."""
    service = SheetsService()
    df = service.read_data(spreadsheet_name=spreadsheet_name, sheet_name="daily_data")
    
    # Mapping kolom jika perlu, atau mengembalikan langsung
    # Dashboard mencari "Suhu_Celcius", "Tanggal", dsb, mari buat mapping sederhana 
    # berdasarkan header yang sebenarnya
    if not df.empty:
        if 'date' in df.columns and 'Tanggal' not in df.columns:
            df['Tanggal'] = pd.to_datetime(df['date'])
        if 'lst_mean' in df.columns and 'Suhu_Celcius' not in df.columns:
            # Gunakan kolom prediksi_suhu_h1 atau lst_mean sebagai "Suhu_Celcius" di UI
            df['Suhu_Celcius'] = pd.to_numeric(df['lst_mean'], errors='coerce')
        if 'soil_moisture' in df.columns and 'Soil_Moisture' not in df.columns:
            df['Soil_Moisture'] = pd.to_numeric(df['soil_moisture'], errors='coerce')
        if 'ndvi' in df.columns and 'NDVI' not in df.columns:
            df['NDVI'] = pd.to_numeric(df['ndvi'], errors='coerce')
        if 'lst_anomaly' in df.columns and 'Anomaly' not in df.columns:
            df['Anomaly'] = pd.to_numeric(df['lst_anomaly'], errors='coerce')
        if 'kabupaten' in df.columns and 'Kabupaten' not in df.columns:
            df['Kabupaten'] = df['kabupaten']
        if 'lat' in df.columns and 'Latitude' not in df.columns:
            df['Latitude'] = pd.to_numeric(df['lat'], errors='coerce')
        if 'lon' in df.columns and 'Longitude' not in df.columns:
            df['Longitude'] = pd.to_numeric(df['lon'], errors='coerce')
            
    return df

@st.cache_data(ttl=3600)
def get_predictions(spreadsheet_name="thermawatch-data"):
    """Mengambil data prediksi untuk dashboard frontend."""
    # Data prediksi sudah tergabung di sheet daily_data, 
    # jadi bisa kita ambil dari sana, atau return df utuhnya saja
    service = SheetsService()
    df = service.read_data(spreadsheet_name=spreadsheet_name, sheet_name="daily_data")
    if not df.empty:
        if 'prediksi_suhu_h1' in df.columns:
            df['pred_h1'] = pd.to_numeric(df['prediksi_suhu_h1'], errors='coerce')
        if 'prediksi_suhu_h3' in df.columns:
            df['pred_h3'] = pd.to_numeric(df['prediksi_suhu_h3'], errors='coerce')
        if 'prediksi_suhu_h7' in df.columns:
            df['pred_h7'] = pd.to_numeric(df['prediksi_suhu_h7'], errors='coerce')
    return df
