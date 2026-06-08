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
