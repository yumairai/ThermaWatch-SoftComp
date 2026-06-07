from services.sheets_service import SheetsService

def insert_headers():
    print("=== Menyisipkan Header Kolom ke Google Sheets ===")
    
    # Skema kolom sesuai rancangan Anda
    headers = [
        "date", "kabupaten", "lst_mean", "modis_lst_mean", "soil_moisture", 
        "elevation", "ndvi", "month", "anomaly", "h1", "h3", "h7", 
        "status_h1", "status_h3", "status_h7", "lat", "lon"
    ]
    
    try:
        service = SheetsService("config/credentials.json")
        sheet = service.get_worksheet("thermawatch-data", "daily_data")
        
        # Periksa apakah baris pertama kosong atau belum diisi header
        first_row = sheet.row_values(1)
        
        if not first_row or first_row[0] != "date":
            # Sisipkan baris header di baris ke-1
            sheet.insert_row(headers, index=1)
            print("[OK] Header kolom berhasil disisipkan di baris pertama!")
        else:
            print("[INFO] Header sudah ada di baris pertama.")
            
    except Exception as e:
        print(f"[ERROR] Gagal menulis header: {e}")

if __name__ == "__main__":
    insert_headers()
