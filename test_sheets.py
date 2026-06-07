import json
from services.sheets_service import SheetsService

def test_connection():
    print("=== Menguji Koneksi Google Sheets ===")
    
    # Baca email client dari credentials untuk memandu user
    try:
        with open("config/credentials.json", "r") as f:
            creds_data = json.load(f)
            client_email = creds_data.get("client_email")
            print(f"Email Service Account Anda: {client_email}")
            print("--------------------------------------------------")
            print(f"PENTING: Pastikan Anda telah melakukan share spreadsheet Anda")
            print(f"ke email di atas dengan akses 'Editor'.")
            print("--------------------------------------------------")
    except Exception as e:
        print(f"Gagal membaca file credentials.json: {e}")
        return

    # Coba inisialisasi dan list spreadsheet
    try:
        service = SheetsService("config/credentials.json")
        print("[OK] Autentikasi Google Sheets API Berhasil!")
        
        spreadsheets = service.client.openall()
        if spreadsheets:
            print("\nSpreadsheet yang dapat diakses oleh Service Account ini:")
            for idx, s in enumerate(spreadsheets, 1):
                print(f"{idx}. {s.title} (ID: {s.id})")
        else:
            print("\n[INFO] Belum ada spreadsheet yang di-share ke email Service Account ini.")
            print("Silakan share spreadsheet Anda terlebih dahulu di Google Sheets web.")
            
    except Exception as e:
        print(f"\n[ERROR] Terjadi kesalahan saat menghubungkan:")
        print(e)

if __name__ == "__main__":
    test_connection()
