from pipeline.gee_extractor import GEEExtractor

def test_gee_connection():
    print("=== Menguji Koneksi Google Earth Engine (GEE) ===")
    try:
        extractor = GEEExtractor("config/credentials.json")
        print("[OK] Inisialisasi GEE Berhasil!")
        
        # Coba ekstrak data ERA5 untuk tanggal 1 Januari 2025 sebagai tes
        test_date = "2025-01-01"
        print(f"\nMencoba mengambil data ERA5 LST untuk tanggal {test_date}...")
        df_era5 = extractor.extract_era5(test_date)
        
        if not df_era5.empty:
            print("[OK] Ekstraksi Data GEE Berhasil!")
            print(f"Jumlah Baris Kabupaten Terambil: {len(df_era5)}")
            print("\nSampel Data Hasil Ekstraksi (5 Baris Pertama):")
            print(df_era5[['Kabupaten', 'ERA5_LST_Mean', 'ERA5_LST_Max']].head())
        else:
            print("[WARN] Data kosong atau tidak ditemukan untuk tanggal tersebut.")
            
    except Exception as e:
        print("\n[ERROR] Gagal menyambungkan atau mengekstrak data dari GEE:")
        print(e)
        print("\nCatatan: Pastikan email Service Account Anda sudah didaftarkan/diberikan hak akses ke Google Earth Engine.")

if __name__ == "__main__":
    test_gee_connection()
