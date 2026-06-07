import ee
import json
from datetime import datetime, timedelta

# Inisialisasi GEE
try:
    with open("config/credentials.json", "r") as f:
        creds_data = json.load(f)
        project_id = creds_data.get("project_id")
    
    import google.oauth2.service_account
    creds = google.oauth2.service_account.Credentials.from_service_account_file(
        "config/credentials.json",
        scopes=["https://www.googleapis.com/auth/earthengine", "https://www.googleapis.com/auth/cloud-platform"]
    )
    ee.Initialize(credentials=creds, project=project_id)
    print("[GEE] Berhasil inisialisasi untuk pencarian tanggal.")
except Exception as e:
    print(f"[ERROR] Gagal inisialisasi GEE: {e}")
    exit(1)

# Wilayah Bandung sebagai titik uji
bandung = ee.FeatureCollection("FAO/GAUL/2015/level2") \
            .filter(ee.Filter.eq('ADM2_NAME', 'Bandung'))

def find_latest_valid_date():
    # Mulai dari H-1 dan mundur ke belakang
    current_check = datetime.now() - timedelta(days=1)
    
    print("\nMencari tanggal terbaru dengan data ERA5 LST valid...")
    for i in range(15): # Mundur maksimal 15 hari
        date_str = current_check.strftime('%Y-%m-%d')
        start_date = ee.Date(date_str)
        
        era5 = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY") \
                 .filterBounds(bandung) \
                 .filterDate(start_date, start_date.advance(1, 'day')) \
                 .select('skin_temperature')
                 
        if era5.size().getInfo() > 0:
            # Hitung rata-rata LST untuk Bandung
            mean_img = era5.mean().subtract(273.15)
            stats = mean_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=bandung.geometry(),
                scale=9000
            )
            
            # Ambil nilai suhunya
            val = stats.get('skin_temperature').getInfo()
            if val is not None:
                print(f"[OK] Tanggal valid terupdate ditemukan: {date_str} (Suhu Rata-rata Bandung: {val:.2f} °C)")
                return date_str
            else:
                print(f"[INFO] Tanggal {date_str} terdaftar di GEE tapi datanya masih kosong (Null).")
        else:
            print(f"[INFO] Tanggal {date_str} tidak memiliki citra di GEE.")
            
        current_check -= timedelta(days=1)
        
    return None

if __name__ == '__main__':
    find_latest_valid_date()
