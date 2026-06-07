import os
import json
import pandas as pd

def generate_baselines():
    print("=== Membuat File Baselines untuk Predictor ===")
    
    # Lokasi dataset master
    master_dataset_path = 'Dataset_Master_ERA5_Ready_LSTM.csv'
    
    if not os.path.exists(master_dataset_path):
        print(f"[ERROR] File '{master_dataset_path}' tidak ditemukan.")
        return
        
    print(f"Membaca {master_dataset_path}...")
    df = pd.read_csv(master_dataset_path)
    
    # Pastikan nama kabupaten bersih (lowercase)
    def clean_kab(name):
        return str(name).lower().strip().replace('kabupaten ', '').replace('kota ', '')
    
    df['Kabupaten'] = df['Kabupaten'].apply(clean_kab)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    
    # 1. Ekstrak Elevasi rata-rata per Kabupaten
    print("Mengekstrak data elevasi...")
    elevasi_dict = df.groupby('Kabupaten')['Elevation_m'].mean().to_dict()
    
    # 2. Ekstrak Rata-rata Historis LST per Kabupaten per Bulan
    print("Mengekstrak rata-rata LST bulanan historis...")
    # Groupby Kabupaten dan Month, hitung rata-rata ERA5_LST_Mean
    grouped = df.groupby(['Kabupaten', 'month'])['ERA5_LST_Mean'].mean().reset_index()
    
    historical_means = {}
    for _, row in grouped.iterrows():
        kab = row['Kabupaten']
        m = str(int(row['month']))
        val = float(row['ERA5_LST_Mean'])
        
        if kab not in historical_means:
            historical_means[kab] = {}
        historical_means[kab][m] = val
        
    # Gabungkan menjadi satu objek data
    baseline_data = {
        "elevasi": elevasi_dict,
        "historical_means": historical_means
    }
    
    # Simpan ke config/baselines.json
    output_path = 'config/baselines.json'
    os.makedirs('config', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(baseline_data, f, indent=4)
        
    print(f"[OK] File baseline berhasil disimpan di '{output_path}'")
    print(f"Jumlah Kabupaten terdaftar: {len(elevasi_dict)}")

if __name__ == '__main__':
    generate_baselines()
