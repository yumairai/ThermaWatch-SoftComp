import pandas as pd
import numpy as np
import os

def main():
    input_path = 'data/Dataset_Master_Modis_Ready_LSTM.csv'
    output_path = 'data/Dataset_Master_Modis_Ready_LSTM_training.csv'
    
    if not os.path.exists(input_path):
        print(f"[ERROR] File asli '{input_path}' tidak ditemukan.")
        return
        
    print(f"1. Membaca data asli MODIS dari {input_path}...")
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['Kabupaten', 'date']).reset_index(drop=True)
    
    # Salin untuk data latih
    df_train = df.copy()
    
    # Parameter Augmentasi
    num_events = 200  # Jumlah event pemanasan buatan yang disuntikkan
    event_duration = 30  # Durasi total event (hari)
    
    print(f"2. Menyuntikkan {num_events} event anomali termal buatan untuk MODIS...")
    np.random.seed(42)  # Agar hasilnya konsisten
    
    kabupaten_list = df_train['Kabupaten'].unique()
    n_rows = len(df_train)
    
    events_injected = 0
    
    for _ in range(num_events):
        kab = np.random.choice(kabupaten_list)
        # Ambil semua indeks untuk kabupaten terpilih
        kab_indices = df_train[df_train['Kabupaten'] == kab].index.tolist()
        
        if len(kab_indices) < event_duration + 10:
            continue
            
        # Pilih indeks awal acak di dalam rentang data kabupaten tersebut
        start_idx_in_list = np.random.randint(0, len(kab_indices) - event_duration - 5)
        start_idx = kab_indices[start_idx_in_list]
        
        # Tentukan kekuatan anomali maksimum acak antara +3.0°C hingga +6.0°C (level WASPADA & BAHAYA)
        max_anomaly = np.random.uniform(3.0, 6.0)
        
        # Buat kurva pemanasan gradual (menggunakan sigmoid atau linier)
        # Hari 0-10: Naik gradual
        # Hari 11-20: Stabil tinggi
        # Hari 21-30: Turun kembali ke normal
        ramp_up = np.linspace(0.2, max_anomaly, 10)
        plateau = np.full(10, max_anomaly)
        ramp_down = np.linspace(max_anomaly, 0.2, 10)
        
        event_profile = np.concatenate([ramp_up, plateau, ramp_down])
        
        # Suntikkan profil anomali ke kolom LST_Anomaly
        for offset, val in enumerate(event_profile):
            target_row_idx = start_idx + offset
            
            # Update anomali
            df_train.at[target_row_idx, 'LST_Anomaly'] = val
            
            # Hitung ulang suhu riil (Suhu Riil = Rerata Historis + Anomali Baru)
            hist_mean = df_train.at[target_row_idx, 'LST_Historical_Mean']
            if pd.notna(hist_mean):
                new_temp = hist_mean + val
                df_train.at[target_row_idx, 'MODIS_LST_Mean'] = new_temp
                df_train.at[target_row_idx, 'LST_Mean'] = new_temp
                
                # Update juga LST_Max dan Percentile95 agar seimbang
                df_train.at[target_row_idx, 'MODIS_LST_Max'] = new_temp + 2.0
                df_train.at[target_row_idx, 'LST_Max'] = new_temp + 2.0
                df_train.at[target_row_idx, 'MODIS_LST_Percentile95'] = new_temp + 1.2
                df_train.at[target_row_idx, 'LST_Percentile95'] = new_temp + 1.2
                
        events_injected += 1
        
    print(f"   Selesai! Berhasil menyuntikkan {events_injected} event pemanasan pada MODIS.")
    
    # 3. Hitung Ulang Shift Target (H+1, H+3, H+7) secara tepat per Kabupaten
    print("3. Menyelaraskan ulang kolom target H+1, H+3, dan H+7...")
    df_train['Target_Anomali_H1'] = df_train.groupby('Kabupaten')['LST_Anomaly'].shift(-1)
    df_train['Target_Anomali_H3'] = df_train.groupby('Kabupaten')['LST_Anomaly'].shift(-3)
    df_train['Target_Anomali_H7'] = df_train.groupby('Kabupaten')['LST_Anomaly'].shift(-7)
    
    # Bersihkan sisa shift yang bernilai NaN di akhir data kabupaten
    df_train = df_train.dropna(subset=['Target_Anomali_H1', 'Target_Anomali_H3', 'Target_Anomali_H7'])
    
    # Ekspor ke file baru
    print(f"4. Menyimpan dataset latih MODIS baru ke '{output_path}'...")
    df_train.to_csv(output_path, index=False)
    
    print("\n=== PEMBUATAN DATASET LATIH AUGMENTASI MODIS SUKSES ===")
    print(f"Total Baris Data Latih MODIS: {len(df_train)}")
    print(f"Anomali Maksimum di Data Latih Baru: {df_train['LST_Anomaly'].max():.2f}°C")
    print(f"Jumlah data dengan potensi bahaya (anomali >= 3.0°C): {len(df_train[df_train['LST_Anomaly'] >= 3.0])} baris.")

if __name__ == '__main__':
    main()
