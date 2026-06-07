import pandas as pd
from pipeline.predictor import PredictorPipeline

def test_predictor():
    print("=== Menguji Pipeline Prediksi ANFIS-LSTM ===")
    
    # Inisialisasi pipeline predictor
    try:
        predictor = PredictorPipeline(
            model_path="best_model.pt",
            scaler_path="scalers.pkl",
            baselines_path="config/baselines.json"
        )
        print("[OK] Inisialisasi PredictorPipeline Berhasil!")
    except Exception as e:
        print("[ERROR] Gagal menginisialisasi PredictorPipeline:")
        print(e)
        return

    # Ambil sampel 30 hari terakhir dari salah satu kabupaten di dataset master
    try:
        master_path = 'Dataset_Master_ERA5_Ready_LSTM.csv'
        print(f"\nMembaca dataset '{master_path}' untuk simulasi...")
        df = pd.read_csv(master_path)
        
        # Standarkan nama kabupaten
        df['Kabupaten'] = df['Kabupaten'].str.lower().str.strip().str.replace('kabupaten ', '').str.replace('kota ', '')
        
        # Filter Kabupaten Garut
        target_kab = "garut"
        df_garut = df[df['Kabupaten'] == target_kab].sort_values(by='date')
        
        if len(df_garut) < 30:
            print(f"[ERROR] Data untuk {target_kab} kurang dari 30 hari (hanya ada {len(df_garut)} baris).")
            return
            
        df_test_window = df_garut.tail(30)
        print(f"[OK] Berhasil mengambil data 30 hari terakhir untuk kabupaten '{target_kab}'.")
        print(f"Rentang tanggal simulasi: {df_test_window['date'].min()} s/d {df_test_window['date'].max()}")
        
        # Jalankan prediksi
        print("\nMenjalankan prediksi model ANFIS-LSTM...")
        results = predictor.predict_kabupaten(df_test_window)
        
        print("\n=== HASIL PREDIKSI ===")
        print(f"Tanggal Prediksi: {results['date']}")
        print(f"Wilayah: {results['kabupaten'].upper()}")
        print(f"Hasil Prediksi Anomali:")
        print(f"  - H+1: {results['prediction']['H1']['anomaly_temp']:.4f} °C")
        print(f"  - H+3: {results['prediction']['H3']['anomaly_temp']:.4f} °C")
        print(f"  - H+7: {results['prediction']['H7']['anomaly_temp']:.4f} °C")
        
    except Exception as e:
        print("[ERROR] Terjadi kesalahan saat simulasi prediksi:")
        print(e)

if __name__ == "__main__":
    test_predictor()
