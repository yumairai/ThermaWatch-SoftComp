import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ============================================================
# 1. ARSITEKTUR MODEL 
# ============================================================

class DifferentiableFuzzyLayer(nn.Module):
    def __init__(self, n_features, n_fuzzy_sets):
        super().__init__()
        self.n_features = n_features
        self.n_fuzzy_sets = n_fuzzy_sets

        # Parameter centers dan log_sigma yang diadaptasi model
        self.centers = nn.Parameter(torch.zeros(n_features, n_fuzzy_sets))
        self.log_sigma = nn.Parameter(torch.zeros(n_features, n_fuzzy_sets))

    def forward(self, x):
        # x: [B, T, F] -> [B, T, F, 1]
        x_exp = x.unsqueeze(-1)
        c = self.centers.unsqueeze(0).unsqueeze(0)
        sig = torch.exp(self.log_sigma).unsqueeze(0).unsqueeze(0)

        # Gaussian Membership Function
        mu = torch.exp(-((x_exp - c) ** 2) / (2 * sig ** 2 + 1e-8))
        B, T, F, S = mu.shape
        return mu.reshape(B, T, F * S)


class ANFIS_LSTM(nn.Module):
    def __init__(self, n_dynamic=3, n_static=3, n_fuzzy=5, lstm_hidden=256, 
                 lstm_layers=2, lstm_drop=0.3, env_embed=64, mlp_hidden=128):
        super().__init__()
        
        fuzzy_out = n_dynamic * n_fuzzy
        self.fuzzy_layer = DifferentiableFuzzyLayer(n_dynamic, n_fuzzy)
        
        self.lstm = nn.LSTM(
            input_size=fuzzy_out,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_drop if lstm_layers > 1 else 0.0
        )
        
        self.env_branch = nn.Sequential(
            nn.Linear(n_static, env_embed),
            nn.ReLU(),
            nn.Linear(env_embed, env_embed),
            nn.ReLU()
        )
        
        fusion_dim = lstm_hidden + env_embed
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        
        self.shared_mlp = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden * 2),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(mlp_hidden * 2, mlp_hidden),
            nn.GELU(),
            nn.Dropout(0.15)
        )
        
        self.head_h1 = nn.Linear(mlp_hidden, 1)
        self.head_h3 = nn.Linear(mlp_hidden, 1)
        self.head_h7 = nn.Linear(mlp_hidden, 1)

    def forward(self, x_dyn, x_stat):
        # x_dyn: [B, T, F_dyn], x_stat: [B, F_stat]
        fuzz = self.fuzzy_layer(x_dyn)
        _, (h_n, _) = self.lstm(fuzz)
        h_last = h_n[-1]
        
        env_emb = self.env_branch(x_stat)
        fused = torch.cat([h_last, env_emb], dim=-1)
        fused = self.fusion_norm(fused)
        
        base = self.shared_mlp(fused)
        return torch.cat([
            self.head_h1(base),
            self.head_h3(base),
            self.head_h7(base)
        ], dim=-1)


# ============================================================
# 2. MODEL INFERENCE SERVICE
# ============================================================

class ModelService:
    def __init__(self, model_path='outputs/best_model.pt', scaler_path='outputs/scalers.pkl'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        
        self.model = None
        self.scaler_dyn = None
        self.scaler_stat = None
        self.scaler_y = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model dan scaler saat inisialisasi
        self.load_artifacts()

    def load_artifacts(self):
        """Memuat file model .pt dan file scaler .pkl."""
        # 1. Load Scalers
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"File scaler tidak ditemukan di {self.scaler_path}. Pastikan sudah di-upload.")
            
        with open(self.scaler_path, 'rb') as f:
            scalers = pickle.load(f)
            self.scaler_dyn = scalers['scaler_dyn']
            self.scaler_stat = scalers['scaler_stat']
            self.scaler_y = scalers['scaler_y']
            
        # 2. Load Model Weights
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"File model tidak ditemukan di {self.model_path}. Pastikan sudah di-upload.")
            
        self.model = ANFIS_LSTM()
        checkpoint = torch.load(self.model_path, map_location=self.device)
        state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, df_window, df_static):
        """
        Menjalankan prediksi anomali geothermal h+1, h+3, h+7.
        
        Parameter:
        - df_window: DataFrame berisi 30 hari data historis berurutan dengan kolom:
                     ['ERA5_LST_Mean', 'LST_Mean', 'LST_Anomaly']
        - df_static: DataFrame/Series/Dict berisi fitur statis saat ini dengan kolom:
                     ['Elevation_m', 'NDVI_8Day_Mean', 'SoilMoisture_Daily_Mean']
        
        Return:
        - dict: Berisi estimasi nilai anomali dan estimasi suhu riil (°C) untuk H+1, H+3, H+7
        """
        # Pastikan input memiliki jumlah baris lookback yang sesuai (30 hari)
        if len(df_window) != 30:
            raise ValueError(f"Data historis dinamis harus berjumlah 30 baris (ditemukan {len(df_window)}).")
            
        # 1. Ekstrak data array
        dynamic_cols = ['ERA5_LST_Mean', 'LST_Mean', 'LST_Anomaly']
        static_cols = ['Elevation_m', 'NDVI_8Day_Mean', 'SoilMoisture_Daily_Mean']
        
        x_dyn_raw = df_window[dynamic_cols].values
        # Ambil baris terakhir data dinamis untuk fitur statis terasosiasi jika berupa DataFrame
        if isinstance(df_static, pd.DataFrame):
            x_stat_raw = df_static[static_cols].iloc[-1].values
        else:
            x_stat_raw = np.array([df_static[c] for c in static_cols])

        # 2. Skalakan data menggunakan StandardScaler resmi
        x_dyn_scaled = self.scaler_dyn.transform(x_dyn_raw).astype(np.float32)
        x_stat_scaled = self.scaler_stat.transform(x_stat_raw.reshape(1, -1)).astype(np.float32)

        # 3. Konversi ke PyTorch Tensor
        x_dyn_tensor = torch.tensor(x_dyn_scaled).unsqueeze(0).to(self.device)  # Shape [1, 30, 3]
        x_stat_tensor = torch.tensor(x_stat_scaled).to(self.device)             # Shape [1, 3]

        # 4. Forward Pass Model
        with torch.no_grad():
            out_scaled = self.model(x_dyn_tensor, x_stat_tensor).cpu().numpy()  # Shape [1, 3]

        # 5. Denormalisasi Hasil Prediksi
        out_raw = self.scaler_y.inverse_transform(out_scaled)[0]  # Nilai asli anomali (°C)

        # 6. Hitung estimasi suhu riil prediksi (Anomali Prediksi + Rerata Historis Bulanan)
        # Diambil dari kolom LST_Historical_Mean di data referensi (opsional, jika ingin ditampilkan)
        anomali_h1, anomali_h3, anomali_h7 = out_raw[0], out_raw[1], out_raw[2]

        return {
            "prediction": {
                "H1": {
                    "anomaly_temp": float(anomali_h1)
                },
                "H3": {
                    "anomaly_temp": float(anomali_h3)
                },
                "H7": {
                    "anomaly_temp": float(anomali_h7)
                }
            }
        }
