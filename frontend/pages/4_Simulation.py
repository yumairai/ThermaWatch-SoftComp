"""
ThermaWatch-SoftComp | Halaman 2: Simulasi & What-If Analysis
Pengguna memasukkan parameter lingkungan secara manual untuk simulasi prediksi.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from backend.services.model_service import ModelService

# ─── CSS Kustom ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .sim-card {
        background: rgba(15,23,42,0.6);
        border: 1px solid rgba(148,163,184,0.15);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 12px;
    }
    .param-summary-row {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid rgba(148,163,184,0.1);
        font-size: 0.88rem;
    }
    .param-label { color: #94a3b8; }
    .param-value { color: #f1f5f9; font-weight: 600; }
    .result-banner {
        background: linear-gradient(135deg, rgba(14,165,233,0.15), rgba(99,102,241,0.15));
        border: 1px solid rgba(14,165,233,0.3);
        border-radius: 12px;
        padding: 16px 24px;
        margin: 16px 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ─── Konstanta ────────────────────────────────────────────────────────────────
KABUPATEN_LIST = [
    "Bandung", "Sumedang", "Garut", "Tasikmalaya", "Ciamis",
    "Kuningan", "Cirebon", "Majalengka", "Subang", "Purwakarta",
    "Karawang", "Bekasi", "Bogor", "Sukabumi", "Cianjur",
]


# ═══════════════════════════════════════════════════════════════════════════════
# FUNGSI HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def _status_info(anomali: float) -> tuple[str, str, str]:
    """Mengembalikan (emoji, label, css_color) berdasarkan nilai anomali."""
    if anomali >= 2.5:
        return "🔴", "BAHAYA (Potensi Manifestasi Aktif)", "#ef4444"
    elif anomali >= 1.0:
        return "🟡", "WASPADA (Pantau Berkala)", "#f59e0b"
    elif anomali <= -2.0:
        return "❄️", "ANOMALI DINGIN (Suhu Ekstrem)", "#3b82f6"
    else:
        return "🟢", "AMAN (Fluktuasi Normal)", "#22c55e"


def render_header() -> None:
    """Menampilkan header halaman simulasi."""
    st.markdown("# 🧪 Simulasi & What-If Analysis")
    st.markdown(
        "Masukkan parameter lingkungan secara manual untuk mensimulasikan "
        "prediksi suhu permukaan tanah menggunakan model **Dual-Branch ANFIS-LSTM**."
    )
    st.divider()


def render_input_form() -> dict:
    """
    Menampilkan form input parameter lingkungan.
    Returns dict berisi semua nilai parameter.
    """
    st.markdown("### ⚙️ Parameter Lingkungan")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### 📍 Lokasi & Waktu")
        kabupaten = st.selectbox(
            "Kabupaten / Kota",
            options=KABUPATEN_LIST,
            help="Pilih lokasi simulasi.",
        )
        month = st.selectbox(
            "Bulan Simulasi",
            options=list(range(1, 13)),
            format_func=lambda m: [
                "Januari", "Februari", "Maret", "April", "Mei", "Juni",
                "Juli", "Agustus", "September", "Oktober", "November", "Desember"
            ][m - 1],
            index=6,
            help="Bulan yang disimulasikan (mempengaruhi pola musiman).",
        )

        st.markdown("#### 🌡️ Suhu & Kelembaban")
        lst_today = st.number_input(
            "Suhu Hari Ini (T) °C",
            min_value=20.0,
            max_value=60.0,
            value=35.0,
            step=0.5,
            format="%.1f",
            help="Suhu target / hari terakhir dalam window simulasi."
        )
        lst_history_str = st.text_area(
            "Suhu Historis (T-13 s/d T-1) °C",
            value="35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0",
            help="Masukkan tepat 13 nilai suhu dipisahkan dengan koma untuk historis sebelum hari ini."
        )
        try:
            history_list = [float(x.strip()) for x in lst_history_str.split(",")]
            lst_mean_list = history_list + [lst_today]
            if len(lst_mean_list) != 14:
                st.warning(f"Jumlah total LST (Historis + Hari Ini) tidak 14 (ditemukan {len(lst_mean_list)}). Prediksi mungkin gagal.")
        except ValueError:
            st.error("Input suhu historis tidak valid. Harap pastikan hanya berisi angka dan koma.")
            lst_mean_list = [35.0] * 14
        soil_moisture = st.slider(
            "Soil Moisture (%)",
            min_value=0.0,
            max_value=100.0,
            value=30.0,
            step=0.5,
            help="Kandungan air tanah volumetrik (persentase).",
        )

    with col2:
        st.markdown("#### 🌿 Vegetasi & Topografi")
        ndvi = st.slider(
            "NDVI (Indeks Vegetasi)",
            min_value=-1.0,
            max_value=1.0,
            value=0.4,
            step=0.01,
            help="Normalized Difference Vegetation Index. Nilai tinggi = vegetasi lebat.",
        )
        elevation = st.number_input(
            "Elevasi (mdpl)",
            min_value=0,
            max_value=3000,
            value=500,
            step=10,
            help="Ketinggian lokasi di atas permukaan laut.",
        )

        st.markdown("#### ℹ️ Panduan Nilai")
        st.markdown("""
        | Parameter | Rendah | Sedang | Tinggi |
        |-----------|--------|--------|--------|
        | LST Mean | < 30°C | 30–40°C | > 40°C |
        | Soil Moisture | < 0.2 | 0.2–0.5 | > 0.5 |
        | NDVI | < 0.2 | 0.2–0.6 | > 0.6 |
        """)

    params = {
        "kabupaten": kabupaten,
        "lst_mean_list": lst_mean_list,
        "soil_moisture": soil_moisture,
        "ndvi": ndvi,
        "elevation": elevation,
        "month": month,
    }
    return params


def run_prediction(params: dict, is_modis: bool = False) -> dict | None:
    try:
        if is_modis:
            service = ModelService(model_path='model/best_model_modis.pt', scaler_path='model/scalers_modis.pkl')
            data_filename = "Dataset_(Jan,2014-Mei,2026).csv"
            main_lst_col = "MODIS_LST_Mean"
        else:
            service = ModelService()
            data_filename = "Dataset_Master_ERA5_Ready_LSTM.csv"
            main_lst_col = "ERA5_LST_Mean"

        # Load data master ERA5/MODIS
        from pathlib import Path
        data_path = Path(__file__).parents[2] / "data" / data_filename
        df_master = pd.read_csv(data_path)
        
        # Filter by kabupaten and month
        df_filtered = df_master.copy()
        if "Kabupaten" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["Kabupaten"].str.lower() == params["kabupaten"].lower()]
        if "month" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["month"] == params["month"]]
        
        # Ensure date column exists and sorted
        if "date" in df_filtered.columns:
            df_filtered["date"] = pd.to_datetime(df_filtered["date"])
            df_filtered = df_filtered.sort_values("date")
            
        # Take last 14 days
        df_window = df_filtered.tail(14)[[main_lst_col, "LST_Mean", "LST_Anomaly"]]
        if df_window.empty:
            st.error("⚠️ Tidak ada data historis yang cocok untuk lokasi & bulan ini.")
            return None
        # Pad if fewer than 14 rows
        if len(df_window) < 14:
            last_row = df_window.iloc[-1]
            pad_rows = pd.DataFrame([last_row] * (14 - len(df_window)), columns=df_window.columns)
            df_window = pd.concat([df_window, pad_rows], ignore_index=True)

        # Cari rata-rata historis bulanan
        import json
        baselines_path = Path(__file__).parents[2] / "backend" / "config" / "baselines.json"
        hist_mean = 30.0  # Fallback default
        if baselines_path.exists():
            with open(baselines_path, "r") as f:
                baselines = json.load(f)
                hist_means = baselines.get("historical_means", {})
                kab_means = hist_means.get(params["kabupaten"].lower(), {})
                hist_mean = kab_means.get(str(params["month"]), 30.0)

        # Modifikasi nilai LST di df_window menggunakan input manual user (lst_mean_list)
        lst_mean_list = params.get("lst_mean_list", [])
        if len(lst_mean_list) == 14:
            df_window[main_lst_col] = lst_mean_list
            df_window["LST_Mean"] = lst_mean_list
            df_window["LST_Anomaly"] = np.array(lst_mean_list) - hist_mean

        # Data statik tetap dari input pengguna
        df_static = pd.DataFrame({
            "Elevation_m": [params["elevation"]],
            "NDVI_8Day_Mean": [params["ndvi"]],
            "SoilMoisture_Daily_Mean": [params["soil_moisture"]],
        })

        raw_result = service.predict(
            df_window=df_window,
            df_static=df_static
        )

        # Prediksi suhu = anomali + suhu historis bulanan
        last_lst = lst_mean_list[-1] if len(lst_mean_list) > 0 else 30.0

        return {
            "h1_temp": raw_result["prediction"]["H1"]["anomaly_temp"] + hist_mean,
            "h3_temp": raw_result["prediction"]["H3"]["anomaly_temp"] + hist_mean,
            "h7_temp": raw_result["prediction"]["H7"]["anomaly_temp"] + hist_mean,
            "h1_anom": raw_result["prediction"]["H1"]["anomaly_temp"],
            "h3_anom": raw_result["prediction"]["H3"]["anomaly_temp"],
            "h7_anom": raw_result["prediction"]["H7"]["anomaly_temp"],
        }

    except Exception as e:
        st.error(f"❌ Prediksi gagal: {e}")
        return None


def render_result_cards(result: dict) -> None:
    """Menampilkan kartu hasil prediksi Hari Ini, H+2, H+6."""
    st.markdown("### 📊 Hasil Prediksi")

    horizons = [
        ("Hari Ini", result.get("h1_temp", 0.0), result.get("h1_anom", 0.0), "Prediksi suhu untuk hari ini"),
        ("H+2", result.get("h3_temp", 0.0), result.get("h3_anom", 0.0), "Prediksi suhu 2 hari ke depan"),
        ("H+6", result.get("h7_temp", 0.0), result.get("h7_anom", 0.0), "Prediksi suhu 6 hari ke depan"),
    ]

    cols = st.columns(3)
    for col, (label, nilai_temp, nilai_anom, keterangan) in zip(cols, horizons):
        emoji, status_txt, warna = _status_info(nilai_anom)
        col.metric(
            label=f"{label} — {keterangan}",
            value=f"{nilai_temp:.2f} °C",
            delta=f"{emoji} {status_txt}",
        )


def render_prediction_chart(result: dict) -> None:
    """Menampilkan grafik proyeksi prediksi dengan Plotly."""
    st.markdown("### 📈 Grafik Proyeksi Prediksi")

    horizons = ["Hari Ini", "H+2", "H+6"]
    values = [result.get("h1_temp", 0.0), result.get("h3_temp", 0.0), result.get("h7_temp", 0.0)]
    anomalies = [result.get("h1_anom", 0.0), result.get("h3_anom", 0.0), result.get("h7_anom", 0.0)]

    # Warna titik berdasarkan status anomali
    colors = []
    for anom in anomalies:
        _, _, warna = _status_info(anom)
        colors.append(warna)

    fig = go.Figure()

    # Area di bawah grafik
    fig.add_trace(go.Scatter(
        x=horizons,
        y=values,
        mode="lines+markers+text",
        line=dict(color="#0ea5e9", width=3),
        marker=dict(size=14, color=colors, line=dict(width=2, color="#0f172a")),
        text=[f"{v:.1f}°C<br>({a:+.1f})" for v, a in zip(values, anomalies)],
        textposition="top center",
        textfont=dict(size=13, color="#f1f5f9"),
        fill="tozeroy",
        fillcolor="rgba(14,165,233,0.1)",
        name="Prediksi Suhu",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Horizon Prediksi",
        yaxis_title="Suhu (°C)",
        height=380,
        margin=dict(l=10, r=80, t=20, b=10),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.1)")

    st.plotly_chart(fig, use_container_width=True)


def render_input_summary(params: dict) -> None:
    """Menampilkan ringkasan parameter input yang digunakan."""
    with st.expander("📋 Ringkasan Parameter Input", expanded=True):
        nama_bulan = [
            "Januari", "Februari", "Maret", "April", "Mei", "Juni",
            "Juli", "Agustus", "September", "Oktober", "November", "Desember"
        ][params["month"] - 1]

        rows = [
            ("Kabupaten / Kota", params["kabupaten"]),
            ("LST Mean (14 Hari)", f"{params.get('lst_mean_list', [])}"),
            ("Soil Moisture", f"{params['soil_moisture']:.1f} %"),
            ("NDVI", f"{params['ndvi']:.2f}"),
            ("Elevasi", f"{params['elevation']} mdpl"),
            ("Bulan Simulasi", nama_bulan),
        ]

        for label, nilai in rows:
            st.markdown(f"""
            <div class="param-summary-row">
                <span class="param-label">{label}</span>
                <span class="param-value">{nilai}</span>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FUNGSI UTAMA
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Entry point halaman Simulasi."""
    render_header()

    # 1. Tambahkan selektor model di sidebar
    model_pilihan = st.sidebar.selectbox(
        "🤖 Model AI Untuk Simulasi",
        options=["ERA5 ANFIS-LSTM", "MODIS ANFIS-LSTM"],
        index=0,
        help="Pilih model dasar sensor yang ingin disimulasikan."
    )
    is_modis = model_pilihan == "MODIS ANFIS-LSTM"

    # 2. Form Input
    params = render_input_form()

    st.markdown("")
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        run_btn = st.button(
            "🚀 Run Prediction",
            type="primary",
            use_container_width=True,
        )

    st.divider()

    # 3. Eksekusi Prediksi dengan mengirimkan parameter is_modis
    if run_btn:
        with st.spinner("🔄 Model sedang memproses prediksi..."):
            result = run_prediction(params, is_modis=is_modis)

        if result is not None:
            st.markdown("""
            <div class="result-banner">
                ✅ <strong>Prediksi berhasil dihasilkan oleh Dual-Branch ANFIS-LSTM</strong>
            </div>
            """, unsafe_allow_html=True)

            render_result_cards(result)
            render_prediction_chart(result)
            render_input_summary(params)

    else:
        st.info("💡 Atur parameter lingkungan di atas, lalu tekan **Run Prediction** untuk memulai simulasi.")


if __name__ == "__main__" or True:
    main()
