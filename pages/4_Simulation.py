"""
ThermaWatch-SoftComp | Halaman 2: Simulasi & What-If Analysis
Pengguna memasukkan parameter lingkungan secara manual untuk simulasi prediksi.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from services.model_service import predict_future

# ─── Konfigurasi Halaman ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="ThermaWatch | Simulasi",
    page_icon="🧪",
    layout="wide",
)

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

def _status_info(suhu: float) -> tuple[str, str, str]:
    """Mengembalikan (emoji, label, css_color) berdasarkan nilai suhu."""
    if suhu < 35:
        return "🟢", "Aman", "#22c55e"
    elif suhu < 40:
        return "🟡", "Waspada", "#f59e0b"
    else:
        return "🔴", "Bahaya", "#ef4444"


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
        lst_mean = st.number_input(
            "LST Mean (°C)",
            min_value=20.0,
            max_value=60.0,
            value=35.0,
            step=0.5,
            format="%.1f",
            help="Rata-rata Land Surface Temperature hasil observasi.",
        )
        soil_moisture = st.slider(
            "Soil Moisture (m³/m³)",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.01,
            help="Kandungan air tanah volumetrik.",
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
        "lst_mean": lst_mean,
        "soil_moisture": soil_moisture,
        "ndvi": ndvi,
        "elevation": elevation,
        "month": month,
    }
    return params


def run_prediction(params: dict) -> dict | None:
    """
    Memanggil predict_future dan mengembalikan hasil prediksi.
    Returns dict {h1, h3, h7} atau None jika gagal.
    """
    try:
        result = predict_future(
            kabupaten=params["kabupaten"],
            lst_mean=params["lst_mean"],
            soil_moisture=params["soil_moisture"],
            ndvi=params["ndvi"],
            elevation=params["elevation"],
            month=params["month"],
        )
        return result
    except Exception as e:
        st.error(f"❌ Prediksi gagal: {e}")
        return None


def render_result_cards(result: dict) -> None:
    """Menampilkan kartu hasil prediksi H+1, H+3, H+7."""
    st.markdown("### 📊 Hasil Prediksi")

    horizons = [
        ("H+1", result.get("h1", 0.0), "Prediksi 1 hari ke depan"),
        ("H+3", result.get("h3", 0.0), "Prediksi 3 hari ke depan"),
        ("H+7", result.get("h7", 0.0), "Prediksi 7 hari ke depan"),
    ]

    cols = st.columns(3)
    for col, (label, nilai, keterangan) in zip(cols, horizons):
        emoji, status_txt, warna = _status_info(nilai)
        col.metric(
            label=f"{label} — {keterangan}",
            value=f"{nilai:.2f} °C",
            delta=f"{emoji} {status_txt}",
        )


def render_prediction_chart(result: dict) -> None:
    """Menampilkan grafik proyeksi prediksi dengan Plotly."""
    st.markdown("### 📈 Grafik Proyeksi Prediksi")

    horizons = ["H+1", "H+3", "H+7"]
    values = [result.get("h1", 0.0), result.get("h3", 0.0), result.get("h7", 0.0)]

    # Warna titik berdasarkan status
    colors = []
    for v in values:
        _, _, warna = _status_info(v)
        colors.append(warna)

    fig = go.Figure()

    # Area di bawah grafik
    fig.add_trace(go.Scatter(
        x=horizons,
        y=values,
        mode="lines+markers+text",
        line=dict(color="#0ea5e9", width=3),
        marker=dict(size=14, color=colors, line=dict(width=2, color="#0f172a")),
        text=[f"{v:.1f}°C" for v in values],
        textposition="top center",
        textfont=dict(size=13, color="#f1f5f9"),
        fill="tozeroy",
        fillcolor="rgba(14,165,233,0.1)",
        name="Prediksi Suhu",
    ))

    # Garis ambang batas
    fig.add_hline(y=35, line_dash="dash", line_color="#f59e0b",
                  annotation_text="🟡 Waspada (35°C)", annotation_position="right")
    fig.add_hline(y=40, line_dash="dash", line_color="#ef4444",
                  annotation_text="🔴 Bahaya (40°C)", annotation_position="right")

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
            ("LST Mean", f"{params['lst_mean']:.1f} °C"),
            ("Soil Moisture", f"{params['soil_moisture']:.2f} m³/m³"),
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

    # ── Form Input ───────────────────────────────────────────────────────────
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

    # ── Eksekusi Prediksi ────────────────────────────────────────────────────
    if run_btn:
        with st.spinner("🔄 Model sedang memproses prediksi..."):
            result = run_prediction(params)

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
