"""
ThermaWatch-SoftComp | Halaman 1: Dashboard Monitoring
Menampilkan kondisi suhu aktual, prediksi, peta, dan tren historis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pydeck as pdk
from datetime import datetime, timedelta

from services.sheets_service import get_daily_data, get_predictions



# ─── CSS Kustom ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Warna tema utama */
    :root {
        --color-danger: #ef4444;
        --color-warning: #f59e0b;
        --color-safe: #22c55e;
        --color-primary: #0ea5e9;
    }

    /* Card prediksi */
    .pred-card {
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
        background: rgba(15,23,42,0.6);
    }
    .pred-card .label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
    }
    .pred-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .pred-card .status {
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-aman   { color: #22c55e; }
    .status-waspada { color: #f59e0b; }
    .status-bahaya { color: #ef4444; }

    /* Hilangkan padding berlebih pada container metric */
    [data-testid="metric-container"] {
        background: rgba(15,23,42,0.5);
        border: 1px solid rgba(148,163,184,0.15);
        border-radius: 10px;
        padding: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FUNGSI HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def _status_badge(suhu: float) -> tuple[str, str]:
    """Mengembalikan (emoji_status, label, css_class) berdasarkan suhu."""
    if suhu < 35:
        return "🟢", "Aman", "status-aman"
    elif suhu < 40:
        return "🟡", "Waspada", "status-waspada"
    else:
        return "🔴", "Bahaya", "status-bahaya"


def _warna_suhu(suhu: float) -> list[int]:
    """Mengkonversi nilai suhu ke warna RGB untuk PyDeck."""
    # Normalisasi antara 25°C (biru) hingga 50°C (merah)
    t = max(0, min(1, (suhu - 25) / 25))
    r = int(255 * t)
    g = int(255 * (1 - abs(t - 0.5) * 2))
    b = int(255 * (1 - t))
    return [r, g, b, 180]


def render_header() -> None:
    """Menampilkan header utama halaman."""
    st.markdown("# 🌋 ThermaWatch Dashboard")
    st.markdown(
        "**Monitoring dan Prediksi Anomali Suhu Permukaan Tanah "
        "Berbasis AI dan Data Geospasial**"
    )
    st.divider()


def render_sidebar(df: pd.DataFrame) -> tuple:
    """
    Menampilkan sidebar filter dan mengembalikan nilai filter terpilih.
    Returns: (lokasi_terpilih, tanggal_mulai, tanggal_akhir, metrik_terpilih)
    """
    with st.sidebar:
        st.markdown("## ⚙️ Filter Data")

        # Filter Lokasi
        lokasi_list = ["Semua Lokasi"] + sorted(df["Kabupaten"].unique().tolist())
        lokasi_terpilih = st.selectbox("📍 Lokasi / Kabupaten", lokasi_list)

        st.markdown("---")

        # Filter Rentang Tanggal
        tanggal_col = pd.to_datetime(df["Tanggal"])
        tgl_min = tanggal_col.min().date()
        tgl_max = tanggal_col.max().date()

        tanggal_mulai = st.date_input("📅 Dari Tanggal", value=tgl_min, min_value=tgl_min, max_value=tgl_max)
        tanggal_akhir = st.date_input("📅 Sampai Tanggal", value=tgl_max, min_value=tgl_min, max_value=tgl_max)

        st.markdown("---")

        # Filter Metrik
        metrik_options = ["Suhu_Celcius", "Soil_Moisture", "NDVI", "Anomaly"]
        metrik_terpilih = st.multiselect(
            "📊 Metrik Grafik", metrik_options, default=["Suhu_Celcius"]
        )

        st.markdown("---")
        st.info("💡 Data diperbarui setiap hari dari Google Earth Engine via Google Sheets.")

    return lokasi_terpilih, tanggal_mulai, tanggal_akhir, metrik_terpilih


def filter_data(
    df: pd.DataFrame,
    lokasi: str,
    tgl_mulai,
    tgl_akhir,
) -> pd.DataFrame:
    """Memfilter DataFrame berdasarkan lokasi dan rentang tanggal."""
    df = df.copy()
    df["Tanggal"] = pd.to_datetime(df["Tanggal"])

    if lokasi != "Semua Lokasi":
        df = df[df["Kabupaten"] == lokasi]

    df = df[(df["Tanggal"].dt.date >= tgl_mulai) & (df["Tanggal"].dt.date <= tgl_akhir)]
    return df.sort_values("Tanggal")


def render_kpi(df: pd.DataFrame) -> None:
    """Menampilkan kartu KPI metrik utama."""
    st.markdown("### 📈 Metrik Utama")

    suhu = df["Suhu_Celcius"]
    idx_terpanas = df["Suhu_Celcius"].idxmax()
    lokasi_terpanas = df.loc[idx_terpanas, "Kabupaten"] if "Kabupaten" in df.columns else "N/A"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌡️ Suhu Rata-rata", f"{suhu.mean():.1f} °C")
    col2.metric("🔥 Suhu Maksimum", f"{suhu.max():.1f} °C")
    col3.metric("❄️ Suhu Minimum", f"{suhu.min():.1f} °C")
    col4.metric("📍 Lokasi Terpanas", lokasi_terpanas)


def render_prediction_cards(pred_df: pd.DataFrame) -> None:
    """Menampilkan kartu prediksi Hari Ini, H+2, H+6."""
    st.markdown("### 🤖 Prediksi Suhu (Dual-Branch ANFIS-LSTM)")

    horizons = {
        "Hari Ini": "pred_h1",
        "H+2": "pred_h3",
        "H+6": "pred_h7",
    }

    cols = st.columns(3)
    for col, (label, field) in zip(cols, horizons.items()):
        # Ambil nilai prediksi terakhir
        nilai = pred_df[field].iloc[-1] if field in pred_df.columns else 37.5

        emoji, status_txt, css_class = _status_badge(nilai)
        with col:
            st.markdown(f"""
            <div class="pred-card">
                <div class="label">{label}</div>
                <div class="value">{nilai:.1f} °C</div>
                <div class="status {css_class}">{emoji} {status_txt}</div>
            </div>
            """, unsafe_allow_html=True)


def render_map(df: pd.DataFrame) -> None:
    """Menampilkan peta interaktif menggunakan PyDeck."""
    st.markdown("### 🗺️ Peta Distribusi Suhu Permukaan")

    # Pastikan kolom yang dibutuhkan tersedia
    required_cols = {"Latitude", "Longitude", "Suhu_Celcius", "Kabupaten"}
    if not required_cols.issubset(df.columns):
        st.warning("⚠️ Kolom peta tidak lengkap. Pastikan data memiliki Latitude, Longitude, Suhu_Celcius, Kabupaten.")
        return

    # Ambil data terbaru per lokasi untuk peta
    df_map = df.sort_values("Tanggal").groupby("Kabupaten").last().reset_index()
    df_map["color"] = df_map["Suhu_Celcius"].apply(_warna_suhu)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position=["Longitude", "Latitude"],
        get_color="color",
        get_radius=8000,
        pickable=True,
        opacity=0.85,
        stroked=True,
        filled=True,
    )

    view_state = pdk.ViewState(
        latitude=df_map["Latitude"].mean(),
        longitude=df_map["Longitude"].mean(),
        zoom=8,
        pitch=30,
    )

    tooltip = {
        "html": "<b>📍 {Kabupaten}</b><br/>🌡️ Suhu: {Suhu_Celcius}°C",
        "style": {
            "backgroundColor": "#0f172a",
            "color": "#f1f5f9",
            "border": "1px solid #334155",
            "borderRadius": "8px",
            "padding": "8px 12px",
        },
    }

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/dark-v10",
    )
    st.pydeck_chart(deck)


def render_trend_chart(df: pd.DataFrame, metrik: list[str]) -> None:
    """Menampilkan grafik tren suhu menggunakan Plotly."""
    st.markdown("### 📉 Tren Suhu Permukaan Tanah")

    if df.empty or not metrik:
        st.info("Pilih minimal satu metrik dari sidebar untuk menampilkan grafik.")
        return

    fig = go.Figure()

    warna_metrik = {
        "Suhu_Celcius": "#f97316",
        "Soil_Moisture": "#38bdf8",
        "NDVI": "#4ade80",
        "Anomaly": "#f43f5e",
    }

    for m in metrik:
        if m not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df["Tanggal"],
            y=df[m],
            mode="lines+markers",
            name=m.replace("_", " "),
            line=dict(color=warna_metrik.get(m, "#94a3b8"), width=2),
            marker=dict(size=4),
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Tanggal",
        yaxis_title="Nilai",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.1)")

    st.plotly_chart(fig, use_container_width=True)


def render_raw_data(df: pd.DataFrame) -> None:
    """Menampilkan data mentah dalam expander."""
    with st.expander("🗄️ Lihat Data Mentah", expanded=False):
        st.markdown(f"**Total baris:** {len(df):,} | **Total kolom:** {len(df.columns)}")
        st.dataframe(
            df.reset_index(drop=True),
            use_container_width=True,
            height=300,
        )
        st.download_button(
            label="⬇️ Unduh CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="thermawatch_data.csv",
            mime="text/csv",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FUNGSI UTAMA
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Entry point halaman Dashboard."""
    render_header()

    # ── Ambil Data ──────────────────────────────────────────────────────────
    with st.spinner("⏳ Memuat data dari Google Sheets..."):
        try:
            df_raw = get_daily_data()
            df_pred = get_predictions()
        except Exception as e:
            st.error(f"❌ Gagal memuat data: {e}")
            st.stop()

    # ── Sidebar Filter ───────────────────────────────────────────────────────
    lokasi, tgl_mulai, tgl_akhir, metrik = render_sidebar(df_raw)

    # ── Filter Data ──────────────────────────────────────────────────────────
    df_filtered = filter_data(df_raw, lokasi, tgl_mulai, tgl_akhir)

    if df_filtered.empty:
        st.warning("⚠️ Tidak ada data untuk filter yang dipilih.")
        st.stop()

    # ── Render Komponen ──────────────────────────────────────────────────────
    render_kpi(df_filtered)
    st.markdown("")

    render_prediction_cards(df_pred)
    st.markdown("")

    col_map, col_trend = st.columns([1, 1], gap="medium")
    with col_map:
        render_map(df_filtered)
    with col_trend:
        render_trend_chart(df_filtered, metrik)

    render_raw_data(df_filtered)


if __name__ == "__main__" or True:
    main()
