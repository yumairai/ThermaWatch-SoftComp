"""
ThermaWatch-SoftComp | Halaman 3: Environmental Analytics
Analisis mendalam terhadap faktor-faktor lingkungan yang mempengaruhi LST.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from services.sheets_service import get_daily_data


# ─── CSS Kustom ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stat-card {
        background: rgba(15,23,42,0.6);
        border: 1px solid rgba(148,163,184,0.15);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .stat-card .stat-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .stat-card .stat-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-top: 4px;
    }
    .stat-card .stat-unit {
        font-size: 0.8rem;
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)


# ─── Konstanta ────────────────────────────────────────────────────────────────
# Kolom yang digunakan dalam analisis
KOLOM_ANALISIS = ["Suhu_Celcius", "Soil_Moisture", "NDVI", "Elevation", "Anomaly"]

WARNA_VARIABEL = {
    "Suhu_Celcius": "#f97316",
    "Soil_Moisture": "#38bdf8",
    "NDVI": "#4ade80",
    "Elevation": "#a78bfa",
    "Anomaly": "#f43f5e",
}

def hex_to_rgba(hex_color, alpha=0.08):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ═══════════════════════════════════════════════════════════════════════════════
# FUNGSI HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def _kolom_tersedia(df: pd.DataFrame, kolom: list[str]) -> list[str]:
    """Mengembalikan hanya kolom yang benar-benar ada di DataFrame."""
    return [c for c in kolom if c in df.columns]


def render_header() -> None:
    """Menampilkan header halaman."""
    st.markdown("# 🌿 Environmental Analytics")
    st.markdown(
        "Analisis mendalam terhadap faktor-faktor lingkungan yang mempengaruhi "
        "**Land Surface Temperature (LST)** dan distribusi anomali suhu."
    )
    st.divider()


def render_sidebar_filter(df: pd.DataFrame) -> tuple:
    """Menampilkan filter sidebar dan mengembalikan data yang difilter."""
    with st.sidebar:
        st.markdown("## ⚙️ Filter Analisis")

        # Filter Kabupaten
        if "Kabupaten" in df.columns:
            lokasi_list = ["Semua Lokasi"] + sorted(df["Kabupaten"].unique().tolist())
            lokasi = st.selectbox("📍 Kabupaten", lokasi_list)
        else:
            lokasi = "Semua Lokasi"

        # Filter rentang tanggal
        if "Tanggal" in df.columns:
            df["Tanggal"] = pd.to_datetime(df["Tanggal"])
            tgl_min = df["Tanggal"].min().date()
            tgl_max = df["Tanggal"].max().date()
            tgl_mulai = st.date_input("📅 Dari", value=tgl_min, min_value=tgl_min, max_value=tgl_max)
            tgl_akhir = st.date_input("📅 Sampai", value=tgl_max, min_value=tgl_min, max_value=tgl_max)
        else:
            tgl_mulai = tgl_akhir = None

        st.markdown("---")
        st.markdown("### 🎯 Variabel Analisis")
        kolom_tersedia = _kolom_tersedia(df, KOLOM_ANALISIS)
        kolom_dipilih = st.multiselect(
            "Pilih variabel",
            options=kolom_tersedia,
            default=kolom_tersedia[:3],
        )

    # Aplikasikan filter
    df_filtered = df.copy()
    if lokasi != "Semua Lokasi" and "Kabupaten" in df.columns:
        df_filtered = df_filtered[df_filtered["Kabupaten"] == lokasi]
    if tgl_mulai and tgl_akhir and "Tanggal" in df.columns:
        df_filtered = df_filtered[
            (df_filtered["Tanggal"].dt.date >= tgl_mulai) &
            (df_filtered["Tanggal"].dt.date <= tgl_akhir)
        ]

    return df_filtered, kolom_dipilih


def render_summary_stats(df: pd.DataFrame, kolom: list[str]) -> None:
    """Menampilkan statistik ringkasan per variabel."""
    st.markdown("### 📊 Statistik Ringkasan")

    kolom_ada = _kolom_tersedia(df, kolom)
    if not kolom_ada:
        st.warning("Tidak ada kolom yang tersedia untuk analisis.")
        return

    stats = df[kolom_ada].agg(["mean", "median", "min", "max"]).T
    stats.columns = ["Mean", "Median", "Min", "Max"]

    # Tampilkan sebagai kartu per variabel
    for var in kolom_ada:
        row = stats.loc[var]
        col1, col2, col3, col4 = st.columns(4)
        unit = "°C" if var == "Suhu_Celcius" else ("m³/m³" if var == "Soil_Moisture" else "")
        label_bersih = var.replace("_", " ")

        with col1:
            st.markdown(f"""<div class="stat-card">
                <div class="stat-label">Mean — {label_bersih}</div>
                <div class="stat-value">{row['Mean']:.3f}</div>
                <div class="stat-unit">{unit}</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="stat-card">
                <div class="stat-label">Median — {label_bersih}</div>
                <div class="stat-value">{row['Median']:.3f}</div>
                <div class="stat-unit">{unit}</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="stat-card">
                <div class="stat-label">Min — {label_bersih}</div>
                <div class="stat-value">{row['Min']:.3f}</div>
                <div class="stat-unit">{unit}</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div class="stat-card">
                <div class="stat-label">Max — {label_bersih}</div>
                <div class="stat-value">{row['Max']:.3f}</div>
                <div class="stat-unit">{unit}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")


def render_time_series_charts(df: pd.DataFrame, kolom: list[str]) -> None:
    """Menampilkan grafik tren Soil Moisture, NDVI, dan LST dengan Plotly."""
    st.markdown("### 📉 Tren Variabel Lingkungan")

    if "Tanggal" not in df.columns:
        st.warning("Kolom 'Tanggal' tidak ditemukan.")
        return

    kolom_ada = _kolom_tersedia(df, kolom)
    if not kolom_ada:
        return

    # Buat subplot per variabel
    fig = make_subplots(
        rows=len(kolom_ada),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[c.replace("_", " ") for c in kolom_ada],
    )

    for i, var in enumerate(kolom_ada, start=1):
        warna = WARNA_VARIABEL.get(var, "#94a3b8")
        fig.add_trace(
            go.Scatter(
                x=df["Tanggal"],
                y=df[var],
                mode="lines",
                name=var.replace("_", " "),
                line=dict(color=warna, width=1.8),
                fill="tozeroy",
                fillcolor=hex_to_rgba(warna, 0.08),
            ),
            row=i,
            col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=200 * len(kolom_ada),
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.08)")

    st.plotly_chart(fig, use_container_width=True)


def render_correlation_heatmap(df: pd.DataFrame) -> None:
    """Menampilkan heatmap korelasi antar variabel lingkungan."""
    st.markdown("### 🔥 Heatmap Korelasi")

    kolom_korelasi = _kolom_tersedia(df, KOLOM_ANALISIS)
    if len(kolom_korelasi) < 2:
        st.info("Diperlukan minimal 2 variabel untuk heatmap korelasi.")
        return

    corr = df[kolom_korelasi].corr()
    label = [c.replace("_", " ") for c in kolom_korelasi]

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=label,
        y=label,
        colorscale=[
            [0.0, "#1e3a5f"],
            [0.5, "#0f172a"],
            [1.0, "#7c3aed"],
        ],
        zmin=-1,
        zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=12, color="#f1f5f9"),
        hoverongaps=False,
        showscale=True,
        colorbar=dict(title="Korelasi", tickfont=dict(color="#94a3b8")),
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11)),
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption("Korelasi berkisar dari -1 (berbanding terbalik) hingga +1 (berbanding lurus).")


def render_histograms(df: pd.DataFrame, kolom: list[str]) -> None:
    """Menampilkan histogram distribusi per variabel."""
    st.markdown("### 📊 Distribusi Variabel")

    kolom_ada = _kolom_tersedia(df, kolom)
    if not kolom_ada:
        return

    cols_per_row = 2
    for i in range(0, len(kolom_ada), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, var in enumerate(kolom_ada[i: i + cols_per_row]):
            warna = WARNA_VARIABEL.get(var, "#94a3b8")
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df[var].dropna(),
                nbinsx=30,
                marker_color=warna,
                opacity=0.8,
                name=var.replace("_", " "),
            ))
            # Garis rata-rata
            mean_val = df[var].mean()
            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color="#f1f5f9",
                annotation_text=f"μ={mean_val:.2f}",
                annotation_position="top right",
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title=dict(text=var.replace("_", " "), font=dict(size=13)),
                xaxis_title=var.replace("_", " "),
                yaxis_title="Frekuensi",
                height=280,
                margin=dict(l=10, r=10, t=40, b=10),
                showlegend=False,
            )
            fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.08)")
            fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.08)")
            with cols[j]:
                st.plotly_chart(fig, use_container_width=True)


def render_summary_table(df: pd.DataFrame, kolom: list[str]) -> None:
    """Menampilkan tabel ringkasan statistik deskriptif."""
    st.markdown("### 📋 Tabel Statistik Deskriptif")

    kolom_ada = _kolom_tersedia(df, kolom)
    if not kolom_ada:
        return

    desc = df[kolom_ada].describe().T
    desc.index = [c.replace("_", " ") for c in kolom_ada]
    desc = desc.round(4)

    st.dataframe(
        desc,
        use_container_width=True,
        height=250,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FUNGSI UTAMA
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Entry point halaman Environmental Analytics."""
    render_header()

    # 1. Tambahkan selektor model di sidebar
    model_pilihan = st.sidebar.selectbox(
        "🤖 Model AI Utama",
        options=["ERA5 ANFIS-LSTM", "MODIS ANFIS-LSTM"],
        index=0,
        help="Pilih data sensor utama untuk dianalisis."
    )
    
    sheet_name = "daily_data" if model_pilihan == "ERA5 ANFIS-LSTM" else "daily_data_modis"

    # 2. Ambil data secara dinamis
    with st.spinner(f"⏳ Memuat data {model_pilihan}..."):
        try:
            df_raw = get_daily_data(sheet_name=sheet_name)
        except Exception as e:
            st.error(f"❌ Gagal memuat data: {e}")
            st.stop()

    # ── Filter dari Sidebar ──────────────────────────────────────────────────
    df, kolom_dipilih = render_sidebar_filter(df_raw)

    if df.empty:
        st.warning("⚠️ Tidak ada data untuk filter yang dipilih.")
        st.stop()

    st.caption(f"📌 Menampilkan **{len(df):,} baris data** setelah filter diterapkan.")

    # ── Render Komponen ──────────────────────────────────────────────────────
    render_summary_stats(df, kolom_dipilih)
    st.markdown("")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📉 Tren Waktu",
        "🔥 Heatmap Korelasi",
        "📊 Distribusi",
        "📋 Tabel Statistik",
    ])

    with tab1:
        render_time_series_charts(df, kolom_dipilih)

    with tab2:
        render_correlation_heatmap(df)

    with tab3:
        render_histograms(df, kolom_dipilih)

    with tab4:
        render_summary_table(df, kolom_dipilih)


if __name__ == "__main__" or True:
    main()
