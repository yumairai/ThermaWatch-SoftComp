"""
ThermaWatch-SoftComp | Halaman 4: Model Information
Menjelaskan arsitektur, performa, dan dataset model Dual-Branch ANFIS-LSTM.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Konfigurasi Halaman ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="ThermaWatch | Model Information",
    page_icon="🤖",
    layout="wide",
)

# ─── CSS Kustom ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Card arsitektur */
    .arch-card {
        background: rgba(15,23,42,0.7);
        border: 1px solid rgba(148,163,184,0.2);
        border-radius: 12px;
        padding: 20px;
        height: 100%;
    }
    .arch-card .arch-title {
        font-size: 1rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .arch-card .arch-desc {
        font-size: 0.86rem;
        color: #94a3b8;
        line-height: 1.6;
    }
    .arch-card ul {
        margin: 8px 0 0 0;
        padding-left: 16px;
        color: #cbd5e1;
        font-size: 0.84rem;
    }

    /* Badge metrik */
    .metric-badge {
        background: rgba(14,165,233,0.12);
        border: 1px solid rgba(14,165,233,0.3);
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-badge .mb-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .metric-badge .mb-value {
        font-size: 2rem;
        font-weight: 800;
        color: #38bdf8;
    }
    .metric-badge .mb-desc {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 2px;
    }

    /* Workflow step */
    .workflow-step {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 14px 18px;
        background: rgba(15,23,42,0.5);
        border-left: 3px solid #0ea5e9;
        border-radius: 0 8px 8px 0;
        margin-bottom: 8px;
    }
    .workflow-step .step-icon { font-size: 1.5rem; }
    .workflow-step .step-text .step-title {
        font-weight: 700;
        color: #f1f5f9;
        font-size: 0.95rem;
    }
    .workflow-step .step-text .step-sub {
        color: #64748b;
        font-size: 0.8rem;
    }
    .workflow-arrow {
        text-align: center;
        color: #334155;
        font-size: 1.3rem;
        margin: 2px 0;
        padding-left: 28px;
    }

    /* Dataset info */
    .ds-item {
        display: flex;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px solid rgba(148,163,184,0.1);
        font-size: 0.9rem;
    }
    .ds-label { color: #94a3b8; }
    .ds-value { font-weight: 600; color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)


# ─── Data Dummy Metrik Performa ───────────────────────────────────────────────
PERFORMANCE_METRICS = {
    "RMSE": {"nilai": "1.42", "satuan": "°C", "keterangan": "Root Mean Squared Error"},
    "MAE": {"nilai": "1.08", "satuan": "°C", "keterangan": "Mean Absolute Error"},
    "R²": {"nilai": "0.947", "satuan": "", "keterangan": "Coefficient of Determination"},
    "MAPE": {"nilai": "3.21", "satuan": "%", "keterangan": "Mean Absolute Percentage Error"},
}

DATASET_INFO = {
    "Jumlah Data": "12.480 baris",
    "Jumlah Fitur": "18 fitur input",
    "Rentang Waktu": "Januari 2021 – Desember 2023",
    "Jumlah Lokasi": "24 kabupaten/kota",
    "Resolusi Temporal": "Harian",
    "Resolusi Spasial": "~1 km × 1 km",
    "Sumber Data": "Google Earth Engine (MODIS, SMAP, NDVI)",
    "Train / Val / Test": "70% / 15% / 15%",
}


# ═══════════════════════════════════════════════════════════════════════════════
# FUNGSI HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def render_header() -> None:
    """Menampilkan header halaman."""
    st.markdown("# 🤖 Model Information")
    st.markdown(
        "Dokumentasi teknis lengkap mengenai model **Dual-Branch ANFIS-LSTM** "
        "yang digunakan untuk prediksi anomali suhu permukaan tanah (LST)."
    )
    st.divider()


def render_model_overview() -> None:
    """Menampilkan informasi umum model dan penjelasan setiap komponen."""
    st.markdown("### 🧠 Arsitektur Model: Dual-Branch ANFIS-LSTM")
    st.markdown(
        "Model ini menggabungkan dua pendekatan komputasi cerdas: "
        "**LSTM** untuk pemodelan deret waktu dan **ANFIS** untuk penalaran berbasis aturan fuzzy, "
        "yang kemudian digabung pada *Fusion Layer* untuk menghasilkan prediksi suhu multi-horizon."
    )

    st.markdown("")

    col1, col2, col3, col4 = st.columns(4, gap="small")

    komponen = [
        {
            "icon": "🔷",
            "judul": "Branch A — LSTM",
            "subjudul": "Time Series Modeling",
            "deskripsi": "Memproses data deret waktu LST historis untuk menangkap pola temporal, tren musiman, dan autokorelasi suhu.",
            "detail": [
                "3 layer LSTM bertumpuk",
                "Hidden size: 128 unit",
                "Dropout: 0.2",
                "Input window: 30 hari",
                "Bidirectional LSTM",
            ],
        },
        {
            "icon": "🔶",
            "judul": "Branch B — ANFIS",
            "subjudul": "Environmental Embedding",
            "deskripsi": "Memodelkan hubungan non-linear antara variabel lingkungan (NDVI, Soil Moisture, Elevasi) menggunakan logika fuzzy adaptif.",
            "detail": [
                "5 membership functions",
                "Fuzzy rules: 32",
                "Input: 5 env features",
                "Takagi-Sugeno FIS",
                "Adaptive parameter tuning",
            ],
        },
        {
            "icon": "🔗",
            "judul": "Fusion Layer",
            "subjudul": "Branch Merging",
            "deskripsi": "Menggabungkan representasi dari Branch A dan Branch B menggunakan mekanisme attention untuk pembobotan kontekstual.",
            "detail": [
                "Concatenation + Attention",
                "Multi-head: 4 heads",
                "Dense 256 → 128 unit",
                "BatchNorm + ReLU",
                "Dropout: 0.15",
            ],
        },
        {
            "icon": "🎯",
            "judul": "Output Layer",
            "subjudul": "Multi-Horizon Prediction",
            "deskripsi": "Menghasilkan prediksi suhu untuk tiga horizon waktu secara simultan: H+1, H+3, dan H+7 hari ke depan.",
            "detail": [
                "Dense 128 → 64 unit",
                "Output nodes: 3",
                "Activation: Linear",
                "Loss: Huber Loss",
                "Optimizer: AdamW",
            ],
        },
    ]

    for col, komp in zip([col1, col2, col3, col4], komponen):
        with col:
            items_html = "".join(f"<li>{d}</li>" for d in komp["detail"])
            st.markdown(f"""
            <div class="arch-card">
                <div class="arch-title">{komp['icon']} {komp['judul']}</div>
                <div style="font-size:0.75rem;color:#64748b;margin-bottom:8px;">{komp['subjudul']}</div>
                <div class="arch-desc">{komp['deskripsi']}</div>
                <ul>{items_html}</ul>
            </div>
            """, unsafe_allow_html=True)


def render_architecture_diagram() -> None:
    """Menampilkan diagram arsitektur model menggunakan Plotly."""
    st.markdown("### 🗺️ Diagram Alur Arsitektur")

    fig = go.Figure()

    # Definisi node diagram
    nodes = [
        # Branch A (kiri)
        dict(x=0.15, y=0.85, label="Input LST\nHistoris", color="#1e3a5f", border="#38bdf8"),
        dict(x=0.15, y=0.65, label="LSTM\nLayer 1", color="#1e40af", border="#60a5fa"),
        dict(x=0.15, y=0.45, label="LSTM\nLayer 2-3", color="#1e40af", border="#60a5fa"),

        # Branch B (kanan)
        dict(x=0.85, y=0.85, label="Input Env\nFeatures", color="#1c3a2c", border="#4ade80"),
        dict(x=0.85, y=0.65, label="Fuzzification\nLayer", color="#166534", border="#4ade80"),
        dict(x=0.85, y=0.45, label="ANFIS\nFuzzy Rules", color="#166534", border="#4ade80"),

        # Fusion
        dict(x=0.50, y=0.25, label="Fusion Layer\n(Attention)", color="#3b0764", border="#a78bfa"),

        # Output
        dict(x=0.28, y=0.05, label="H+1\nPrediction", color="#7c2d12", border="#f97316"),
        dict(x=0.50, y=0.05, label="H+3\nPrediction", color="#7c2d12", border="#f97316"),
        dict(x=0.72, y=0.05, label="H+7\nPrediction", color="#7c2d12", border="#f97316"),
    ]

    # Gambar node sebagai scatter
    for node in nodes:
        fig.add_trace(go.Scatter(
            x=[node["x"]], y=[node["y"]],
            mode="markers+text",
            marker=dict(
                size=55,
                color=node["color"],
                line=dict(color=node["border"], width=2),
                symbol="square",
            ),
            text=node["label"],
            textposition="middle center",
            textfont=dict(size=9, color="#e2e8f0"),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Panah koneksi Branch A
    connections = [
        (0.15, 0.85, 0.15, 0.65),  # Input → LSTM 1
        (0.15, 0.65, 0.15, 0.45),  # LSTM 1 → LSTM 2-3
        (0.15, 0.45, 0.50, 0.25),  # LSTM → Fusion

        (0.85, 0.85, 0.85, 0.65),  # Input → Fuzzification
        (0.85, 0.65, 0.85, 0.45),  # Fuzzification → ANFIS
        (0.85, 0.45, 0.50, 0.25),  # ANFIS → Fusion

        (0.50, 0.25, 0.28, 0.05),  # Fusion → H+1
        (0.50, 0.25, 0.50, 0.05),  # Fusion → H+3
        (0.50, 0.25, 0.72, 0.05),  # Fusion → H+7
    ]

    for x0, y0, x1, y1 in connections:
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="paper", yref="paper",
            axref="paper", ayref="paper",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=1.5,
            arrowcolor="#475569",
        )

    # Label branch
    fig.add_annotation(x=0.15, y=0.97, text="<b>Branch A — LSTM</b>",
                        showarrow=False, font=dict(size=11, color="#38bdf8"),
                        xref="paper", yref="paper")
    fig.add_annotation(x=0.85, y=0.97, text="<b>Branch B — ANFIS</b>",
                        showarrow=False, font=dict(size=11, color="#4ade80"),
                        xref="paper", yref="paper")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(visible=False, range=[-0.05, 1.05]),
        yaxis=dict(visible=False, range=[-0.05, 1.05]),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_performance_metrics() -> None:
    """Menampilkan badge metrik performa model."""
    st.markdown("### 📐 Metrik Performa Model")

    cols = st.columns(4)
    for col, (nama, info) in zip(cols, PERFORMANCE_METRICS.items()):
        with col:
            st.markdown(f"""
            <div class="metric-badge">
                <div class="mb-label">{nama}</div>
                <div class="mb-value">{info['nilai']}{info['satuan']}</div>
                <div class="mb-desc">{info['keterangan']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # Grafik perbandingan RMSE per horizon
    horizons = ["H+1", "H+3", "H+7"]
    rmse_train = [0.98, 1.14, 1.42]
    rmse_val = [1.05, 1.28, 1.57]
    rmse_test = [1.08, 1.31, 1.62]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Train", x=horizons, y=rmse_train, marker_color="#38bdf8"))
    fig.add_trace(go.Bar(name="Validasi", x=horizons, y=rmse_val, marker_color="#f59e0b"))
    fig.add_trace(go.Bar(name="Test", x=horizons, y=rmse_test, marker_color="#f97316"))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="group",
        title="RMSE per Horizon Prediksi (°C)",
        xaxis_title="Horizon",
        yaxis_title="RMSE (°C)",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_dataset_info() -> None:
    """Menampilkan informasi dataset yang digunakan."""
    st.markdown("### 🗂️ Informasi Dataset")

    col1, col2 = st.columns(2, gap="large")

    items = list(DATASET_INFO.items())
    mid = len(items) // 2

    with col1:
        for label, nilai in items[:mid]:
            st.markdown(f"""
            <div class="ds-item">
                <span class="ds-label">{label}</span>
                <span class="ds-value">{nilai}</span>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        for label, nilai in items[mid:]:
            st.markdown(f"""
            <div class="ds-item">
                <span class="ds-label">{label}</span>
                <span class="ds-value">{nilai}</span>
            </div>
            """, unsafe_allow_html=True)


def render_system_workflow() -> None:
    """Menampilkan alur kerja sistem secara keseluruhan."""
    st.markdown("### ⚙️ Alur Kerja Sistem (System Workflow)")

    langkah = [
        {
            "icon": "🛰️",
            "judul": "Google Earth Engine",
            "sub": "Akuisisi data satelit — MODIS LST, SMAP Soil Moisture, Landsat NDVI, SRTM DEM",
        },
        {
            "icon": "⚗️",
            "judul": "Feature Engineering",
            "sub": "Normalisasi, lag features (t-1, t-7), rolling statistics, encoding bulan",
        },
        {
            "icon": "🤖",
            "judul": "Dual-Branch ANFIS-LSTM",
            "sub": "Inferensi model AI — Branch A (LSTM) + Branch B (ANFIS) → Fusion → Prediksi",
        },
        {
            "icon": "📊",
            "judul": "Google Spreadsheet",
            "sub": "Penyimpanan hasil prediksi dan data historis via Google Sheets API",
        },
        {
            "icon": "🖥️",
            "judul": "ThermaWatch Dashboard",
            "sub": "Visualisasi real-time — peta, tren, kartu prediksi, dan analisis lingkungan",
        },
    ]

    for i, langkah_item in enumerate(langkah):
        st.markdown(f"""
        <div class="workflow-step">
            <div class="step-icon">{langkah_item['icon']}</div>
            <div class="step-text">
                <div class="step-title">{langkah_item['judul']}</div>
                <div class="step-sub">{langkah_item['sub']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if i < len(langkah) - 1:
            st.markdown('<div class="workflow-arrow">↓</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FUNGSI UTAMA
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Entry point halaman Model Information."""
    render_header()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🧠 Arsitektur Model",
        "📐 Performa",
        "🗂️ Dataset",
        "⚙️ Workflow Sistem",
    ])

    with tab1:
        render_model_overview()
        st.markdown("")
        render_architecture_diagram()

    with tab2:
        render_performance_metrics()

    with tab3:
        render_dataset_info()
        st.markdown("")
        st.info(
            "💡 Data dikumpulkan menggunakan Google Earth Engine (GEE) "
            "dengan skrip JavaScript yang dieksekusi secara otomatis dan diekspor ke Google Drive."
        )

    with tab4:
        render_system_workflow()


if __name__ == "__main__" or True:
    main()
