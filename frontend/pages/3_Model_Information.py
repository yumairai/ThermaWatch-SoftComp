"""
ThermaWatch-SoftComp | Halaman 4: Model Information
Menjelaskan arsitektur, performa, dan dataset model Dual-Branch ANFIS-LSTM.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def get_performance_metrics(is_modis: bool = False) -> dict:
    if is_modis:
        return {
            "RMSE": {"nilai": "1.19", "satuan": "°C (H1)", "keterangan": "Root Mean Squared Error"},
            "MAE": {"nilai": "0.73", "satuan": "°C (H1)", "keterangan": "Mean Absolute Error"},
            "R²": {"nilai": "0.662", "satuan": " (H1)", "keterangan": "Coefficient of Determination"},
        }
    return {
        "RMSE": {"nilai": "0.72", "satuan": "°C (H1)", "keterangan": "Root Mean Squared Error"},
        "MAE": {"nilai": "0.55", "satuan": "°C (H1)", "keterangan": "Mean Absolute Error"},
        "R²": {"nilai": "0.697", "satuan": " (H1)", "keterangan": "Coefficient of Determination"},
    }

def get_dataset_info(is_modis: bool = False) -> dict:
    if is_modis:
        return {
            "Jumlah Data": "86.265 baris",
            "Jumlah Fitur": "18 fitur input",
            "Rentang Waktu": "Januari 2014 – Mei 2026",
            "Jumlah Lokasi": "15 kabupaten/kota (Jawa Barat)",
            "Resolusi Temporal": "Harian",
            "Resolusi Spasial": "~1 km × 1 km",
            "Sumber Data": "Google Earth Engine (MODIS LST, SMAP, NDVI)",
            "Train / Val / Test": "80% / 10% / 10% (Kronologis)",
        }
    return {
        "Jumlah Data": "84.764 baris",
        "Jumlah Fitur": "18 fitur input",
        "Rentang Waktu": "Januari 2014 – Mei 2026",
        "Jumlah Lokasi": "15 kabupaten/kota (Jawa Barat)",
        "Resolusi Temporal": "Harian",
        "Resolusi Spasial": "~1 km × 1 km",
        "Sumber Data": "Google Earth Engine (ERA5 LST, SMAP, NDVI)",
        "Train / Val / Test": "80% / 10% / 10% (Kronologis)",
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
            "deskripsi": "Memproses data deret waktu LST historis (lookback 14 hari) untuk menangkap pola temporal dan tren musiman suhu.",
            "detail": [
                "Update Bobot: Parameter internal gate (input, forget, cell, output) diperbarui hulu-ke-hilir.",
                "Arsitektur: 2 layer LSTM bertumpuk",
                "Hidden size: 256 unit",
                "Input window: 14 hari",
                "Unidirectional LSTM",
            ],
        },
        {
            "icon": "🔶",
            "judul": "Branch B — ANFIS",
            "subjudul": "Environmental Embedding",
            "deskripsi": "Memodelkan hubungan non-linear antara variabel lingkungan (NDVI, Soil Moisture, Elevasi) menggunakan logika fuzzy adaptif.",
            "detail": [
                "Update Bobot: Parameter centers & log_sigma kurva Gaussian (DifferentiableFuzzyLayer) dioptimasi dinamis.",
                "5 membership functions per fitur",
                "Fuzzy rules: 32 aturan",
                "Input: 3 variabel lingkungan",
                "Takagi-Sugeno FIS",
            ],
        },
        {
            "icon": "🔗",
            "judul": "Fusion Layer",
            "subjudul": "Branch Merging",
            "deskripsi": "Menggabungkan representasi temporal dari LSTM dan representasi spasial-lingkungan dari ANFIS.",
            "detail": [
                "Update Bobot: Bobot LayerNorm dan parameter dense projection layer.",
                "Concatenation merging",
                "Dimensi: 256 (LSTM) + 64 (ANFIS) = 320 unit",
                "Shared MLP: 2 hidden layers (GELU)",
                "Dropout: 0.25 dan 0.15",
            ],
        },
        {
            "icon": "🎯",
            "judul": "Output Layer",
            "subjudul": "Multi-Horizon Prediction",
            "deskripsi": "Menghasilkan ramalan anomali suhu untuk tiga horizon waktu secara terpisah dan simultan.",
            "detail": [
                "Update Bobot: Parameter linear projection head pada masing-masing target head.",
                "Output nodes: 3 projection heads",
                "Target: H+1 (Hari Ini), H+3 (H+2), H+7 (H+6)",
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


def render_architecture_diagram():
    st.markdown("### 🧠 ANFIS-LSTM Architecture")

    fig = go.Figure()

    nodes = [
        # ===============================
        # TEMPORAL BRANCH
        # ===============================
        {
            "x": 0.23,
            "y": 0.88,
            "label": "Dynamic Features\n(LST Time Series)",
            "color": "#1e40af",
        },
        {
            "x": 0.23,
            "y": 0.68,
            "label": "Differentiable\nFuzzy Layer\n<b>[Update: Centers & Sigma]</b>",
            "color": "#2563eb",
        },
        {
            "x": 0.23,
            "y": 0.48,
            "label": "LSTM Encoder\n<b>[Update: Gate Weights]</b>",
            "color": "#3b82f6",
        },

        # ===============================
        # ENVIRONMENTAL BRANCH
        # ===============================
        {
            "x": 0.77,
            "y": 0.88,
            "label": "Static Features\n(NDVI, Elev, Soil Moisture)",
            "color": "#166534",
        },
        {
            "x": 0.77,
            "y": 0.58,
            "label": "Environment\nBranch MLP\n<b>[Update: Dense Weights]</b>",
            "color": "#22c55e",
        },

        # ===============================
        # FUSION
        # ===============================
        {
            "x": 0.50,
            "y": 0.22,
            "label": "Feature Fusion\n<b>[Update: LayerNorm]</b>",
            "color": "#7c3aed",
        },
        {
            "x": 0.50,
            "y": -0.04,
            "label": "Shared MLP\n<b>[Update: Dense Weights]</b>",
            "color": "#9333ea",
        },

        # ===============================
        # OUTPUTS
        # ===============================
        {
            "x": 0.25,
            "y": -0.28,
            "label": "H+1\n<b>[Update: Head Weights]</b>",
            "color": "#ea580c",
        },
        {
            "x": 0.50,
            "y": -0.28,
            "label": "H+3\n<b>[Update: Head Weights]</b>",
            "color": "#ea580c",
        },
        {
            "x": 0.75,
            "y": -0.28,
            "label": "H+7\n<b>[Update: Head Weights]</b>",
            "color": "#ea580c",
        },
    ]

    # ===============================
    # DRAW BOXES
    # ===============================
    for node in nodes:
        if "H+" in node["label"]:
            width = 0.11
            height = 0.065
        else:
            width = 0.17
            height = 0.08

        fig.add_shape(
            type="rect",
            x0=node["x"] - width,
            x1=node["x"] + width,
            y0=node["y"] - height,
            y1=node["y"] + height,
            fillcolor=node["color"],
            line=dict(color="white", width=1.5),
        )

        fig.add_annotation(
            x=node["x"],
            y=node["y"],
            text=node["label"].replace("\n", "<br>"),
            showarrow=False,
            font=dict(
                color="white",
                size=11,
            ),
        )

    # Helper to add a line shape
    def add_line(x_start, y_start, x_end, y_end, color, width=1.5, dash=None):
        line_dict = dict(color=color, width=width)
        if dash:
            line_dict["dash"] = dash
        fig.add_shape(
            type="line",
            x0=x_start, y0=y_start, x1=x_end, y1=y_end,
            line=line_dict
        )

    def draw_arrow(x_start, y_start, x_end, y_end, color, arrowhead=2, size=1.0):
        fig.add_annotation(
            x=x_end, y=y_end,
            ax=x_start, ay=y_start,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=arrowhead,
            arrowsize=size,
            arrowwidth=1.5,
            arrowcolor=color,
            standoff=0,
            startstandoff=0,
            text=""
        )

    # ===============================
    # 1. FORWARD PASS CONNECTIONS (GREY #94a3b8)
    # ===============================
    # Left Branch vertical arrows
    draw_arrow(0.23, 0.80, 0.23, 0.76, "#94a3b8")
    draw_arrow(0.23, 0.60, 0.23, 0.56, "#94a3b8")
    
    # Right Branch vertical arrow
    draw_arrow(0.77, 0.80, 0.77, 0.66, "#94a3b8")

    # LSTM to Fusion (Boxy: Down then Right)
    add_line(0.23, 0.40, 0.23, 0.22, "#94a3b8")
    draw_arrow(0.23, 0.22, 0.33, 0.22, "#94a3b8")

    # Env MLP to Fusion (Boxy: Down then Left)
    add_line(0.77, 0.50, 0.77, 0.22, "#94a3b8")
    draw_arrow(0.77, 0.22, 0.67, 0.22, "#94a3b8")

    # Fusion to Shared MLP (Vertical straight)
    draw_arrow(0.50, 0.14, 0.50, 0.04, "#94a3b8")

    # Shared MLP to Outputs (Split at y = -0.165)
    add_line(0.50, -0.12, 0.50, -0.165, "#94a3b8")
    
    add_line(0.50, -0.165, 0.25, -0.165, "#94a3b8")
    draw_arrow(0.25, -0.165, 0.25, -0.215, "#94a3b8")

    draw_arrow(0.50, -0.165, 0.50, -0.215, "#94a3b8")

    add_line(0.50, -0.165, 0.75, -0.165, "#94a3b8")
    draw_arrow(0.75, -0.165, 0.75, -0.215, "#94a3b8")

    # ===============================
    # 2. BACKPROPAGATION PASS (RED DOTTED #f43f5e - SIDE ROUTED)
    # ===============================
    # H1, H3, H7 up to Shared MLP (split at y = -0.185)
    add_line(0.27, -0.215, 0.27, -0.185, "#f43f5e", dash="dot")
    add_line(0.27, -0.185, 0.47, -0.185, "#f43f5e", dash="dot")
    draw_arrow(0.47, -0.185, 0.47, -0.12, "#f43f5e", arrowhead=3, size=1.1)

    draw_arrow(0.49, -0.215, 0.49, -0.12, "#f43f5e", arrowhead=3, size=1.1)

    add_line(0.73, -0.215, 0.73, -0.185, "#f43f5e", dash="dot")
    add_line(0.73, -0.185, 0.53, -0.185, "#f43f5e", dash="dot")
    draw_arrow(0.53, -0.185, 0.53, -0.12, "#f43f5e", arrowhead=3, size=1.1)

    # Shared MLP up to Fusion (shifted left)
    draw_arrow(0.47, 0.04, 0.47, 0.14, "#f43f5e", arrowhead=3, size=1.1)

    # Left Backprop Side-Route (x = 0.03) for Temporal Branch
    add_line(0.33, 0.17, 0.03, 0.17, "#f43f5e", dash="dot")
    add_line(0.03, 0.17, 0.03, 0.88, "#f43f5e", dash="dot")
    draw_arrow(0.03, 0.48, 0.06, 0.48, "#f43f5e", arrowhead=3, size=1.1)
    draw_arrow(0.03, 0.68, 0.06, 0.68, "#f43f5e", arrowhead=3, size=1.1)
    draw_arrow(0.03, 0.88, 0.06, 0.88, "#f43f5e", arrowhead=3, size=1.1)

    # Right Backprop Side-Route (x = 0.97) for Environmental Branch
    add_line(0.67, 0.17, 0.97, 0.17, "#f43f5e", dash="dot")
    add_line(0.97, 0.17, 0.97, 0.88, "#f43f5e", dash="dot")
    draw_arrow(0.97, 0.58, 0.94, 0.58, "#f43f5e", arrowhead=3, size=1.1)
    draw_arrow(0.97, 0.88, 0.94, 0.88, "#f43f5e", arrowhead=3, size=1.1)

    # ===============================
    # BRANCH LABELS & LEGENDS
    # ===============================
    fig.add_annotation(
        x=0.23,
        y=1.00,
        text="<b>Temporal Branch (LSTM)</b>",
        showarrow=False,
        font=dict(size=13, color="#60a5fa")
    )

    fig.add_annotation(
        x=0.77,
        y=1.00,
        text="<b>Environmental Branch (ANFIS)</b>",
        showarrow=False,
        font=dict(size=13, color="#4ade80")
    )

    # Legenda Aliran
    fig.add_annotation(
        x=0.08,
        y=0.10,
        text="➡ Aliran Forward Pass (Prediksi)<br><span style='color:#f43f5e;'>⬅ Aliran Backward Pass (Pembaruan Bobot Paralel)</span>",
        showarrow=False,
        align="left",
        font=dict(size=11, color="#cbd5e1"),
        bgcolor="rgba(15,23,42,0.8)",
        bordercolor="rgba(148,163,184,0.2)",
        borderwidth=1,
        borderpad=6
    )

    # ===============================
    # LAYOUT
    # ===============================
    fig.update_layout(
        template="plotly_dark",
        height=800,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            visible=False,
            range=[0, 1]
        ),
        yaxis=dict(
            visible=False,
            range=[-0.35, 1.05]
        ),
    )

    st.plotly_chart(fig, width="stretch")


def render_performance_metrics(is_modis: bool = False) -> None:
    """Menampilkan badge metrik performa model."""
    st.markdown("### 📐 Metrik Performa Model")

    metrics = get_performance_metrics(is_modis)
    cols = st.columns(3)
    for col, (nama, info) in zip(cols, metrics.items()):
        with col:
            st.markdown(f"""
            <div class="metric-badge">
                <div class="mb-label">{nama}</div>
                <div class="mb-value">{info['nilai']}{info['satuan']}</div>
                <div class="mb-desc">{info['keterangan']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # Grafik perbandingan Metrik Evaluasi Test per horizon
    horizons = ["H+1 (Hari Ini)", "H+3 (H+2)", "H+7 (H+6)"]
    if is_modis:
        rmse_test = [1.194, 1.384, 1.506]
        mae_test = [0.726, 0.925, 1.081]
        r2_test = [0.662, 0.555, 0.480]
    else:
        rmse_test = [0.723, 0.807, 0.904]
        mae_test = [0.546, 0.608, 0.659]
        r2_test = [0.697, 0.624, 0.518]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="RMSE (°C)", x=horizons, y=rmse_test, marker_color="#38bdf8"))
    fig.add_trace(go.Bar(name="MAE (°C)", x=horizons, y=mae_test, marker_color="#f59e0b"))
    fig.add_trace(go.Bar(name="R² (Determinasi)", x=horizons, y=r2_test, marker_color="#4ade80"))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="group",
        title="Metrik Evaluasi Set Uji (Test Set) per Horizon Prediksi",
        xaxis_title="Horizon Prediksi",
        yaxis_title="Nilai Metrik",
        height=350,
        margin=dict(l=10, r=10, t=100, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, width="stretch")


def render_dataset_info(is_modis: bool = False) -> None:
    """Menampilkan informasi dataset yang digunakan."""
    st.markdown("### 🗂️ Informasi Dataset")

    col1, col2 = st.columns(2, gap="large")

    dataset_info = get_dataset_info(is_modis)
    items = list(dataset_info.items())
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
    
    # 1. Tambahkan selektor model di sidebar
    model_pilihan = st.sidebar.selectbox(
        "🤖 Model AI Utama",
        options=["ERA5 ANFIS-LSTM", "MODIS ANFIS-LSTM"],
        index=0,
        help="Pilih model dasar yang ingin dilihat informasinya."
    )
    is_modis = model_pilihan == "MODIS ANFIS-LSTM"

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧠 Arsitektur Model",
        "📐 Performa",
        "🗂️ Dataset",
        "⚙️ Workflow Sistem",
        "⚖️ Etika & Transparansi"
    ])

    with tab1:
        render_model_overview()
        st.markdown("")
        render_architecture_diagram()

    with tab2:
        render_performance_metrics(is_modis)

    with tab3:
        render_dataset_info(is_modis)
        st.markdown("")
        st.info(
            "💡 Data dikumpulkan menggunakan Google Earth Engine (GEE) "
            "dengan skrip JavaScript yang dieksekusi secara otomatis dan diekspor ke Google Drive."
        )

    with tab4:
        render_system_workflow()

    with tab5:
        render_ethics_and_transparency()


def render_ethics_and_transparency() -> None:
    """Menampilkan informasi evaluasi etika, tanggung jawab, dan transparansi (XAI)."""
    st.markdown("### ⚖️ Pengembangan AI yang Bertanggung Jawab (Responsible AI)")
    st.markdown(
        "Penerapan prinsip etika kecerdasan buatan pada ThermaWatch untuk keadilan wilayah, "
        "keterjelasan model, serta keamanan data publik."
    )
    st.divider()

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### 🌾 Keadilan Prediksi & Wilayah (Fairness)")
        st.info("""
        * **Karakteristik Lokal**: Wilayah pegunungan (Garut/Sumedang) memiliki profil suhu LST berbeda dengan pesisir (Karawang/Bekasi).
        * **Mitigasi Bias**: Nilai anomali dihitung secara lokal spesifik per kabupaten berdasarkan historis bulanan wilayah terkait (bukan rata-rata global Jawa Barat).
        * **Hasil**: Akurasi deteksi tetap adil untuk kawasan dataran tinggi maupun dataran rendah.
        """)

        st.markdown("#### 🔒 Perlindungan Data & Keamanan (Privacy & Security)")
        st.success("""
        * **Sumber Data**: 100% menggunakan data satelit publik (MODIS, ERA5, NDVI).
        * **Data Pribadi**: Tidak mengumpulkan data personal sensitif atau melacak lokasi fisik GPS pengguna.
        * **Keamanan Sistem**: Kredensial API Google Sheets & GCP diamankan di backend menggunakan enkripsi dan tidak dipublikasikan ke repositori.
        """)

    with col2:
        st.markdown("#### 🔍 Transparansi & Keterjelasan (Explainable AI)")
        st.warning("""
        * **Logika Fuzzy Terbuka**: Model *hybrid* ANFIS-LSTM memproses variabel suhu melalui `DifferentiableFuzzyLayer` yang kurva keanggotaannya dapat dilacak secara matematis.
        * **Aturan Status Objektif**: Klasifikasi status kerawanan didasarkan pada nilai anomali suhu:
          * **Aman**: < 1.5°C
          * **Waspada**: 1.5°C s.d. 3.0°C
          * **Bahaya**: ≥ 3.0°C
        * **Penanganan Eror (Fallback)**: Otomatis menggunakan koordinat historis terdekat yang valid jika satelit terhalang awan tebal.
        """)

        st.markdown("#### 📢 Dampak Sosial (Social Impact)")
        st.info("""
        * **Potensi Risiko**: Kesalahan prediksi suhu berisiko memicu spekulasi harga tanah atau kepanikan publik.
        * **Mitigasi**: Menampilkan *disclaimer* bahwa hasil sistem merupakan simulasi akademis yang wajib diverifikasi dengan data ground-truth BMKG/BPBD sebelum pengambilan kebijakan.
        """)


if __name__ == "__main__" or True:
    main()
