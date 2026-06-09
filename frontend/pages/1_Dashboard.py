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
import math
import re
from datetime import datetime, timedelta

from backend.services.sheets_service import get_daily_data, get_predictions



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

def _status_badge(anomali: float) -> tuple[str, str, str]:
    """Mengembalikan (emoji_status, label, css_class) berdasarkan anomali suhu."""
    if pd.isna(anomali):
        return "⚪", "N/A", ""
        
    if anomali >= 2.5:
        return "🔴", "BAHAYA (Potensi Manifestasi Aktif)", "status-bahaya"
    elif anomali >= 1.0:
        return "🟡", "WASPADA (Pantau Berkala)", "status-waspada"
    elif anomali <= -2.0:
        return "❄️", "ANOMALI DINGIN (Suhu Ekstrem)", "status-aman"
    else:
        return "🟢", "AMAN (Fluktuasi Normal)", "status-aman"


def _warna_suhu(suhu: float) -> list[int]:
    """Mengkonversi nilai suhu ke warna RGB untuk PyDeck."""
    # Normalisasi antara 25°C (biru) hingga 50°C (merah)
    t = max(0, min(1, (suhu - 25) / 25))
    r = int(255 * t)
    g = int(255 * (1 - abs(t - 0.5) * 2))
    b = int(255 * (1 - t))
    return [r, g, b, 180]


def _warna_anomali(anomali: float) -> list[int]:
    """Mengembalikan warna status anomali untuk peta: aman/waspada/bahaya."""
    if pd.isna(anomali):
        return [148, 163, 184, 180]
    if anomali >= 3.0:
        return [239, 68, 68, 180]   # merah
    if anomali >= 1.5:
        return [245, 158, 11, 180]  # kuning
    return [34, 197, 94, 180]       # hijau


def _normalize_name(value: str) -> str:
    """Normalisasi nama kabupaten agar cocok antara dataframe dan GeoJSON."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


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
        tanggal_col = pd.to_datetime(df["Tanggal"], errors="coerce").dropna()
        if tanggal_col.empty:
            today = pd.Timestamp.today().normalize()
            tgl_min = today.date()
            tgl_max = today.date()
        else:
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


def render_kpi(df: pd.DataFrame, tgl_mulai=None, tgl_akhir=None) -> None:
    """Menampilkan kartu KPI metrik utama."""
    st.markdown("### 📈 Metrik Utama")
    st.caption(
        "Nilai di bawah adalah rata-rata, maksimum, minimum, dan lokasi terpanas dari data aktual yang sedang difilter."
    )
    if tgl_mulai is not None and tgl_akhir is not None:
        st.caption(
            f"Rentang tanggal yang tampil: {tgl_mulai.strftime('%d %b %Y')} sampai {tgl_akhir.strftime('%d %b %Y')}"
        )

    suhu = df["Suhu_Celcius"]
    idx_terpanas = df["Suhu_Celcius"].idxmax()
    lokasi_terpanas = df.loc[idx_terpanas, "Kabupaten"] if "Kabupaten" in df.columns else "N/A"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌡️ Suhu Rata-rata", f"{suhu.mean():.1f} °C")
    col2.metric("🔥 Suhu Maksimum", f"{suhu.max():.1f} °C")
    col3.metric("❄️ Suhu Minimum", f"{suhu.min():.1f} °C")
    col4.metric("📍 Lokasi Terpanas", lokasi_terpanas)


def _normalize_prediction_frame(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Normalisasi nama kolom prediksi agar mengikuti data spreadsheet yang sama."""
    pred_norm = pred_df.copy()
    pred_norm = pred_norm.rename(columns={
        "prediksi_suhu_h1": "pred_h1",
        "prediksi_suhu_h3": "pred_h3",
        "prediksi_suhu_h7": "pred_h7",
        "prediksi_anomali_h1": "anom_h1",
        "prediksi_anomali_h3": "anom_h3",
        "prediksi_anomali_h7": "anom_h7",
        "kabupaten": "Kabupaten",
    })
    for col in ["pred_h1", "pred_h3", "pred_h7", "anom_h1", "anom_h3", "anom_h7"]:
        if col in pred_norm.columns:
            pred_norm[col] = pd.to_numeric(pred_norm[col], errors="coerce")
    if "Kabupaten" in pred_norm.columns:
        pred_norm["Kabupaten"] = pred_norm["Kabupaten"].astype(str).str.strip().str.title()
    return pred_norm


def _resolve_prediction_pair(pred_df: pd.DataFrame, label: str, kabupaten: str = "Semua Lokasi") -> tuple[float, float]:
    """Ambil nilai prediksi dari spreadsheet yang sama, sesuai kabupaten yang sedang difilter."""
    pred_norm = _normalize_prediction_frame(pred_df)

    if kabupaten and kabupaten != "Semua Lokasi" and "Kabupaten" in pred_norm.columns:
        pred_norm = pred_norm[pred_norm["Kabupaten"] == kabupaten]

    if pred_norm.empty:
        return 37.5, 0.0

    row = pred_norm.iloc[-1].copy()
    mapping = {
        "Hari Ini": ("pred_h1", "anom_h1"),
        "H+2": ("pred_h3", "anom_h3"),
        "H+6": ("pred_h7", "anom_h7"),
    }
    suhu_col, anom_col = mapping[label]
    suhu = float(row[suhu_col]) if pd.notna(row.get(suhu_col)) else 37.5
    anom = float(row[anom_col]) if pd.notna(row.get(anom_col)) else 0.0
    return suhu, anom


def render_prediction_cards(pred_df: pd.DataFrame, kabupaten: str = "Semua Lokasi") -> None:
    """Menampilkan kartu prediksi Hari Ini, H+2, H+6."""
    st.markdown("### 🤖 Prediksi Suhu (Dual-Branch ANFIS-LSTM)")

    horizons = {
        "Hari Ini": "Hari Ini",
        "H+2": "H+2",
        "H+6": "H+6",
    }

    cols = st.columns(3)
    for col, (label, _) in zip(cols, horizons.items()):
        nilai_suhu, nilai_anomali = _resolve_prediction_pair(pred_df, label, kabupaten)

        emoji, status_txt, css_class = _status_badge(nilai_anomali)
        with col:
            st.markdown(f"""
            <div class="pred-card">
                <div class="label">{label}</div>
                <div class="value">{nilai_suhu:.1f} °C</div>
                <div class="status {css_class}" style="font-size:0.75rem; line-height:1.2;">{emoji} {status_txt}</div>
            </div>
            """, unsafe_allow_html=True)


import json


with open("data/geojson/jawa_barat_kabupaten_detailed.json", "r", encoding="utf-8") as f:
    SAMPLE_GEOJSON = json.load(f)


def render_gis_map(df: pd.DataFrame, pred_df: pd.DataFrame, geojson: dict, kabupaten: str = "Semua Lokasi") -> None:
    """GIS Map: prediksi jadi utama + heatmap data aktual + tooltip sinkron kabupaten."""


    st.markdown("### 🗺️ GIS Map")

    KAB_CENTROIDS = {
        "Bandung": (-6.898, 107.639),
        "Bandung Barat": (-6.898, 107.425),
        "Banjar": (-7.380, 108.553),
        "Bekasi": (-6.306, 106.959),
        "Bogor": (-6.586, 106.782),
        "Ciamis": (-7.291, 108.425),
        "Cianjur": (-7.023, 107.159),
        "Cimahi": (-6.891, 107.545),
        "Cirebon": (-6.740, 108.553),
        "Depok": (-6.394, 106.828),
        "Garut": (-7.266, 107.842),
        "Indramayu": (-6.482, 108.126),
        "Karawang": (-6.281, 107.406),
        "Kuningan": (-6.979, 108.582),
        "Majalengka": (-6.751, 108.195),
        "Pangandaran": (-7.656, 108.517),
        "Purwakarta": (-6.601, 107.379),
        "Subang": (-6.449, 107.687),
        "Sukabumi": (-6.935, 106.928),
        "Sumedang": (-6.838, 107.991),
        "Tasikmalaya": (-7.359, 108.229)
    }

    # =========================
    # 1. CLEAN DATA
    # =========================
    df_map = df.copy()
    df_map["Kabupaten"] = df_map["Kabupaten"].astype(str).str.strip().str.title()
    
    KAB_GEOMETRIES = {feat['properties']['name']: feat['geometry'] for feat in geojson['features']}

    def is_point_in_polygon(x, y, poly):
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def is_inside_kabupaten(lat, lon, geom):
        t = geom['type']
        if t == "Polygon":
            return is_point_in_polygon(lon, lat, geom['coordinates'][0])
        elif t == "MultiPolygon":
            for poly in geom['coordinates']:
                if is_point_in_polygon(lon, lat, poly[0]):
                    return True
        return False

    # Override database coordinates with centroids only if they are outside the polygon boundary
    corrected_coords = []
    for idx, row in df_map.iterrows():
        kab = row["Kabupaten"]
        lat = row.get("Latitude")
        lon = row.get("Longitude")
        centroid = KAB_CENTROIDS.get(kab, (-7.2, 107.5))
        
        if pd.isna(lat) or pd.isna(lon):
            corrected_coords.append(centroid)
        else:
            geom = KAB_GEOMETRIES.get(kab)
            if geom:
                try:
                    if is_inside_kabupaten(float(lat), float(lon), geom):
                        corrected_coords.append((float(lat), float(lon)))
                    else:
                        corrected_coords.append(centroid)
                except Exception:
                    corrected_coords.append(centroid)
            else:
                corrected_coords.append((float(lat), float(lon)))

    df_map["Latitude"] = [c[0] for c in corrected_coords]
    df_map["Longitude"] = [c[1] for c in corrected_coords]
    df_map["Suhu_Celcius"] = pd.to_numeric(df_map["Suhu_Celcius"], errors="coerce")

    # =========================
    # 2. AGREGASI KABUPATEN (DATA HARI INI / LAST RECORD)
    # =========================
    kab_df = df_map.sort_values("Tanggal").groupby("Kabupaten").last().reset_index()

    # =========================
    # 3. AMBIL PREDIKSI H+1 DARI SPREADSHEET YANG SAMA DENGAN KARTU PREDIKSI
    # =========================
    pred_norm = _normalize_prediction_frame(pred_df)
    if kabupaten and kabupaten != "Semua Lokasi" and "Kabupaten" in pred_norm.columns:
        pred_norm = pred_norm[pred_norm["Kabupaten"] == kabupaten]
    if pred_norm.empty:
        pred_norm = _normalize_prediction_frame(pred_df)

    pred_map = pred_norm[["Kabupaten", "pred_h1", "anom_h1"]].copy()
    pred_map = pred_map.drop_duplicates(subset=["Kabupaten"], keep="last")
    pred_map["Kabupaten"] = pred_map["Kabupaten"].astype(str).str.title()

    kab_df = kab_df.merge(pred_map, on="Kabupaten", how="left")
    kab_df["pred_suhu"] = pd.to_numeric(kab_df["pred_h1"], errors="coerce")
    kab_df["pred_anomali"] = pd.to_numeric(kab_df["anom_h1"], errors="coerce")
    kab_df = kab_df.dropna(subset=["pred_suhu", "pred_anomali"])

    # =========================
    # 4. JOIN KE GEOJSON (POLYGON DATA)
    # =========================
    kab_data = kab_df.set_index("Kabupaten")[["pred_suhu", "pred_anomali"]].to_dict("index")

    def ceil1(x):
        return math.ceil(float(x) * 10) / 10 if x is not None else None

    for feature in geojson["features"]:
        name = feature["properties"].get("name", "").strip().title()

        if name in kab_data:
            suhu = kab_data[name]["pred_suhu"]
            anom = kab_data[name]["pred_anomali"]

            feature["properties"]["pred_suhu"] = ceil1(suhu)
            feature["properties"]["pred_anomali"] = ceil1(anom)

            # warna polygon berdasarkan anomali prediksi, bukan suhu absolut
            if anom >= 3.0:
                color = [239, 68, 68, 160]
            elif anom >= 1.5:
                color = [245, 158, 11, 140]
            else:
                color = [34, 197, 94, 120]
        else:
            feature["properties"]["pred_suhu"] = None
            feature["properties"]["pred_anomali"] = None
            color = [30, 41, 59, 60]

        feature["properties"]["color"] = color

    polygon_layer = pdk.Layer(
        "GeoJsonLayer",
        data=geojson,
        get_fill_color="properties.color",
        get_line_color=[255, 255, 255, 120],
        line_width_min_pixels=1.5,
        pickable=True,
    )

    # =========================
    # 6. HEATMAP LAYER (DATA AKTUAL SPREADSHEET)
    # =========================
    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=df_map,
        get_position=["Longitude", "Latitude"],
        get_weight="Suhu_Celcius",
        radius_pixels=55,
        intensity=1.0,
        threshold=0.05,
    )

    # Dynamic center based on selected kabupaten
    lat_center = -7.2
    lon_center = 107.5
    zoom_level = 7.5
    if kabupaten and kabupaten != "Semua Lokasi" and kabupaten in KAB_CENTROIDS:
        lat_center, lon_center = KAB_CENTROIDS[kabupaten]
        zoom_level = 8.8

    view_state = pdk.ViewState(
        latitude=lat_center,
        longitude=lon_center,
        zoom=zoom_level,
        pitch=35,
    )

    # =========================
    # 8. TOOLTIP POLYGON (PREDIKSI H1)
    # =========================
    tooltip = {
        "html": """
        <div style="font-size: 11px; line-height: 1.4;">
            <b>📍 {name}</b><br/>
            🌡️ Prediksi Suhu: {pred_suhu} °C<br/>
            ⚠️ Prediksi Anomali: {pred_anomali} °C
        </div>
        """,
        "style": {
            "backgroundColor": "rgba(15, 23, 42, 0.95)",
            "color": "#f1f5f9",
            "borderRadius": "6px",
            "padding": "6px 10px",
            "border": "1px solid rgba(255, 255, 255, 0.15)",
            "zIndex": "1000",
        },
    }

    # =========================
    # 9. RENDER MAP
    # =========================
    st.pydeck_chart(
        pdk.Deck(
            layers=[polygon_layer, heatmap_layer],
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        ),
        width="stretch",
    )


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

    st.plotly_chart(fig, width="stretch")


def render_raw_data(df: pd.DataFrame) -> None:
    """Menampilkan data mentah dalam expander."""
    with st.expander("🗄️ Lihat Data Mentah", expanded=False):
        st.markdown(f"**Total baris:** {len(df):,} | **Total kolom:** {len(df.columns)}")
        st.dataframe(
            df.reset_index(drop=True),
            width="stretch",
            height=300,
        )
        st.download_button(
            label="⬇️ Unduh CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="thermawatch_data.csv",
            mime="text/csv",
        )


def render_dashboard_explanation(df_filtered: pd.DataFrame, pred_df: pd.DataFrame, lokasi: str) -> None:
    """Menampilkan analisis kontributor prediksi untuk lokasi terpilih (XAI)."""
    if lokasi == "Semua Lokasi":
        return

    with st.expander("🔍 Analisis Faktor Prediksi (Explainable AI)", expanded=True):
        # Ambil record terbaru untuk lokasi tersebut
        latest_row = df_filtered.iloc[-1]
        
        # Ambil nilai parameter aktual saat ini
        lst_val = latest_row.get("Suhu_Celcius", 30.0)
        anom_val = latest_row.get("Anomaly", 0.0)
        sm_val = latest_row.get("Soil_Moisture", 30.0)
        ndvi_val = latest_row.get("NDVI", 0.4)
        elev_val = latest_row.get("Elevation", 500)
        
        # Rata-rata historis bulanan = Suhu aktual - Anomali aktual
        hist_mean = lst_val - anom_val
        
        # 1. Analisis Suhu (LST)
        if anom_val > 1.5:
            lst_val_txt = "🔴 Anomali Panas Aktif"
            lst_exp = f"Suhu saat ini ({lst_val:.1f}°C) berada signifikan (+{anom_val:+.1f}°C) di atas rata-rata historis ({hist_mean:.1f}°C)."
        elif anom_val < -1.5:
            lst_val_txt = "❄️ Anomali Dingin"
            lst_exp = f"Suhu saat ini ({lst_val:.1f}°C) berada di bawah rata-rata historis bulanan ({hist_mean:.1f}°C)."
        else:
            lst_val_txt = "🟢 Normal"
            lst_exp = f"Suhu permukaan tanah saat ini stabil mendekati rata-rata historis bulanan ({hist_mean:.1f}°C)."

        # 2. Analisis Kelembaban Tanah (Soil Moisture)
        sm_display = sm_val
        if sm_val <= 1.0:
            sm_display = sm_val * 100
            
        if sm_display < 20.0:
            sm_val_txt = "🔴 Sangat Kering"
            sm_exp = f"Kelembaban tanah rendah ({sm_display:.1f}%) membatasi pendinginan permukaan tanah alami."
        elif sm_display < 40.0:
            sm_val_txt = "🟡 Sedang"
            sm_exp = f"Kelembaban tanah ({sm_display:.1f}%) berada pada level sedang/normal."
        else:
            sm_val_txt = "🟢 Lembab / Basah"
            sm_exp = f"Kelembaban tanah basah ({sm_display:.1f}%) meredam kenaikan suhu ekstrem secara efektif."

        # 3. Analisis NDVI
        if ndvi_val < 0.2:
            ndvi_val_txt = "🔴 Lahan Terbuka"
            ndvi_exp = f"Indeks NDVI sangat rendah ({ndvi_val:.2f}) mencerminkan dominasi area gundul penyerap panas matahari."
        elif ndvi_val < 0.6:
            ndvi_val_txt = "🟡 Vegetasi Sedang"
            ndvi_exp = f"Tingkat vegetasi sedang ({ndvi_val:.2f}) memberikan perlindungan termal campuran."
        else:
            ndvi_val_txt = "🟢 Vegetasi Lebat"
            ndvi_exp = f"Kanopi vegetasi lebat ({ndvi_val:.2f}) memaksimalkan evapotranspirasi alami untuk mendinginkan area."

        # 4. Ketinggian (Elevasi)
        if elev_val > 1000:
            elev_val_txt = "🏔️ Dataran Tinggi"
            elev_exp = f"Ketinggian wilayah ({elev_val:.0f} mdpl) secara alami menurunkan baseline suhu."
        else:
            elev_val_txt = "🏖️ Dataran Rendah"
            elev_exp = f"Ketinggian rendah ({elev_val:.0f} mdpl) cenderung menahan suhu dasar lingkungan yang lebih hangat."

        rows = [
            ("Suhu Permukaan (LST)", lst_val_txt, lst_exp),
            ("Kelembaban Tanah", sm_val_txt, sm_exp),
            ("Kerapatan Vegetasi (NDVI)", ndvi_val_txt, ndvi_exp),
            ("Elevasi Wilayah", elev_val_txt, elev_exp),
        ]

        xai_html = '<div style="background: rgba(15,23,42,0.6); border: 1px solid rgba(148,163,184,0.15); border-radius: 12px; padding: 20px; margin-bottom: 12px;">'
        for label, status, detail in rows:
            xai_html += (
                '<div style="display: flex; flex-direction: column; align-items: flex-start; padding: 10px 0; border-bottom: 1px solid rgba(148,163,184,0.1);">'
                '<div style="display: flex; justify-content: space-between; width: 100%; font-weight: 700;">'
                f'<span style="color: #94a3b8; font-size: 0.88rem;">{label}</span>'
                f'<span style="color: #f1f5f9; font-size: 0.88rem;">{status}</span>'
                '</div>'
                f'<div style="font-size: 0.82rem; color: #64748b; margin-top: 4px;">{detail}</div>'
                '</div>'
            )
        xai_html += "</div>"
        st.markdown(xai_html, unsafe_allow_html=True)


def main() -> None:
    """Entry point halaman Dashboard."""
    render_header()

    # 1. Tambahkan selektor model di sidebar bagian paling atas
    model_pilihan = st.sidebar.selectbox(
        "🤖 Model AI Utama",
        options=["ERA5 ANFIS-LSTM", "MODIS ANFIS-LSTM"],
        index=0,
        help="Pilih model sensor utama yang ingin divisualisasikan."
    )
    
    # 2. Tentukan nama worksheet Google Sheets secara dinamis
    sheet_name = "daily_data" if model_pilihan == "ERA5 ANFIS-LSTM" else "daily_data_modis"

    # 3. Ambil data dengan memberikan parameter sheet_name
    with st.spinner(f"⏳ Memuat data {model_pilihan} dari Google Sheets..."):
        try:
            df_raw = get_daily_data(sheet_name=sheet_name)
            df_pred = get_predictions(sheet_name=sheet_name)
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
    render_kpi(df_filtered, tgl_mulai, tgl_akhir)
    st.markdown("")

    render_prediction_cards(df_pred, lokasi)
    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    render_dashboard_explanation(df_filtered, df_pred, lokasi)
    st.markdown("")

    if "horizon" not in st.session_state:
        st.session_state.horizon = "H1"

    st.session_state.horizon = st.selectbox(
        "📍 Horizon Prediksi untuk Peta",
        ["Hari ini", "H+2", "H+6"],
        index=0
    )

    col_map, col_trend = st.columns([1, 1], gap="medium")
    with col_map:
        render_gis_map(df_filtered, df_pred, SAMPLE_GEOJSON, lokasi)
    with col_trend:
        render_trend_chart(df_filtered, metrik)

    render_raw_data(df_filtered)


if __name__ == "__main__" or True:
    main()
