"""
AgriScope – Smart Agriculture Prediction System
Streamlit Dashboard – 2-Page Application
Run with: streamlit run app/app.py
"""

import os, sys, json, warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from utils.weather_api import get_weather, DISTRICT_COORDS
from utils.prediction import predict
from database.database import save_prediction, fetch_predictions, DB_PATH

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriScope – Smart Agriculture",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── AGRICULTURAL THEME CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Main background – clean light cream */
.stApp {
    background: linear-gradient(160deg, #f0f7ec 0%, #e8f4e2 40%, #f5faf2 70%, #eef7e8 100%);
    color: #1a3a0a;
}

/* Sidebar – rich dark forest (kept for contrast) */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d2204 0%, #122a06 60%, #162f07 100%);
    border-right: 2px solid #3a7010;
}
[data-testid="stSidebar"] * { color: #b8df7a !important; }
[data-testid="stSidebar"] .stRadio label {
    font-size: 1.05rem; font-weight: 600;
    padding: 8px 0; display: block;
}

/* Headings */
h1, h2, h3 { font-family: 'Poppins', sans-serif !important; color: #1e5c0a !important; font-weight: 700; }
h1 { font-size: 2.8rem !important; }
h2 { font-size: 1.8rem !important; }
h3 { font-size: 1.3rem !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #ffffff, #f2fbed);
    border: 1px solid #b2d98a;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 4px 18px rgba(60,120,20,0.10), 0 1px 4px rgba(0,0,0,0.06);
}
[data-testid="metric-container"] label { color: #4a8020 !important; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #1e5c0a !important; font-size: 1.9rem !important; font-weight: 800; font-family: 'Poppins', sans-serif;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] { color: #b07a10 !important; font-size: 0.8rem !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #3a8f1a, #2d7010) !important;
    color: #f0ffd6 !important;
    border: 1px solid #5ab830 !important;
    border-radius: 12px !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    padding: 0.65rem 2rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 18px rgba(58,143,26,0.25) !important;
    font-family: 'Poppins', sans-serif !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(58,143,26,0.40) !important;
    background: linear-gradient(135deg, #4aaa22, #3a8f1a) !important;
}

/* Selectboxes / Inputs */
.stSelectbox > div > div,
.stTextInput > div > div > input {
    background-color: #ffffff !important;
    color: #1a3a0a !important;
    border: 1px solid #8dc868 !important;
    border-radius: 10px !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff !important;
    border-radius: 12px !important;
    padding: 6px !important;
    gap: 4px !important;
    border: 1px solid #b2d98a !important;
    box-shadow: 0 2px 8px rgba(60,120,20,0.08) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #3a7a10 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Poppins', sans-serif !important;
    padding: 8px 18px !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #2d7010, #3a8f1a) !important;
    color: #f0ffd6 !important;
    box-shadow: 0 2px 12px rgba(58,143,26,0.30) !important;
}

/* Alerts / Info boxes */
.stSuccess { background: linear-gradient(135deg, #e6f9ed, #d4f5e0) !important; border-radius: 10px !important; border-left: 3px solid #27a854 !important; color: #145a2e !important; }
.stInfo    { background: linear-gradient(135deg, #e8f4fb, #d0eaf8) !important; border-radius: 10px !important; border-left: 3px solid #2980b9 !important; color: #1a3c5e !important; }
.stWarning { background: linear-gradient(135deg, #fef9e7, #fdeece) !important; border-radius: 10px !important; border-left: 3px solid #d4a010 !important; color: #5a4000 !important; }
.stError   { background: linear-gradient(135deg, #fdecea, #fbd7d5) !important; border-radius: 10px !important; color: #7a1010 !important; }

/* Light result card */
.result-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(230,248,215,0.95));
    border-radius: 20px;
    padding: 32px 24px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(58,143,26,0.12), 0 2px 8px rgba(0,0,0,0.06);
    border: 1px solid rgba(122,184,64,0.40);
    animation: fadeSlideUp 0.5s ease-out;
}
.result-card h3 { color: #3a8f1a !important; font-size: 1rem; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.08em; }
.result-card .value { color: #1e5c0a; font-size: 2.4rem; font-weight: 800; font-family: 'Poppins', sans-serif; }

/* Weather card */
.weather-card {
    background: linear-gradient(135deg, #ffffff, #f0fae8);
    border-radius: 14px;
    padding: 20px 14px;
    text-align: center;
    border: 1px solid rgba(140,200,90,0.50);
    box-shadow: 0 4px 14px rgba(60,120,20,0.08);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    height: 100%;
}
.weather-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(60,120,20,0.14); }
.weather-card .icon { font-size: 2.2rem; }
.weather-card .label { color: #4a8020; font-size: 0.82rem; font-weight: 600; margin-top: 6px; text-transform: uppercase; letter-spacing: 0.04em; }
.weather-card .val { color: #1e5c0a; font-size: 1.6rem; font-weight: 700; font-family: 'Poppins', sans-serif; }

/* Stat card – for project details */
.stat-card {
    background: linear-gradient(135deg, #ffffff, #f4faf0);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(140,200,90,0.40);
    box-shadow: 0 4px 14px rgba(60,120,20,0.07);
    margin-bottom: 12px;
}
.stat-card h4 { color: #b07a10; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 4px; }
.stat-card p  { color: #1e5c0a; font-size: 1.1rem; font-weight: 600; margin: 0; }

/* Banner */
.hero-banner {
    background: linear-gradient(135deg, rgba(255,255,255,0.97), rgba(230,248,215,0.92), rgba(255,250,230,0.88));
    border-radius: 24px;
    padding: 36px 40px;
    border: 1px solid rgba(140,200,90,0.40);
    box-shadow: 0 8px 40px rgba(58,143,26,0.10), 0 2px 8px rgba(0,0,0,0.05);
    margin-bottom: 24px;
}

/* Tech badge */
.tech-badge {
    display: inline-block;
    background: linear-gradient(135deg, #e8f7d8, #d4f0b8);
    color: #2d6e10;
    border: 1px solid rgba(140,200,90,0.50);
    border-radius: 20px;
    padding: 6px 16px;
    font-size: 0.85rem;
    font-weight: 600;
    margin: 4px;
}

/* DataFrame */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* Divider */
hr { border-color: #c8e8a0 !important; margin: 1.5rem 0 !important; }

/* Footer */
.footer { color: #5a8c20; font-size: 0.78rem; text-align: center; margin-top: 24px; line-height: 1.8; }

@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ─────────────────────────────────────────────────────────────────
GUJARAT_DISTRICTS = [
    "Ahmedabad", "Vadodara", "Surat", "Rajkot", "Bhavnagar",
    "Junagadh", "Jamnagar", "Mehsana", "Kutch", "Banaskantha",
    "Patan", "Sabarkantha", "Aravalli", "Gandhinagar", "Surendranagar",
    "Morbi", "Devbhumi Dwarka", "Porbandar", "Gir Somnath", "Amreli",
    "Botad", "Anand", "Kheda", "Panchmahal", "Mahisagar",
    "Dahod", "Chhota Udaipur", "Narmada", "Bharuch", "Navsari",
    "Valsad", "Tapi",
]
SEASONS       = ["Monsoon", "Winter", "Summer"]
DATA_PATH     = os.path.join(BASE_DIR, "data", "final_data.csv")
RAINFALL_PATH = os.path.join(BASE_DIR, "data", "ANNUAL_AVERAGE_RAINFALL_2.csv")
CLEANED_PATH  = os.path.join(BASE_DIR, "cleaned_data", "cleaned_data.csv")
METRICS_PATH  = os.path.join(BASE_DIR, "models", "metrics.json")

# ─── DATA LOADERS ──────────────────────────────────────────────────────────────
@st.cache_data
def load_main_data():
    if os.path.exists(CLEANED_PATH):
        return pd.read_csv(CLEANED_PATH)
    elif os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()

@st.cache_data
def load_rainfall_data():
    if os.path.exists(RAINFALL_PATH):
        return pd.read_csv(RAINFALL_PATH)
    return pd.DataFrame()

@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

# ─── MATPLOTLIB STYLE ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#f5faf2",
    "axes.facecolor":   "#ffffff",
    "axes.edgecolor":   "#b2d98a",
    "axes.labelcolor":  "#2d6e10",
    "xtick.color":      "#3a6a10",
    "ytick.color":      "#3a6a10",
    "text.color":       "#1a3a0a",
    "grid.color":       "#d8eebc",
    "grid.linestyle":   "--",
    "grid.alpha":       0.7,
    "font.family":      "DejaVu Sans",
})
PALETTE = ["#3a8f1a","#5ab830","#c89010","#2070b8","#c04020",
           "#8030a8","#10a880","#d06810","#a02020","#18a840"]

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:28px 0 16px 0;'>
        <div style='font-size:3.6rem; animation: pulse 3s infinite;'>🌾</div>
        <div style='font-size:1.6rem; font-weight:800; color:#c8e88a;
                    font-family:"Poppins",sans-serif; margin-top:8px;'>AgriScope</div>
        <div style='font-size:0.82rem; color:#8dc868; margin-top:4px;
                    letter-spacing:0.06em; text-transform:uppercase;'>Smart Agriculture AI</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🌾 Dashboard", "📖 Project Details"],
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Quick stats in sidebar
    mdata = load_metrics()
    if mdata:
        best = mdata.get("best_model","–")
        acc  = mdata.get("models",{}).get(best,{}).get("Accuracy", 0)
        st.markdown(f"""
        <div style='text-align:center; padding:12px;'>
            <div style='color:#c8a830; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em;'>Best Model</div>
            <div style='color:#c8e88a; font-size:1.1rem; font-weight:700;'>{best}</div>
            <div style='color:#a0cc60; font-size:0.9rem;'>{acc:.1f}% Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='footer'>
        Gujarat Agricultural Data<br>2016 – 2024<br>Powered by Open-Meteo API<br>
        <span style='color:#3a6010;'>AgriScope v2.0</span>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🌾 Dashboard":
    # Hero banner
    mdata = load_metrics()
    best_name = mdata.get("best_model", "ExtraTrees")
    best_acc  = mdata.get("models", {}).get(best_name, {}).get("Accuracy", 67.27)

    st.markdown(f"""
    <div class='hero-banner'>
        <div style='display:flex; align-items:center; gap:20px;'>
            <div style='font-size:4rem;'>🌾</div>
            <div>
                <div style='font-size:2.4rem; font-weight:800; color:#1e5c0a;
                            font-family:"Poppins",sans-serif; line-height:1.1;'>AgriScope</div>
                <div style='font-size:1rem; color:#3a7a10; margin-top:6px;'>
                    Smart Agriculture Decision Support System for <b style='color:#b07a10;'>Gujarat, India</b>
                </div>
                <div style='margin-top:12px; font-size:0.88rem; color:#4a8a18;'>
                    🌱 AI-powered crop yield prediction &nbsp;|&nbsp; 🌧️ Rainfall analytics &nbsp;|&nbsp;
                    📊 District-level insights &nbsp;|&nbsp; 🤖 {best_name} ({best_acc:.1f}% accuracy)
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Districts", "32", delta="All Gujarat" )
    with col2: st.metric("Crop Types", "6+", delta="Major crops")
    with col3: st.metric("Data Years", "9", delta="2016–2024")
    with col4: st.metric("Best Model", best_name, delta="Top performer")
    with col5: st.metric("Accuracy", f"{best_acc:.1f}%", delta="R²×100")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 5 TABS ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Overview", "🌱 Crop Prediction",
        "📊 Analytics", "🌧️ Rainfall", "📋 History"
    ])

    # ── TAB 1: OVERVIEW ─────────────────────────────────────────────────────
    with tab1:
        col_a, col_b = st.columns([1.5, 1])
        with col_a:
            st.markdown("## 🌿 What is AgriScope?")
            st.markdown("""
**AgriScope** is an AI-powered agricultural decision-support system built specifically for **Gujarat, India**.
It uses historical crop production data, rainfall statistics, and **real-time weather data** to:

- 🌿 **Predict the most suitable crop** for a district based on current conditions
- 📈 **Forecast expected crop yield** (kg/ha) using trained ML models
- 🌧️ **Analyse rainfall trends** across all Gujarat districts (2014–2024)
- 📊 **Provide district-level analytics** on area, production, and seasonal patterns
- 🗄️ **Log every prediction** for historical review and export

The system helps **farmers & agriculture planners** make data-driven decisions for optimal crop selection.
            """)
        with col_b:
            st.markdown("## 🔄 Workflow")
            st.markdown("""
```
📁 Historical Datasets (GSDMA & IMD)
           ↓
🧹 Data Cleaning & EDA
           ↓
🤖 ML Model Training (9 models)
           ↓
💾 Best Model Saved (ExtraTrees)
           ↓
🌐 Real-time Weather API
           ↓
🎯 Prediction Engine
           ↓
📊 Dashboard Results
           ↓
🗄️ SQLite Database Storage
```
            """)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("## 📌 Getting Started")
        st.info(
            "**Step 1:** Open `notebooks/AgriScope_Model_Training.ipynb` and run all cells to train & save models  \n"
            "**Step 2:** Navigate to the **🌱 Crop Prediction** tab and select a district + season  \n"
            "**Step 3:** Click **Predict** to get AI-powered crop and yield recommendations!"
        )

        # Model summary table right on overview
        if mdata:
            st.markdown("## 🏆 Model Performance Snapshot")
            models_dict = mdata.get("models", {})
            rows = []
            for name, m in models_dict.items():
                crown = " 🏆" if name == best_name else ""
                rows.append({"Model": name + crown,
                              "Accuracy (%)": m.get("Accuracy", 0),
                              "R² Score": m.get("R2", 0),
                              "MAE (kg/ha)": m.get("MAE", 0)})
            df_snap = pd.DataFrame(rows).sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)
            st.dataframe(
                df_snap.style.format({"Accuracy (%)": "{:.2f}%", "R² Score": "{:.4f}", "MAE (kg/ha)": "{:,.1f}"})
                             .background_gradient(cmap="YlGn", subset=["Accuracy (%)"]),
                use_container_width=True, height=280
            )

    # ── TAB 2: CROP PREDICTION ───────────────────────────────────────────────
    with tab2:
        st.markdown("## 🌱 Crop Prediction")
        st.markdown("Select a district and season to get AI-powered crop and yield predictions.")

        col1, col2 = st.columns(2)
        with col1:
            district = st.selectbox("📍 District", GUJARAT_DISTRICTS, index=0, key="pred_district")
        with col2:
            season = st.selectbox("📅 Season", SEASONS, index=0, key="pred_season")

        st.markdown("### 🌤️ Live Weather Data")
        weather_placeholder = st.empty()
        predict_btn = st.button("🔮 Predict Crop & Yield", use_container_width=True)

        if predict_btn:
            with st.spinner("Fetching real-time weather data…"):
                weather = get_weather(district)

            with weather_placeholder.container():
                wc1, wc2, wc3, wc4 = st.columns(4)
                weather_items = [
                    (wc1, "🌡️", "Temperature", f"{weather['temperature']} °C"),
                    (wc2, "💧", "Humidity",    f"{weather['humidity']} %"),
                    (wc3, "🌧️", "Rainfall",   f"{weather['rainfall']} mm"),
                    (wc4, "💨", "Wind Speed",  f"{weather['wind_speed']} km/h"),
                ]
                for col, icon, label, val in weather_items:
                    with col:
                        st.markdown(f"""
                        <div class='weather-card'>
                            <div class='icon'>{icon}</div>
                            <div class='label'>{label}</div>
                            <div class='val'>{val}</div>
                        </div>
                        """, unsafe_allow_html=True)

            st.markdown("### 🎯 Prediction Results")
            total_rainfall = weather["rainfall"] * 90 + 200
            rainy_days     = max(30, int(total_rainfall / 8))
            avg_tmax       = weather["temperature"] + 4.0
            avg_tmin       = weather["temperature"] - 5.0
            avg_humidity   = weather["humidity"]

            with st.spinner("Running ML model…"):
                result = predict(
                    district=district, season=season,
                    total_rainfall=round(total_rainfall, 1),
                    rainy_days=rainy_days,
                    avg_tmax=round(avg_tmax, 1),
                    avg_tmin=round(avg_tmin, 1),
                    avg_humidity=round(avg_humidity, 1),
                )

            res1, res2 = st.columns(2)
            with res1:
                st.markdown(f"""
                <div class='result-card'>
                    <h3>🌿 Recommended Crop</h3>
                    <div class='value'>{result['predicted_crop']}</div>
                </div>
                """, unsafe_allow_html=True)
            with res2:
                st.markdown(f"""
                <div class='result-card'>
                    <h3>📈 Expected Yield</h3>
                    <div class='value'>{result['predicted_yield']:,.1f} <span style='font-size:1.1rem'>kg/ha</span></div>
                </div>
                """, unsafe_allow_html=True)

            try:
                pred_id = save_prediction(
                    district=district, season=season,
                    temperature=weather["temperature"], humidity=weather["humidity"],
                    rainfall=weather["rainfall"],
                    predicted_crop=result["predicted_crop"],
                    predicted_yield=result["predicted_yield"],
                    db_path=DB_PATH,
                )
                st.success(f"✅ Prediction saved to database (ID: {pred_id})")
            except Exception as e:
                st.warning(f"⚠️ Could not save to database: {e}")

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("#### 📋 Input Summary")
            st.info(
                f"**District:** {district}  |  **Season:** {season}  |  "
                f"**Est. Rainfall:** {total_rainfall:.0f} mm  |  "
                f"**Rainy Days:** {rainy_days}  |  "
                f"**Tmax:** {avg_tmax:.1f}°C  |  **Tmin:** {avg_tmin:.1f}°C  |  "
                f"**Humidity:** {avg_humidity:.1f}%"
            )

    # ── TAB 3: ANALYTICS ────────────────────────────────────────────────────
    with tab3:
        st.markdown("## 📊 Agricultural Analytics")
        df = load_main_data()
        if df.empty:
            st.error("No data found. Please run data cleaning first.")
        else:
            df.columns = (df.columns.str.strip().str.lower()
                          .str.replace(" ", "_", regex=False)
                          .str.replace(r"[^a-z0-9_]", "", regex=True))

            # Chart 1: Crop Distribution
            st.markdown("### 🌿 Crop Type Distribution")
            if "crop_type" in df.columns:
                crop_counts = df["crop_type"].value_counts().head(10)
                fig1, ax1 = plt.subplots(figsize=(10, 4))
                colors1 = [PALETTE[i % len(PALETTE)] for i in range(len(crop_counts))]
                bars = ax1.barh(crop_counts.index[::-1], crop_counts.values[::-1], color=colors1)
                ax1.set_xlabel("Number of Records"); ax1.set_title("Top 10 Crop Types in Dataset")
                ax1.grid(axis="x")
                for bar in bars:
                    ax1.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                             f"{int(bar.get_width())}", va="center", fontsize=9, color="#a8d56a")
                plt.tight_layout(); st.pyplot(fig1); plt.close(fig1)

            st.markdown("<hr>", unsafe_allow_html=True)
            col_a, col_b = st.columns(2)

            # Chart 2: Rainfall vs Yield
            with col_a:
                st.markdown("### 🌧️ Rainfall vs Yield")
                rf_col  = next((c for c in df.columns if "rainfall" in c and "total" in c), None)
                yld_col = next((c for c in df.columns if c == "yield"), None)
                if rf_col and yld_col:
                    sample = df[[rf_col, yld_col]].dropna()
                    sample = sample[(sample[yld_col] > 0) & (sample[yld_col] < sample[yld_col].quantile(0.99))]
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    ax2.scatter(sample[rf_col], sample[yld_col], alpha=0.4, s=15, c=PALETTE[1], edgecolors="none")
                    z = np.polyfit(sample[rf_col].dropna(), sample[yld_col].dropna(), 1)
                    xr = np.linspace(sample[rf_col].min(), sample[rf_col].max(), 100)
                    ax2.plot(xr, np.poly1d(z)(xr), color=PALETTE[2], lw=2, label="Trend")
                    ax2.set_xlabel("Total Rainfall (mm)"); ax2.set_ylabel("Yield (kg/ha)")
                    ax2.set_title("Rainfall vs Yield"); ax2.legend(); ax2.grid(True)
                    plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)
                else:
                    st.info("Rainfall / Yield columns not found in dataset.")

            # Chart 3: Season vs Yield
            with col_b:
                st.markdown("### 📅 Season vs Average Yield")
                season_col = next((c for c in df.columns if "season" in c and "encoded" not in c), None)
                yld_col = next((c for c in df.columns if c == "yield"), None)
                if season_col and yld_col:
                    sv = df.groupby(season_col)[yld_col].mean().dropna().sort_values(ascending=False)
                    fig3, ax3 = plt.subplots(figsize=(6, 4))
                    ax3.bar(sv.index.astype(str), sv.values, color=PALETTE[:len(sv)], edgecolor="none", width=0.5)
                    ax3.set_xlabel("Season"); ax3.set_ylabel("Average Yield (kg/ha)")
                    ax3.set_title("Average Yield by Season"); ax3.grid(axis="y")
                    plt.xticks(rotation=20, ha="right"); plt.tight_layout(); st.pyplot(fig3); plt.close(fig3)
                else:
                    st.info("Season / Yield columns not found.")

            st.markdown("<hr>", unsafe_allow_html=True)

            # Chart 4: District vs Production
            st.markdown("### 🗺️ Top Districts by Total Production")
            dist_col = next((c for c in df.columns if c == "district"), None)
            prod_col = next((c for c in df.columns if "production" in c), None)
            if dist_col and prod_col:
                dp = df.groupby(dist_col)[prod_col].sum().sort_values(ascending=False).head(15)
                fig4, ax4 = plt.subplots(figsize=(12, 5))
                ax4.bar(dp.index, dp.values, color=[PALETTE[i % len(PALETTE)] for i in range(len(dp))], edgecolor="none")
                ax4.set_xlabel("District"); ax4.set_ylabel("Total Production")
                ax4.set_title("Top 15 Districts by Total Production")
                ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
                ax4.grid(axis="y"); plt.xticks(rotation=35, ha="right")
                plt.tight_layout(); st.pyplot(fig4); plt.close(fig4)

    # ── TAB 4: RAINFALL ─────────────────────────────────────────────────────
    with tab4:
        st.markdown("## 🌧️ Rainfall Analysis")
        st.markdown("District-wise and year-wise rainfall trends across Gujarat (2014–2024).")

        rf_df = load_rainfall_data()
        if rf_df.empty:
            st.error("Rainfall dataset not found. Ensure ANNUAL_AVERAGE_RAINFALL_2.csv is in the data/ folder.")
        else:
            rf_df.columns = rf_df.columns.str.strip()
            dist_col  = rf_df.columns[1]
            year_cols = [c for c in rf_df.columns if any(str(y) in c for y in range(2014, 2025))]
            rf_df = rf_df[rf_df[dist_col].str.lower() != "total"].copy()
            rf_melt = rf_df.melt(id_vars=[dist_col], value_vars=year_cols, var_name="Year_Label", value_name="Rainfall")
            rf_melt["Year"] = rf_melt["Year_Label"].str.extract(r"(\d{4})").astype(int)
            rf_melt["Rainfall"] = pd.to_numeric(rf_melt["Rainfall"], errors="coerce")

            st.markdown("### 📈 State-wide Average Annual Rainfall")
            yearly_avg = rf_melt.groupby("Year")["Rainfall"].mean().reset_index()
            fig_t, ax_t = plt.subplots(figsize=(12, 4))
            ax_t.plot(yearly_avg["Year"], yearly_avg["Rainfall"], marker="o", color=PALETTE[0], lw=2.5, markersize=8)
            ax_t.fill_between(yearly_avg["Year"], yearly_avg["Rainfall"], alpha=0.18, color=PALETTE[0])
            ax_t.set_xlabel("Year"); ax_t.set_ylabel("Avg Rainfall (mm)")
            ax_t.set_title("Gujarat – Average Annual Rainfall (2014–2024)"); ax_t.grid(True)
            for _, row in yearly_avg.iterrows():
                ax_t.annotate(f"{row['Rainfall']:.0f}", (row["Year"], row["Rainfall"]),
                              textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.5, color="#a8d56a")
            plt.tight_layout(); st.pyplot(fig_t); plt.close(fig_t)

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("### 🗺️ District Rainfall Comparison (Latest Year)")
            latest_col = year_cols[-1]
            rf_latest = rf_df[[dist_col, latest_col]].copy()
            rf_latest.columns = ["District", "Rainfall"]
            rf_latest["Rainfall"] = pd.to_numeric(rf_latest["Rainfall"], errors="coerce")
            rf_latest = rf_latest.dropna().sort_values("Rainfall", ascending=True)
            fig_dc, ax_dc = plt.subplots(figsize=(12, 7))
            colors_dc = plt.cm.YlGn(np.linspace(0.3, 0.9, len(rf_latest)))
            bars = ax_dc.barh(rf_latest["District"], rf_latest["Rainfall"], color=colors_dc, edgecolor="none")
            ax_dc.set_xlabel("Rainfall (mm)"); ax_dc.set_title(f"District-wise Rainfall – {latest_col}"); ax_dc.grid(axis="x")
            for bar in bars:
                ax_dc.text(bar.get_width() + 15, bar.get_y() + bar.get_height()/2,
                           f"{bar.get_width():.0f}", va="center", fontsize=8, color="#a8d56a")
            plt.tight_layout(); st.pyplot(fig_dc); plt.close(fig_dc)

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("### 📊 District-level Yearly Trend")
            selected_dist = st.selectbox("Select District", sorted(rf_df[dist_col].unique()), key="rf_district")
            dist_data = rf_melt[rf_melt[dist_col] == selected_dist].sort_values("Year")
            fig_dd, ax_dd = plt.subplots(figsize=(10, 4))
            ax_dd.bar(dist_data["Year"], dist_data["Rainfall"], color=PALETTE[3], edgecolor="none", width=0.6)
            ax_dd.plot(dist_data["Year"], dist_data["Rainfall"], marker="D", color=PALETTE[0], lw=2, markersize=7, zorder=5)
            ax_dd.set_xlabel("Year"); ax_dd.set_ylabel("Rainfall (mm)")
            ax_dd.set_title(f"{selected_dist} – Annual Rainfall Trend"); ax_dd.grid(axis="y")
            plt.xticks(dist_data["Year"].astype(int)); plt.tight_layout(); st.pyplot(fig_dd); plt.close(fig_dd)

            st.markdown("#### 📋 Raw Rainfall Data")
            st.dataframe(rf_df.set_index(dist_col)[year_cols].rename(columns={c: c.strip() for c in year_cols})
                         .style.background_gradient(cmap="YlGn", axis=1), use_container_width=True)

    # ── TAB 5: HISTORY ──────────────────────────────────────────────────────
    with tab5:
        st.markdown("## 📋 Prediction History")
        st.markdown("All past predictions stored in the SQLite database.")

        try:
            records = fetch_predictions(limit=200, db_path=DB_PATH)
            if not records:
                st.info("No predictions yet. Make a prediction in the **🌱 Crop Prediction** tab first.")
            else:
                hist_df = pd.DataFrame(records)
                hist_df = hist_df[["id","timestamp","district","season",
                                   "temperature","humidity","rainfall",
                                   "predicted_crop","predicted_yield"]]
                hist_df.columns = ["ID","Timestamp","District","Season",
                                   "Temp (°C)","Humidity (%)","Rainfall (mm)",
                                   "Predicted Crop","Predicted Yield (kg/ha)"]

                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Total Predictions", len(hist_df))
                with c2: st.metric("Unique Districts", hist_df["District"].nunique())
                with c3: st.metric("Avg Predicted Yield", f"{hist_df['Predicted Yield (kg/ha)'].mean():,.1f} kg/ha")

                st.markdown("<hr>", unsafe_allow_html=True)
                st.dataframe(hist_df, use_container_width=True, height=380)
                st.download_button("⬇️ Download as CSV",
                                   hist_df.to_csv(index=False).encode("utf-8"),
                                   "agriscope_history.csv", "text/csv",
                                   use_container_width=True)

                if len(hist_df) >= 2:
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.markdown("### 📊 Predictions by District")
                    dist_counts = hist_df["District"].value_counts().head(12)
                    fig_h, ax_h = plt.subplots(figsize=(10, 4))
                    ax_h.bar(dist_counts.index, dist_counts.values,
                             color=[PALETTE[i % len(PALETTE)] for i in range(len(dist_counts))], edgecolor="none")
                    ax_h.set_xlabel("District"); ax_h.set_ylabel("Prediction Count")
                    ax_h.set_title("Prediction Count by District"); ax_h.grid(axis="y")
                    plt.xticks(rotation=30, ha="right"); plt.tight_layout(); st.pyplot(fig_h); plt.close(fig_h)

        except Exception as e:
            st.error(f"Database error: {e}")
            st.info("If this is your first run, make a prediction on the **🌱 Crop Prediction** tab.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – PROJECT DETAILS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📖 Project Details":
    st.markdown("""
    <div class='hero-banner'>
        <div style='display:flex; align-items:center; gap:20px;'>
            <div style='font-size:3.8rem;'>📖</div>
            <div>
                <div style='font-size:2.2rem; font-weight:800; color:#1e5c0a;
                            font-family:"Poppins",sans-serif;'>Project Details</div>
                <div style='font-size:0.95rem; color:#3a7a10; margin-top:6px;'>
                    Architecture · ML Models · Technology Stack · Dataset
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section 1: About ────────────────────────────────────────────────────
    st.markdown("## 🎯 Project Overview")
    col_a, col_b = st.columns([1.4, 1])
    with col_a:
        st.markdown("""
**AgriScope** is a complete end-to-end machine learning web application that helps
farmers and agriculture planners in **Gujarat, India** make data-driven decisions.

**Problem Statement:**
Gujarat faces significant variability in agricultural yields due to irregular rainfall,
seasonal temperature swings, and improper crop selection. Farmers lack access to
data-driven tools for predicting which crops will perform best given current conditions.

**Solution:**
AgriScope integrates:
- Historical crop yield data (2016–2024) across 32 Gujarat districts
- Annual rainfall datasets (GSDMA / IMD)
- Real-time weather data via Open-Meteo API
- Trained ML regression models that predict expected yield in kg/ha
- A crop recommendation heuristic based on district geography and season
        """)
    with col_b:
        st.markdown("### 📊 Dataset Facts")
        facts = [
            ("Districts Covered", "32 (all Gujarat)"),
            ("Total Records", "970 samples (after cleaning)"),
            ("Years Covered", "2016 – 2024"),
            ("Seasons", "Monsoon, Winter, Summer"),
            ("Target Variable", "Crop Yield (kg/ha)"),
            ("Features Used", "8 input features"),
        ]
        for label, val in facts:
            st.markdown(f"""
            <div class='stat-card'>
                <h4>{label}</h4>
                <p>{val}</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Section 2: Tech Stack ────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## 🛠️ Technology Stack")
    tech_items = [
        ("🐍", "Python 3.10", "Core language"),
        ("📊", "Streamlit", "Web dashboard"),
        ("🤖", "scikit-learn", "ML models"),
        ("⚡", "XGBoost", "Gradient boosting"),
        ("🌲", "LightGBM", "LGBM model"),
        ("🐼", "Pandas / NumPy", "Data processing"),
        ("📈", "Matplotlib", "Visualizations"),
        ("🌐", "Open-Meteo API", "Live weather data"),
        ("🗄️", "SQLite", "Prediction storage"),
        ("📓", "Jupyter", "Model training"),
    ]
    cols = st.columns(5)
    for i, (icon, name, desc) in enumerate(tech_items):
        with cols[i % 5]:
            st.markdown(f"""
            <div style='background:#ffffff; border:1px solid rgba(140,200,90,0.45);
                        border-radius:14px; padding:18px 12px; text-align:center;
                        box-shadow:0 4px 12px rgba(60,120,20,0.08); margin-bottom:12px;'>
                <div style='font-size:2rem;'>{icon}</div>
                <div style='color:#1e5c0a; font-weight:700; font-size:0.9rem;
                            font-family:"Poppins",sans-serif; margin-top:6px;'>{name}</div>
                <div style='color:#5a8c20; font-size:0.75rem; margin-top:4px;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Section 3: Feature Engineering ───────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## ⚙️ Features Used for ML Training")
    mdata = load_metrics()
    features = mdata.get("features", [
        "district_encoded", "season_encoded", "crop_type_encoded",
        "total_rainfall", "rainy_days", "average_tmax", "average_tmin", "average_humidity"
    ])
    feature_desc = {
        "district_encoded":   "District label-encoded (0–31)",
        "season_encoded":     "Season label-encoded (Monsoon/Winter/Summer)",
        "crop_type_encoded":  "Crop type label-encoded (KEY feature – different crops have very different yields)",
        "total_rainfall":     "Total seasonal rainfall in mm",
        "rainy_days":         "Number of rainy days in the season",
        "average_tmax":       "Average daily maximum temperature (°C)",
        "average_tmin":       "Average daily minimum temperature (°C)",
        "average_humidity":   "Average relative humidity (%)",
    }
    feat_rows = [{"Feature": f, "Description": feature_desc.get(f, f)} for f in features]
    st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)
    st.info("**Log-transform applied to Yield** during training to reduce skewness and outlier impact. Predictions are inverse-transformed (expm1) back to kg/ha.")

    # ── Section 4: All ML Models ─────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## 🤖 All ML Models Tested")
    st.markdown("The following regression models were trained and evaluated on the same 80/20 train-test split:")

    if mdata:
        best_model_name = mdata.get("best_model", "ExtraTrees")
        top5            = mdata.get("top5", [])
        models_dict     = mdata.get("models", {})

        all_rows = []
        for name, m in models_dict.items():
            crown   = " 🏆" if name == best_model_name else ""
            star    = " ⭐" if name in top5 and name != best_model_name else ""
            rank_tag = crown or star
            all_rows.append({
                "Model":        name + rank_tag,
                "Accuracy (%)": m.get("Accuracy", 0),
                "R² Score":     m.get("R2", 0),
                "MAE (kg/ha)":  m.get("MAE", 0),
                "RMSE (kg/ha)": m.get("RMSE", 0),
            })
        all_df = pd.DataFrame(all_rows).sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)

        def highlight_best(row):
            if "🏆" in str(row["Model"]):
                return ["background-color:rgba(90,158,47,0.25); font-weight:800;"] * len(row)
            elif "⭐" in str(row["Model"]):
                return ["background-color:rgba(212,168,67,0.12);"] * len(row)
            return [""] * len(row)

        st.dataframe(
            all_df.style.apply(highlight_best, axis=1)
                        .format({"Accuracy (%)": "{:.2f}%", "R² Score": "{:.4f}",
                                 "MAE (kg/ha)": "{:,.1f}", "RMSE (kg/ha)": "{:,.1f}"})
                        .background_gradient(cmap="YlGn", subset=["Accuracy (%)", "R² Score"]),
            use_container_width=True, height=320
        )

        # Accuracy bar chart
        st.markdown("<hr>", unsafe_allow_html=True)
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("### 🎯 Accuracy Comparison")
            raw_names  = [r["Model"].replace(" 🏆","").replace(" ⭐","") for _, r in all_df.iterrows()]
            accs       = list(all_df["Accuracy (%)"])
            bar_colors = [PALETTE[0] if n == best_model_name else PALETTE[3] for n in raw_names]
            fig_ac, ax_ac = plt.subplots(figsize=(7, 4))
            bars_ac = ax_ac.barh(raw_names[::-1], accs[::-1], color=bar_colors[::-1], edgecolor="none", height=0.55)
            ax_ac.set_xlabel("Accuracy (%)"); ax_ac.set_title("All Models – Accuracy")
            ax_ac.set_xlim(0, max(accs + [10]) * 1.15); ax_ac.grid(axis="x")
            for bar, val in zip(bars_ac, accs[::-1]):
                ax_ac.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height()/2,
                           f"{val:.1f}%", va="center", fontsize=9, fontweight="bold", color="#a8d56a")
            plt.tight_layout(); st.pyplot(fig_ac); plt.close(fig_ac)

        with col_r:
            st.markdown("### 📉 MAE & RMSE Comparison")
            mae_vals  = [models_dict[n]["MAE"]  for n in raw_names]
            rmse_vals = [models_dict[n]["RMSE"] for n in raw_names]
            x_pos = np.arange(len(raw_names)); w = 0.35
            fig_er, ax_er = plt.subplots(figsize=(7, 4))
            ax_er.bar(x_pos - w/2, mae_vals,  w, label="MAE",  color=PALETTE[2], edgecolor="none")
            ax_er.bar(x_pos + w/2, rmse_vals, w, label="RMSE", color=PALETTE[3], edgecolor="none")
            ax_er.set_xticks(x_pos); ax_er.set_xticklabels(raw_names, rotation=30, ha="right", fontsize=8)
            ax_er.set_ylabel("Error (kg/ha)"); ax_er.set_title("MAE vs RMSE (lower is better)")
            ax_er.legend(); ax_er.grid(axis="y")
            plt.tight_layout(); st.pyplot(fig_er); plt.close(fig_er)

        # ── Best Model Call-out ──────────────────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        best_m = models_dict.get(best_model_name, {})
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,rgba(40,90,15,0.75),rgba(60,120,20,0.8));
                    border-radius:20px; padding:32px 36px;
                    border:1px solid rgba(168,213,106,0.4);
                    box-shadow:0 8px 40px rgba(0,0,0,0.5);
                    backdrop-filter:blur(12px);'>
            <div style='display:flex; align-items:center; gap:20px;'>
                <div style='font-size:3.5rem;'>🏆</div>
                <div>
                    <div style='color:#d4a843; font-size:0.82rem; text-transform:uppercase;
                                letter-spacing:0.08em; font-weight:600;'>Best Performing Model</div>
                    <div style='color:#d4f08a; font-size:2rem; font-weight:800;
                                font-family:"Poppins",sans-serif;'>{best_model_name}</div>
                    <div style='color:#a8d56a; font-size:0.95rem; margin-top:8px;'>
                        Accuracy: <b style='color:#a8ff6a; font-size:1.15rem;'>{best_m.get('Accuracy',0):.2f}%</b>
                        &nbsp;|&nbsp; R²: <b style='color:#a8ff6a;'>{best_m.get('R2',0):.4f}</b>
                        &nbsp;|&nbsp; MAE: <b style='color:#d4a843;'>{best_m.get('MAE',0):,.1f} kg/ha</b>
                        &nbsp;|&nbsp; RMSE: <b style='color:#d4a843;'>{best_m.get('RMSE',0):,.1f} kg/ha</b>
                    </div>
                </div>
            </div>
            <div style='margin-top:20px; color:#8bb94a; font-size:0.9rem; line-height:1.8;'>
                <b style='color:#c8e88a;'>Why ExtraTrees?</b> Extra Trees (Extremely Randomised Trees) builds
                many decision trees with random feature thresholds, reducing variance while maintaining bias
                comparable to Random Forest. It performs well on tabular agricultural data with non-linear
                feature interactions (crop type × district × season), and trains faster due to random
                (instead of best) split selection. With <b>log-transform on yield</b>, it captured the
                non-linear distribution of yields across Gujarat's diverse crop portfolio.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("⚠️ No metrics found. Please run `python train_model.py` first.")

    # ── Section 5: Metric Definitions ───────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## 📐 Metric Definitions")
    st.info(
        "**Accuracy %** = R² × 100 — percentage of yield variance explained by the model  \n"
        "**R² Score** = 1.0 is perfect. >0.5 is good. Negative = worse than predicting the mean  \n"
        "**MAE** = Mean Absolute Error in kg/ha — average absolute prediction gap  \n"
        "**RMSE** = Root Mean Squared Error — penalises large errors more heavily  \n"
        "**Log-Transform** was applied to yield values before training to reduce skewness (skew reduced from ~3.5 to ~0.3)"
    )

    # ── Section 6: Files & Architecture ─────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## 🗂️ Project Architecture")
    st.markdown("""
```
AgriScope/
├── app/
│   └── app.py              ← This Streamlit dashboard (2-page app)
├── data/
│   ├── final_data.csv      ← Raw Gujarat crop + weather dataset
│   └── ANNUAL_AVERAGE_RAINFALL_2.csv  ← District rainfall (2014–2024)
├── cleaned_data/
│   └── cleaned_data.csv    ← Output of data cleaning pipeline
├── database/
│   ├── database.py         ← SQLite helper (save/fetch predictions)
│   └── agriscope.db        ← SQLite database file
├── models/
│   ├── model.pkl           ← Best model (ExtraTrees)
│   ├── scaler.pkl          ← StandardScaler
│   ├── encoders.pkl        ← LabelEncoders (district/season/crop)
│   ├── transform_info.pkl  ← Log-transform flag
│   ├── metrics.json        ← All model metrics
│   └── *_model.pkl         ← All individual model files
├── notebooks/
│   └── AgriScope_Model_Training.ipynb  ← Full training notebook
├── utils/
│   ├── data_cleaning.py    ← Data cleaning pipeline
│   ├── prediction.py       ← ML prediction module
│   └── weather_api.py      ← Open-Meteo API integration
└── train_model.py          ← CLI model trainer script
```
    """)
