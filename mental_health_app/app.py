"""
Aplikasi Web Deteksi Tingkat Stres (Growing_Stress) — Mental Health Dataset
Dibangun dengan Streamlit, menggunakan model Random Forest hasil UAS Big Data.
Terhubung langsung ke pipeline & model dari UAS_BigData_Mental_Health.ipynb.
"""

import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="MindCheck | Deteksi Tingkat Stres",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS KUSTOM — TEMA BIRU & PUTIH
# ============================================================
BIRU_TUA = "#0B3D91"
BIRU_UTAMA = "#1B6DE0"
BIRU_TERANG = "#4FA3FF"
BIRU_SANGAT_MUDA = "#EAF3FF"
TEKS_GELAP = "#0F1F33"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

    .stApp {{
        background: linear-gradient(180deg, #FFFFFF 0%, {BIRU_SANGAT_MUDA} 100%) !important;
    }}
    .main .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }}

    .main, .main p, .main span, .main label, .main div {{ color: {TEKS_GELAP}; }}
    h1, h2, h3, h4, h5 {{
        font-family: 'Poppins', sans-serif;
        color: {BIRU_TUA} !important;
        font-weight: 700;
    }}

    /* ===== Sidebar ===== */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {BIRU_TUA} 0%, #06265C 100%) !important;
    }}
    section[data-testid="stSidebar"] * {{ color: #EAF3FF !important; }}
    section[data-testid="stSidebar"] .stRadio label {{ font-weight: 500; }}
    section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] div:first-child {{
        border-color: #EAF3FF !important;
    }}
    section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.15) !important; }}

    /* ===== Label widget ===== */
    label, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {{
        color: {TEKS_GELAP} !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        opacity: 1 !important;
    }}

    /* ===== Selectbox ===== */
    div[data-baseweb="select"] > div {{
        background-color: #FFFFFF !important;
        border: 1.5px solid #C9DFFB !important;
        border-radius: 10px !important;
        color: {TEKS_GELAP} !important;
    }}
    div[data-baseweb="select"] span {{ color: {TEKS_GELAP} !important; }}
    ul[data-testid="stSelectboxVirtualDropdown"] {{ background-color: #FFFFFF !important; }}
    ul[data-testid="stSelectboxVirtualDropdown"] li {{ color: {TEKS_GELAP} !important; }}
    ul[data-testid="stSelectboxVirtualDropdown"] li:hover {{ background-color: {BIRU_SANGAT_MUDA} !important; }}

    /* ===== Form container ===== */
    div[data-testid="stForm"] {{
        background: #FFFFFF;
        border-radius: 18px;
        padding: 1.8rem 2rem 1.4rem 2rem;
        border: 1px solid #DCE9FB;
        box-shadow: 0 6px 24px rgba(11, 61, 145, 0.07);
    }}

    /* ===== Hero card ===== */
    .hero-card {{
        background: linear-gradient(120deg, {BIRU_TUA} 0%, {BIRU_UTAMA} 55%, {BIRU_TERANG} 100%);
        padding: 2.6rem 2.4rem;
        border-radius: 20px;
        margin-bottom: 1.6rem;
        box-shadow: 0 12px 30px rgba(11, 61, 145, 0.25);
    }}
    .hero-card h1 {{ color: #FFFFFF !important; margin-bottom: 0.4rem; font-size: 2.1rem; }}
    .hero-card p {{ color: #EAF3FF !important; font-size: 1.02rem; margin: 0; }}

    /* ===== Info card ===== */
    .info-card {{
        background: #FFFFFF;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 4px 18px rgba(11, 61, 145, 0.06);
        border: 1px solid #DCE9FB;
        height: 100%;
    }}
    .info-card h4 {{ color: {BIRU_TUA} !important; margin-bottom: 0.8rem; }}
    .info-card p {{ color: #33465E !important; margin: 0.35rem 0; line-height: 1.55; }}

    /* ===== Metric card ===== */
    .metric-card {{
        background: #FFFFFF;
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        text-align: center;
        box-shadow: 0 4px 18px rgba(11, 61, 145, 0.06);
        border: 1px solid #DCE9FB;
    }}
    .metric-card .value {{ font-size: 1.9rem; font-weight: 700; color: {BIRU_UTAMA} !important; }}
    .metric-card .label {{ font-size: 0.85rem; color: #5A6B82 !important; margin-top: 4px; }}

    /* ===== Disclaimer box ===== */
    .disclaimer-box {{
        background: #FFF6E5;
        border-left: 5px solid #F4B740;
        border-radius: 10px;
        padding: 1rem 1.3rem;
        font-size: 0.92rem;
        margin-top: 1rem;
    }}
    .disclaimer-box, .disclaimer-box * {{ color: #6B5620 !important; }}

    /* ===== Result box ===== */
    .result-box {{
        border-radius: 18px;
        padding: 1.8rem 2rem;
        margin-top: 1rem;
        box-shadow: 0 10px 26px rgba(0,0,0,0.14);
    }}
    .result-box h2, .result-box p {{ color: #FFFFFF !important; }}
    .result-box h2 {{ margin-bottom: 0.3rem; }}

    /* ===== Section header dalam form ===== */
    .main h4 {{
        margin-top: 1.4rem;
        margin-bottom: 0.6rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid {BIRU_SANGAT_MUDA};
        color: {BIRU_UTAMA} !important;
    }}

    /* ===== Tombol ===== */
    .stButton > button, .stFormSubmitButton > button {{
        background: linear-gradient(120deg, {BIRU_TUA}, {BIRU_UTAMA}) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 1.6rem !important;
        font-weight: 600 !important;
        font-size: 1.02rem !important;
        width: 100%;
    }}
    .stButton > button:hover, .stFormSubmitButton > button:hover {{ opacity: 0.9; color: #FFFFFF !important; }}

    /* ===== Tabs ===== */
    button[data-baseweb="tab"] {{ color: #5A6B82 !important; font-weight: 600; }}
    button[data-baseweb="tab"][aria-selected="true"] {{ color: {BIRU_UTAMA} !important; }}
    div[data-baseweb="tab-highlight"] {{ background-color: {BIRU_UTAMA} !important; }}

    /* ===== Expander ===== */
    details summary {{ color: {TEKS_GELAP} !important; font-weight: 600; }}

    /* ===== Dataframe ===== */
    [data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; border: 1px solid #DCE9FB; }}

    /* ===== st.info / st.metric ===== */
    div[data-testid="stAlert"] p {{ color: {TEKS_GELAP} !important; }}
    div[data-testid="stAlert"] {{ background-color: {BIRU_SANGAT_MUDA} !important; border-radius: 10px; }}
    [data-testid="stMetricValue"] {{ color: {BIRU_UTAMA} !important; }}
    [data-testid="stMetricLabel"] {{ color: #5A6B82 !important; }}

    footer, #MainMenu {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD ARTEFAK (DI-CACHE AGAR CEPAT)
# ============================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("model_rf.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("label_encoders.pkl")
    with open("fitur_terpilih.json") as f:
        fitur = json.load(f)
    return model, scaler, encoders, fitur


@st.cache_data
def load_sample_data():
    return pd.read_csv("mental_health_sample.csv")


@st.cache_data
def load_model_metrics():
    with open("hasil_metrik_model.json") as f:
        data = json.load(f)
    return pd.DataFrame(data)


model, scaler, encoders, FITUR_TERPILIH = load_artifacts()
df_sample = load_sample_data()
df_metrik = load_model_metrics()

# Palet warna: chrome UI biru-putih, hasil prediksi tetap pakai warna semantik (hijau/kuning/merah)
# agar mudah dibaca sebagai "aman / waspada / risiko tinggi".
WARNA_TARGET = {"No": "#1B9C5A", "Maybe": "#F4B740", "Yes": "#E24C4C"}
WARNA_MODEL = [BIRU_TUA, BIRU_UTAMA, BIRU_TERANG, "#7FC4FF", "#B7DDFF"]

# ============================================================
# SIDEBAR NAVIGASI — Prediksi Stres jadi halaman pertama/default
# ============================================================
with st.sidebar:
    st.markdown("## 🧠 MindCheck")
    st.markdown("Deteksi Tingkat Stres berbasis *Machine Learning*")
    st.markdown("---")
    halaman = st.radio(
        "Navigasi",
        ["🔮 Prediksi Stres", "🏠 Beranda", "📊 Dataset & EDA", "⚖️ Perbandingan Model", "ℹ️ Tentang"],
        index=0,
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("UAS Big Data · Teknik Informatika")
    st.caption("Universitas Mercu Buana")
    st.caption("Abdullah Al Manan · 41523010025")

# ============================================================
# HALAMAN — PREDIKSI STRES (LANDING PAGE / DEFAULT)
# ============================================================
if halaman == "🔮 Prediksi Stres":
    st.markdown(f"""
    <div class="hero-card">
        <h1>🔮 Prediksi Tingkat Stres</h1>
        <p>Isi kondisi berikut sesuai keadaanmu, model Random Forest akan memprediksi
        kecenderungan peningkatan stres (Growing Stress) secara langsung.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("form_prediksi"):
        st.markdown("#### 👤 Data Diri & Pekerjaan")
        c1, c2, c3 = st.columns(3)
        gender = c1.selectbox("Gender", ["Female", "Male"])
        country = c2.selectbox("Negara", sorted(encoders["Country"].classes_))
        occupation = c3.selectbox("Pekerjaan", ["Corporate", "Student", "Business", "Housewife", "Others"])

        c1, c2, c3 = st.columns(3)
        self_employed = c1.selectbox("Wiraswasta (Self-employed)?", ["No", "Yes"])
        days_indoors = c2.selectbox(
            "Berapa lama biasanya berada di dalam rumah?",
            ["Go out Every day", "1-14 days", "15-30 days", "31-60 days", "More than 2 months"],
        )
        care_options = c3.selectbox("Tahu adanya opsi bantuan/perawatan kesehatan mental?", ["Yes", "No", "Not sure"])

        st.markdown("#### 🩺 Riwayat & Kondisi Kesehatan Mental")
        c1, c2, c3, c4 = st.columns(4)
        family_history = c1.selectbox("Riwayat keluarga?", ["No", "Yes"])
        treatment = c2.selectbox("Pernah menjalani treatment?", ["No", "Yes"])
        mental_health_history = c3.selectbox("Riwayat kesehatan mental pribadi?", ["No", "Yes", "Maybe"])
        coping_struggles = c4.selectbox("Kesulitan coping/mengatasi stres?", ["No", "Yes"])

        st.markdown("#### 🧩 Kebiasaan & Perilaku Sehari-hari")
        c1, c2, c3 = st.columns(3)
        changes_habits = c1.selectbox("Ada perubahan kebiasaan akhir-akhir ini?", ["No", "Yes", "Maybe"])
        mood_swings = c2.selectbox("Tingkat perubahan mood (mood swings)?", ["Low", "Medium", "High"])
        work_interest = c3.selectbox("Minat terhadap pekerjaan menurun?", ["No", "Yes", "Maybe"])

        c1, c2 = st.columns(2)
        social_weakness = c1.selectbox("Merasa lemah/canggung secara sosial?", ["No", "Yes", "Maybe"])
        mental_health_interview = c2.selectbox("Nyaman membahas kesehatan mental saat wawancara kerja?", ["No", "Yes", "Maybe"])

        submitted = st.form_submit_button("🔮 Prediksi Sekarang")

    if submitted:
        risk_score = sum([
            1 if family_history == "Yes" else 0,
            1 if treatment == "Yes" else 0,
            1 if mental_health_history == "Yes" else 0,
            1 if coping_struggles == "Yes" else 0,
        ])

        input_mentah = {
            "Changes_Habits": changes_habits, "Occupation": occupation, "Work_Interest": work_interest,
            "Days_Indoors": days_indoors, "Mood_Swings": mood_swings, "Coping_Struggles": coping_struggles,
            "Gender": gender, "risk_score": risk_score, "Social_Weakness": social_weakness,
            "family_history": family_history, "Mental_Health_History": mental_health_history,
            "treatment": treatment, "care_options": care_options, "Country": country,
            "self_employed": self_employed, "mental_health_interview": mental_health_interview,
        }

        baris = {}
        for kolom in FITUR_TERPILIH:
            if kolom == "risk_score":
                baris[kolom] = risk_score
            else:
                le = encoders[kolom]
                baris[kolom] = le.transform([input_mentah[kolom]])[0]

        X_input = pd.DataFrame([baris])[FITUR_TERPILIH]
        X_input_scaled = scaler.transform(X_input)

        pred_encoded = model.predict(X_input_scaled)[0]
        pred_proba = model.predict_proba(X_input_scaled)[0]

        target_le = encoders["Growing_Stress"]
        pred_label = target_le.inverse_transform([pred_encoded])[0]
        kelas_urut = target_le.classes_

        pesan_hasil = {
            "No": "Berdasarkan pola jawabanmu, model memprediksi kecenderungan stres yang meningkat "
                  "relatif rendah saat ini. Tetap jaga keseimbangan hidup dan rutinitas sehat, ya!",
            "Maybe": "Model mendeteksi indikasi campuran — ada kemungkinan level stres kamu mulai "
                     "meningkat. Ada baiknya mulai memperhatikan pola istirahat, aktivitas sosial, "
                     "dan berbicara dengan orang terdekat jika diperlukan.",
            "Yes": "Model memprediksi kecenderungan peningkatan stres yang cukup tinggi berdasarkan "
                   "pola jawabanmu. Sangat disarankan untuk berbicara dengan seseorang yang kamu "
                   "percaya atau tenaga profesional kesehatan mental.",
        }

        st.markdown(f"""
        <div class="result-box" style="background: linear-gradient(120deg, {WARNA_TARGET[pred_label]}, {WARNA_TARGET[pred_label]}CC);">
            <h2>Hasil Prediksi: Growing Stress = "{pred_label}"</h2>
            <p>{pesan_hasil[pred_label]}</p>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.markdown("#### Probabilitas Tiap Kelas")
            df_proba = pd.DataFrame({"Kelas": kelas_urut, "Probabilitas": pred_proba})
            fig = go.Figure(go.Bar(
                x=df_proba["Probabilitas"], y=df_proba["Kelas"], orientation="h",
                marker_color=[WARNA_TARGET[k] for k in df_proba["Kelas"]],
                text=[f"{p*100:.1f}%" for p in df_proba["Probabilitas"]], textposition="outside",
                textfont=dict(size=14, color=TEKS_GELAP),
            ))
            fig.update_layout(
                height=320, plot_bgcolor="white", paper_bgcolor="white",
                xaxis_range=[0, 1], xaxis_title="Probabilitas",
                font=dict(color=TEKS_GELAP, size=13),
                xaxis=dict(gridcolor="#E4EEFB"), margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("#### Ringkasan Input Risiko")
            st.markdown(f"""<div class="info-card">
            <p>🧬 <b>Riwayat keluarga:</b> {family_history}</p>
            <p>💊 <b>Pernah treatment:</b> {treatment}</p>
            <p>📋 <b>Riwayat kesehatan mental:</b> {mental_health_history}</p>
            <p>😔 <b>Kesulitan coping:</b> {coping_struggles}</p>
            <p>⭐ <b>Skor risiko gabungan:</b> {risk_score} dari 4</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="disclaimer-box">
        ⚠️ Hasil ini dihasilkan oleh model Machine Learning untuk keperluan edukatif/akademik,
        <b>bukan diagnosis medis</b>. Untuk penanganan yang tepat, silakan konsultasikan kondisimu
        dengan psikolog atau psikiater profesional.
        </div>""", unsafe_allow_html=True)

# ============================================================
# HALAMAN — BERANDA
# ============================================================
elif halaman == "🏠 Beranda":
    st.markdown("""
    <div class="hero-card">
        <h1>🧠 MindCheck — Deteksi Tingkat Stres</h1>
        <p>Aplikasi berbasis Machine Learning untuk memprediksi kecenderungan peningkatan
        stres (Growing Stress) berdasarkan pola kebiasaan, riwayat kesehatan mental, dan
        kondisi psikososial responden.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card"><div class="value">{len(df_sample):,}</div>
        <div class="label">Sampel Data Ditampilkan</div></div>""", unsafe_allow_html=True)
    with col2:
        akurasi_rf = df_metrik.loc[df_metrik['Model'] == 'Random Forest (Tuned)', 'Accuracy'].values[0]
        st.markdown(f"""<div class="metric-card"><div class="value">{akurasi_rf*100:.1f}%</div>
        <div class="label">Akurasi Model (Random Forest)</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card"><div class="value">16</div>
        <div class="label">Fitur Digunakan Model</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card"><div class="value">3</div>
        <div class="label">Kelas Prediksi</div></div>""", unsafe_allow_html=True)

    st.write("")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.markdown("""<div class="info-card">
        <h4>Tentang Aplikasi</h4>
        <p>MindCheck dikembangkan sebagai bagian dari Ujian Akhir Semester (UAS) mata kuliah
        <b>Big Data</b>, Program Studi Teknik Informatika, Universitas Mercu Buana. Aplikasi
        ini menggunakan model <b>Random Forest</b> yang dilatih pada <b>Mental Health Dataset</b>
        (±290 ribu data responden survei kesehatan mental) untuk memprediksi apakah seseorang
        cenderung mengalami peningkatan level stres (<i>Growing Stress</i>): <b>Yes</b>, <b>No</b>,
        atau <b>Maybe</b>.</p>
        <p>Seluruh pipeline preprocessing dan parameter model pada aplikasi ini <b>identik</b>
        dengan notebook <code>UAS_BigData_Mental_Health.ipynb</code> — mulai dari missing value
        handling, feature engineering, feature selection, hingga hyperparameter hasil GridSearchCV.</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="disclaimer-box">
        ⚠️ <b>Disclaimer:</b> Aplikasi ini dibuat untuk keperluan akademik/edukatif dan
        <b>bukan merupakan alat diagnosis klinis</b>. Hasil prediksi tidak menggantikan
        konsultasi dengan psikolog atau tenaga profesional kesehatan mental. Jika kamu atau
        seseorang yang kamu kenal sedang mengalami kesulitan secara emosional, sangat
        disarankan untuk berbicara dengan profesional kesehatan mental terpercaya.
        </div>""", unsafe_allow_html=True)

# ============================================================
# HALAMAN — DATASET & EDA
# ============================================================
elif halaman == "📊 Dataset & EDA":
    st.markdown("## 📊 Eksplorasi Dataset")
    st.write("Ringkasan karakteristik **Mental Health Dataset** yang digunakan untuk melatih model (sesuai notebook UAS).")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Baris (sampel)", f"{len(df_sample):,}")
    col2.metric("Total Kolom", f"{df_sample.shape[1]}")
    col3.metric("Negara Responden", f"{df_sample['Country'].nunique()}")

    st.write("")
    tab1, tab2, tab3 = st.tabs(["🎯 Distribusi Target", "🌍 Demografi", "🔗 Hubungan Antar Variabel"])

    layout_chart = dict(
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color=TEKS_GELAP, size=13),
        xaxis=dict(gridcolor="#E4EEFB", linecolor="#C9DFFB"),
        yaxis=dict(gridcolor="#E4EEFB", linecolor="#C9DFFB"),
        margin=dict(l=10, r=10, t=50, b=10),
    )

    with tab1:
        counts = df_sample["Growing_Stress"].value_counts().reindex(["No", "Maybe", "Yes"])
        fig = go.Figure(go.Bar(
            x=counts.index, y=counts.values,
            marker_color=[WARNA_TARGET[k] for k in counts.index],
            text=counts.values, textposition="outside",
            textfont=dict(size=14, color=TEKS_GELAP),
        ))
        fig.update_layout(title="Distribusi Growing_Stress", height=420,
                           yaxis_title="Jumlah Responden", xaxis_title="", **layout_chart)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            top_country = df_sample["Country"].value_counts().head(10).sort_values()
            fig = go.Figure(go.Bar(
                x=top_country.values, y=top_country.index, orientation="h",
                marker_color=BIRU_UTAMA,
                text=top_country.values, textposition="outside",
                textfont=dict(size=12, color=TEKS_GELAP),
            ))
            fig.update_layout(title="10 Negara Terbanyak", height=420, **layout_chart)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            occ = df_sample["Occupation"].value_counts()
            fig = px.pie(values=occ.values, names=occ.index, hole=0.45,
                         color_discrete_sequence=[BIRU_TUA, BIRU_UTAMA, BIRU_TERANG, "#7FC4FF", "#B7DDFF"])
            fig.update_traces(textfont=dict(size=13, color="white"))
            fig.update_layout(title="Distribusi Pekerjaan", height=420,
                               plot_bgcolor="white", paper_bgcolor="white",
                               font=dict(color=TEKS_GELAP, size=13), margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fitur_pilihan = st.selectbox(
            "Pilih variabel untuk dibandingkan dengan Growing_Stress:",
            ["family_history", "treatment", "Mental_Health_History", "Mood_Swings",
             "Changes_Habits", "Work_Interest", "Days_Indoors"],
        )
        ct = pd.crosstab(df_sample[fitur_pilihan], df_sample["Growing_Stress"], normalize="index") * 100
        ct = ct[["No", "Maybe", "Yes"]]
        fig = go.Figure()
        for kelas in ["No", "Maybe", "Yes"]:
            fig.add_trace(go.Bar(name=kelas, x=ct.index, y=ct[kelas], marker_color=WARNA_TARGET[kelas]))
        fig.update_layout(
            barmode="stack", title=f"Growing_Stress berdasarkan {fitur_pilihan}",
            height=450, yaxis_title="Persentase (%)", legend=dict(font=dict(color=TEKS_GELAP)),
            **layout_chart,
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔍 Lihat contoh data mentah"):
        st.dataframe(df_sample.head(50), use_container_width=True)

# ============================================================
# HALAMAN — PERBANDINGAN MODEL (bug matplotlib sudah diperbaiki)
# ============================================================
elif halaman == "⚖️ Perbandingan Model":
    st.markdown("## ⚖️ Perbandingan Performa Model")
    st.write("Kelima algoritma dilatih dan dievaluasi pada data uji yang sama, identik dengan notebook UAS (`UAS_BigData_Mental_Health.ipynb`).")

    df_tampil = df_metrik.sort_values("Accuracy", ascending=False).reset_index(drop=True)

    # Catatan: .background_gradient() memerlukan matplotlib yang belum tentu ter-install
    # di semua environment, sehingga diganti dengan .highlight_max() yang tidak butuh
    # dependency tambahan sama sekali agar bebas error di lingkungan mana pun.
    styled = (
        df_tampil.style
        .format({"Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}"})
        .highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-Score"], color="#CFE6FF")
    )
    st.dataframe(styled, use_container_width=True)

    fig = go.Figure()
    metrik_list = ["Accuracy", "Precision", "Recall", "F1-Score"]
    warna_metrik = [BIRU_TUA, BIRU_UTAMA, BIRU_TERANG, "#9FCBFF"]
    for i, m in enumerate(metrik_list):
        fig.add_trace(go.Bar(name=m, x=df_tampil["Model"], y=df_tampil[m], marker_color=warna_metrik[i]))
    fig.update_layout(
        barmode="group", height=480, title="Perbandingan Metrik Antar Model",
        plot_bgcolor="white", paper_bgcolor="white", yaxis_range=[0, 1.05],
        font=dict(color=TEKS_GELAP, size=13),
        xaxis=dict(gridcolor="#E4EEFB"), yaxis=dict(gridcolor="#E4EEFB"),
        legend=dict(orientation="h", y=1.14, font=dict(color=TEKS_GELAP)),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "💡 **Random Forest** dipilih sebagai model produksi pada aplikasi ini karena memberikan "
        "akurasi tertinggi dan paling stabil dibandingkan model lainnya, termasuk setelah proses "
        "*hyperparameter tuning* (GridSearchCV) dan dibandingkan dengan XGBoost pada notebook UAS."
    )

# ============================================================
# HALAMAN — TENTANG
# ============================================================
elif halaman == "ℹ️ Tentang":
    st.markdown("## ℹ️ Tentang Aplikasi Ini")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="info-card">
        <h4>📚 Informasi Akademik</h4>
        <p><b>Mata Kuliah:</b> Big Data</p>
        <p><b>Jenis Tugas:</b> UAS (Individu)</p>
        <p><b>Program Studi:</b> Teknik Informatika</p>
        <p><b>Fakultas:</b> Ilmu Komputer</p>
        <p><b>Universitas:</b> Universitas Mercu Buana</p>
        <p><b>Nama:</b> Abdullah Al Manan</p>
        <p><b>NIM:</b> 41523010025</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="info-card">
        <h4>🛠️ Teknologi yang Digunakan</h4>
        <p>🐍 Python 3</p>
        <p>🎈 Streamlit — antarmuka aplikasi web</p>
        <p>🌲 Scikit-learn — Random Forest, preprocessing</p>
        <p>🚀 XGBoost — model pembanding</p>
        <p>📊 Plotly — visualisasi interaktif</p>
        <p>🐼 Pandas & NumPy — pengolahan data</p>
        </div>""", unsafe_allow_html=True)

    st.write("")
    st.markdown("""<div class="info-card">
    <h4>📦 Dataset & Keterkaitan dengan Notebook</h4>
    <p><b>Mental Health Dataset</b> — kumpulan data survei kesehatan mental berisi ±290 ribu baris
    data responden dari berbagai negara, mencakup informasi demografi, riwayat kesehatan mental,
    kebiasaan sehari-hari, dan kondisi psikososial.</p>
    <p>Aplikasi ini <b>terhubung langsung</b> ke notebook
    <code>UAS_BigData_Mental_Health_Final.ipynb</code> (disertakan dalam paket aplikasi ini).
    Skrip <code>train_model.py</code> disusun dari kode yang persis sama dengan notebook tersebut:
    missing value handling, duplicate removal, encoding, feature engineering <code>risk_score</code>,
    feature selection Chi-Square (16 fitur terpilih), standardisasi, hingga parameter Random Forest
    hasil <code>GridSearchCV</code> (<code>n_estimators=100, max_depth=15, min_samples_split=5</code>,
    akurasi ±98,92%). Hasil prediksi di aplikasi ini karena itu konsisten 1:1 dengan hasil evaluasi
    pada notebook.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="disclaimer-box" style="margin-top: 1.2rem;">
    ⚠️ Aplikasi ini murni untuk keperluan akademik dan edukasi mengenai penerapan Machine Learning
    pada data kesehatan mental, <b>bukan alat diagnosis klinis</b>. Selalu konsultasikan kondisi
    kesehatan mental dengan tenaga profesional yang kompeten.
    </div>""", unsafe_allow_html=True)
