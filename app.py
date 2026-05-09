import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Capstone Project: Calm Social Battery", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("dashboard/main_data_social_battre.csv")
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    return df, month_order, day_order

try:
    df, month_order, day_order = load_data()

    st.title("🌙 Capstone Project: Calm Social Battery")
    st.markdown("Analisis Mendalam Kelelahan Sosial dan Pola Pemulihan Energi Pengguna 2026")
    st.divider()

    # KPI
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1: st.metric("Total Sampel", f"{len(df)} Records")
    with col_m2: st.metric("Rata-rata Battery", f"{df['battery_score'].mean():.1f}")
    with col_m3: 
        ex_rate = (df['is_exhausted'].sum() / len(df)) * 100
        st.metric("Tingkat Exhaustion", f"{ex_rate:.1f}%")
    with col_m4: st.metric("Status Sistem", "Aktif")

    # 1. TREN BULANAN
    st.header("1. Analisis Tren Social Fatigue Bulanan")
    col1, col2 = st.columns([2, 1])
    with col1:
        monthly_ex = df.groupby('month_name')['is_exhausted'].mean() * 100
        monthly_ex = monthly_ex.reindex(month_order)
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.barplot(x=monthly_ex.index, y=monthly_ex.values, palette="RdYlGn_r", ax=ax1, edgecolor='black', hue=monthly_ex.index, legend=False)
        ax1.set_ylabel("Persentase Exhausted (%)")
        plt.xticks(rotation=45)
        st.pyplot(fig1)
    with col2:
        st.markdown("""
        **Penjelasan:** Berdasarkan analisis 2.000 record, tingkat kelelahan meningkat drastis di pertengahan tahun.
        - **Puncak Kelelahan:** Juni (**16.5%**).
        - **Periode Tersegar:** Maret (**5.8%**).
        - **Kesimpulan:** Lonjakan fatigue di bulan Juni mencapai 3x lipat dibanding Maret.
        """)

    st.divider()

    # 2. BATAS AMAN DURASI
    st.header("2. Penentuan Batas Aman Durasi Sosial")
    col3, col4 = st.columns([2, 1])
    with col3:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        if 'battery_category' not in df.columns:
            df['battery_category'] = pd.cut(df['battery_score'], bins=[-20, 0, 20, 50, 80, 120], labels=['Exhausted', 'Low', 'Medium', 'High', 'Very High'])
        sns.boxplot(data=df, x='battery_category', y='total_duration_minutes', palette="coolwarm", ax=ax2, hue='battery_category', legend=False)
        st.pyplot(fig2)
    with col4:
        st.markdown("""
        **Penjelasan:** Zona Merah (Low) memiliki durasi jauh lebih tinggi dibanding Zona Sehat.
        - **Median Zona Merah:** 978 Menit (16.3 Jam).
        - **Kesimpulan:** Batas aman durasi sosial harian adalah **< 950 menit**.
        """)

    st.divider()

    # 3. POLA HARIAN
    st.header("3. Pola Kelelahan Berdasarkan Hari (Weekly Pattern)")
    col5, col6 = st.columns([2, 1])
    with col5:
        daily_score = df.groupby('day_of_week')['battery_score'].mean().reindex(day_order)
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(daily_score.index, daily_score.values, marker='o', color='navy', linewidth=3)
        ax3.fill_between(daily_score.index, daily_score.values, color='skyblue', alpha=0.3)
        st.pyplot(fig3)
    with col6:
        st.markdown("""
        **Penjelasan:** Pola "U-Turn" energi mingguan terdeteksi jelas.
        - **Puncak Energi:** Sunday (**56.2**).
        - **Palung Energi (Crash):** Wednesday (**41.5**).
        - **Kesimpulan:** Intervensi paling dibutuhkan di hari Rabu.
        """)

    st.divider()

    # 4. KORELASI INTENSITAS
    st.header("4. Korelasi Intensitas Sosial vs Kelelahan")
    col7, col8 = st.columns([2, 1])
    with col7:
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        sns.regplot(data=df, x='social_intensity_score', y='battery_score', scatter_kws={'alpha':0.3, 'color':'gray'}, line_kws={'color':'red'}, ax=ax4)
        st.pyplot(fig4)
    with col8:
        st.markdown("""
        **Penjelasan:** Uji r menunjukkan korelasi negatif sebesar **-0.352**.
        - **Kesimpulan:** Semakin tinggi intensitas sosial harian, semakin rendah sisa energi sosial pengguna.
        """)

    st.divider()

    # 5. PERBANDINGAN SEHAT VS LELAH
    st.header("5. Perbandingan Efek Durasi: Sehat vs Lelah")
    col9, col10 = st.columns([2, 1])
    with col9:
        df['group'] = np.where(df['total_duration_minutes'] < 900, 'Sehat (<15h)', 'Lelah (>16.7h)')
        fig5, ax5 = plt.subplots(figsize=(10, 5))
        sns.barplot(data=df, x='group', y='battery_score', palette="viridis", ax=ax5, hue='group', legend=False)
        st.pyplot(fig5)
    with col10:
        st.markdown("""
        **Penjelasan:** Kelompok durasi aman memiliki energi **37% lebih tinggi**.
        - **Kesimpulan:** Durasi aktivitas sosial berpengaruh signifikan terhadap risiko kelelahan ekstrem.
        """)

    st.divider()
    st.markdown("<center><b>Capstone Project: Calm Social Battery - Analisis Data Selesai</b></center>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Gagal memuat dashboard. Error: {e}")