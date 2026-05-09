import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Capstone Project: Calm Social Battery", layout="wide")

# Load Data
@st.cache_data
def load_data():
    # Karena file di GitHub ada di folder utama, tidak perlu pakai "dashboard/"
    df = pd.read_csv("main_data_social_battre.csv")
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    if 'battery_category' not in df.columns:
        df['battery_category'] = pd.cut(df['battery_score'], bins=[-20, 0, 20, 50, 80, 120], 
                                        labels=['Exhausted', 'Low', 'Medium', 'High', 'Very High'])
    return df, month_order, day_order

try:
    df, month_order, day_order = load_data()

    # ==========================================
    # SIDEBAR (CONTROL PANEL)
    # ==========================================
    st.sidebar.title("🎮 Dashboard Controller")
    st.sidebar.markdown("Atur filter untuk melihat perubahan data secara real-time.")
    
    selected_months = st.sidebar.multiselect(
        "Pilih Bulan:", 
        options=month_order, 
        default=month_order
    )
    
    selected_categories = st.sidebar.multiselect(
        "Status Energi:", 
        options=df['battery_category'].unique().tolist(), 
        default=df['battery_category'].unique().tolist()
    )

    # Filter Data
    df_filtered = df[(df['month_name'].isin(selected_months)) & (df['battery_category'].isin(selected_categories))]

    # ==========================================
    # HEADER UTAMA
    # ==========================================
    st.title("🌙 Capstone Project: Calm Social Battery")
    st.markdown("Analisis Perilaku Sosial dan Manajemen Energi Pengguna 2026")
    st.divider()

    # KPI Metrics di bagian atas
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Sampel Data", f"{len(df_filtered)} Records")
    m2.metric("Rata-rata Skor Baterai", f"{df_filtered['battery_score'].mean():.1f}")
    m3.metric("Tingkat Kelelahan (Exhausted)", f"{(df_filtered['is_exhausted'].sum()/len(df_filtered)*100):.1f}%")
    st.divider()

    # ==========================================
    # 1. TREN BULANAN
    # ==========================================
    st.header("1. Tren Social Fatigue Bulanan")
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    data_m = df_filtered.groupby('month_name')['is_exhausted'].mean().reindex([m for m in month_order if m in selected_months])
    sns.barplot(x=data_m.index, y=data_m.values, palette="viridis", ax=ax1, hue=data_m.index, legend=False)
    plt.xticks(rotation=45)
    st.pyplot(fig1)
    st.write("**Kesimpulan:** Grafik ini menjawab kapan risiko kelelahan tertinggi terjadi (puncak di bulan Juni).")
    st.divider()

    # ==========================================
    # 2. BATAS AMAN DURASI
    # ==========================================
    st.header("2. Batas Aman Durasi Sosial")
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df_filtered, x='battery_category', y='total_duration_minutes', palette="coolwarm", ax=ax2)
    st.pyplot(fig2)
    st.write("**Kesimpulan:** Menentukan ambang batas durasi sosial (~950 menit) agar baterai tidak masuk zona 'Low'.")
    st.divider()

    # ==========================================
    # 3. POLA HARIAN
    # ==========================================
    st.header("3. Pola Kelelahan Mingguan (Monday - Sunday)")
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    data_d = df_filtered.groupby('day_of_week')['battery_score'].mean().reindex(day_order)
    ax3.plot(data_d.index, data_d.values, marker='o', color='royalblue', linewidth=4, markersize=10)
    ax3.fill_between(data_d.index, data_d.values, alpha=0.2, color='royalblue')
    st.pyplot(fig3)
    st.write("**Kesimpulan:** Mengidentifikasi 'Mid-Week Crash' pada hari Rabu dan proses recovery di hari Minggu.")
    st.divider()

    # ==========================================
    # 4. KORELASI INTENSITAS
    # ==========================================
    st.header("4. Hubungan Intensitas Sosial vs Skor Baterai")
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    sns.regplot(data=df_filtered, x='social_intensity_score', y='battery_score', scatter_kws={'alpha':0.2}, line_kws={'color':'red'}, ax=ax4)
    st.pyplot(fig4)
    st.write("**Kesimpulan:** Membuktikan bahwa semakin ramai/intens interaksi sosial, semakin cepat baterai energi terkuras.")
    st.divider()

    # ==========================================
    # 5. EFEKTIVITAS PEMBATASAN
    # ==========================================
    st.header("5. Perbandingan Kelompok Durasi Sehat vs Lelah")
    df_filtered['group'] = np.where(df_filtered['total_duration_minutes'] < 900, 'Sehat (<15h)', 'Lelah (>16h)')
    fig5, ax5 = plt.subplots(figsize=(12, 5))
    sns.barplot(data=df_filtered, x='group', y='battery_score', palette="plasma", ax=ax5, hue='group', legend=False)
    st.pyplot(fig5)
    st.write("**Kesimpulan:** Menjawab pertanyaan apakah membatasi durasi sosial benar-benar efektif menjaga energi.")

    st.divider()
    st.info("💡 Dashboard ini dirancang untuk memberikan insight mendalam bagi manajemen energi sosial pengguna.")

except Exception as e:
    st.error(f"Gagal memuat dashboard. Error: {e}")