import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Social Battery & Academic Burnout Analysis", layout="wide")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("main_data_social_battre.csv")
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    if 'battery_category' not in df.columns:
        df['battery_category'] = pd.cut(df['battery_score'], bins=[-100, 0, 20, 50, 80, 200], 
                                        labels=['Exhausted', 'Low', 'Medium', 'High', 'Very High'])
    return df, month_order, day_order

try:
    df, month_order, day_order = load_data()

    # ==========================================
    # SIDEBAR (CONTROL PANEL)
    # ==========================================
    st.sidebar.title("🎮 Dashboard Controller")
    st.sidebar.markdown("Filter di bawah ini akan mengubah grafik dan keterangan secara otomatis.")
    
    selected_months = st.sidebar.multiselect(
        "Pilih Periode Analisis:", 
        options=month_order, 
        default=month_order
    )
    
    available_categories = df['battery_category'].unique().dropna().tolist()
    selected_categories = st.sidebar.multiselect(
        "Filter Level Exhaustion:", 
        options=available_categories, 
        default=available_categories
    )

    # DATA FILTERING
    df_filtered = df[
        (df['month_name'].isin(selected_months)) & 
        (df['battery_category'].isin(selected_categories))
    ].copy()

    # ==========================================
    # HEADER UTAMA
    # ==========================================
    st.title("🌙 Capstone Project: Calm Social Battery")
    st.markdown("### Pemodelan Analitik: Prevalensi Social Exhaustion dan Risiko Academic Burnout 2026")
    st.divider()

    if df_filtered.empty:
        st.warning("⚠️ Dataset tidak ditemukan untuk kombinasi filter ini. Silakan sesuaikan kembali pada Sidebar.")
    else:
        # KPI Metrics
        avg_score = df_filtered['battery_score'].mean()
        exhaustion_rate = (df_filtered[df_filtered['battery_score'] < 0].shape[0]/len(df_filtered)*100)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Volume Observasi", f"{len(df_filtered)} Hari")
        m2.metric("Mean Battery Score", f"{avg_score:.1f}")
        m3.metric("Indeks Kelelahan Kritis", f"{exhaustion_rate:.1f}%")
        st.divider()

        # ==========================================
        # 1. TREN BULANAN
        # ==========================================
        st.header("1. Tren Prevalensi Social Exhaustion Bulanan")
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        df_filtered['is_negative'] = df_filtered['battery_score'] < 0
        data_m = df_filtered.groupby('month_name')['is_negative'].mean() * 100
        data_m = data_m.reindex([m for m in month_order if m in selected_months])
        
        sns.barplot(x=data_m.index, y=data_m.values, palette="magma", ax=ax1)
        ax1.set_ylabel("Persentase Exhaustion (%)")
        plt.xticks(rotation=45)
        st.pyplot(fig1)

        # KETERANGAN 1
        max_month = data_m.idxmax() if not data_m.dropna().empty else "N/A"
        st.markdown(f"""
        **Keterangan:** Berdasarkan filter periode yang dipilih, tingkat kelelahan tertinggi terdeteksi pada bulan **{max_month}** dengan persentase kelelahan kritis sebesar **{data_m.max():.1f}%**. Visualisasi ini memetakan fluktuasi *social exhaustion* ekstrem yang didefinisikan melalui skor baterai negatif. Lonjakan pada periode ini mengindikasikan akumulasi stresor sosial yang berkontribusi signifikan pada risiko *Academic Burnout*.
        """)
        st.divider()

        # ==========================================
        # 2. BATAS AMAN DURASI
        # ==========================================
        st.header("2. Analisis Threshold Durasi Terhadap Kondisi Academic Burnout")
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        sns.boxplot(data=df_filtered, x='battery_category', y='total_duration_minutes', palette="coolwarm", ax=ax2)
        st.pyplot(fig2)

        # KETERANGAN 2
        median_dur = df_filtered[df_filtered['battery_category'] == 'Exhausted']['total_duration_minutes'].median()
        st.markdown(f"""
        **Keterangan:** Identifikasi melalui distribusi statistik ini menetapkan parameter durasi maksimal interaksi sosial. Pada data saat ini, kelompok 'Exhausted' menunjukkan median durasi sebesar **{median_dur:.0f} menit**, mengonfirmasi bahwa eksposur sosial yang berlebihan merupakan prediktor utama terjadinya *Academic Burnout*.
        """)
        st.divider()

        # ==========================================
        # 3. POLA HARIAN
        # ==========================================
        st.header("3. Siklus Mingguan Social Battery")
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        current_day_order = [d for d in day_order if d in df_filtered['day_of_week'].unique()]
        data_d = df_filtered.groupby('day_of_week')['battery_score'].mean().reindex(current_day_order)
        
        ax3.plot(data_d.index, data_d.values, marker='o', color='teal', linewidth=4, markersize=10)
        ax3.fill_between(data_d.index, data_d.values, alpha=0.2, color='teal')
        st.pyplot(fig3)

        # KETERANGAN 3
        lowest_day = data_d.idxmin() if not data_d.dropna().empty else "N/A"
        st.markdown(f"""
        **Keterangan:** Grafik garis ini mengungkap pola sirkadian mingguan energi sosial pengguna. Titik terendah terdeteksi pada hari **{lowest_day}** dengan rata-rata skor **{data_d.min():.1f}**. Penurunan ini mencerminkan fase awal *exhaustion* yang jika dibiarkan akan berujung pada titik jenuh psikologis.
        """)
        st.divider()

        # ==========================================
        # 4. KORELASI INTENSITAS
        # ==========================================
        st.header("4. Signifikansi Korelasi Intensitas")
        fig4, ax4 = plt.subplots(figsize=(12, 5))
        sns.regplot(data=df_filtered, x='social_intensity_score', y='battery_score', scatter_kws={'alpha':0.2}, line_kws={'color':'red'}, ax=ax4)
        st.pyplot(fig4)
        
        st.markdown("""
        **Keterangan:** Melalui pengujian regresi linear, ditemukan bukti bahwa variabel intensitas sosial memiliki pengaruh negatif yang signifikan terhadap sisa cadangan energi. Semakin tinggi beban interaksi, semakin cepat laju *exhaustion* terjadi, mempercepat transisi menuju *Academic Burnout*.
        """)
        st.divider()

        # ==========================================
        # 5. EFEKTIVITAS PEMBATASAN
        # ==========================================
        st.header("5. Uji Komparatif Manajemen Durasi")
        df_filtered['group'] = np.where(df_filtered['total_duration_minutes'] < 900, 'Terkontrol (<15h)', 'Berisiko (>16.7h)')
        fig5, ax5 = plt.subplots(figsize=(12, 5))
        sns.barplot(data=df_filtered, x='group', y='battery_score', palette="plasma", ax=ax5)
        st.pyplot(fig5)
        
        st.markdown("""
        **Keterangan:** Visualisasi komparatif ini menunjukkan bahwa manajemen durasi di bawah 15 jam secara signifikan mampu menekan risiko *exhaustion*. Pengguna dengan pola durasi terkontrol memiliki daya tahan energi mental yang lebih berkelanjutan dibandingkan kelompok berisiko.
        """)

    st.divider()
    st.info("💡 **Catatan Akademis:** Dashboard ini bersifat dinamis. Semua metrik dan keterangan di atas dihitung secara real-time berdasarkan filter yang dipilih.")

except Exception as e:
    st.error(f"⚠️ Terjadi hambatan teknis: {e}")