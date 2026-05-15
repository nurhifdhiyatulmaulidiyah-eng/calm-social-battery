import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# KONFIGURASI DASHBOARD
# ==========================================
st.set_page_config(page_title="Analisis Baterai Sosial & Stress Kuliah", layout="wide")

# Custom CSS untuk Sidebar dan Metrik
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #fcfcfc; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("main_data_social_battre.csv")
    # Urutan Bulan & Hari
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Memastikan kolom kategori dibuat dengan benar
    bins = [-100, 0, 20, 50, 80, 200]
    labels = ['Sangat Lelah', 'Lelah', 'Cukup', 'Bagus', 'Sangat Bagus']
    df['battery_category'] = pd.cut(df['battery_score'], bins=bins, labels=labels)
    
    return df, month_order, day_order

try:
    df, month_order, day_order = load_data()

    # ==========================================
    # SIDEBAR (PENGATURAN)
    # ==========================================
    with st.sidebar:
        st.title("🎮 Control Panel")
        st.write("Filter data di bawah untuk melihat hasil analisis secara real-time.")
        st.divider()

        with st.expander("📅 Filter Waktu", expanded=True):
            selected_months = st.multiselect("Pilih Bulan:", options=month_order, default=month_order)
        
        with st.expander("⚡ Filter Energi", expanded=True):
            available_categories = ['Sangat Lelah', 'Lelah', 'Cukup', 'Bagus', 'Sangat Bagus']
            selected_categories = st.multiselect("Kondisi Baterai:", options=available_categories, default=available_categories)
        
        st.divider()
        st.caption("🏷️ **Status Proyek**")
        st.caption("Capstone: Mood Jar")
        st.caption("Fokus: Academic Burnout")

    # PROSES FILTERING
    df_filtered = df[
        (df['month_name'].isin(selected_months)) & 
        (df['battery_category'].isin(selected_categories))
    ].copy()

    # ==========================================
    # HEADER UTAMA
    # ==========================================
    st.title("🌙 Capstone Project: Calm Social Battery")
    st.markdown("### Memahami Pola Kelelahan Sosial dan Dampaknya pada Academic Burnout")
    st.divider()

    if df_filtered.empty:
        st.warning("⚠️ Data tidak ditemukan. Silakan sesuaikan pilihan di Sidebar.")
    else:
        # Metrik Utama
        avg_score = df_filtered['battery_score'].mean()
        exhaustion_rate = (df_filtered[df_filtered['battery_score'] < 0].shape[0]/len(df_filtered)*100)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Volume Observasi", f"{len(df_filtered)} Hari")
        m2.metric("Mean Battery Score", f"{avg_score:.1f}")
        m3.metric("Indeks Kelelahan Kritis", f"{exhaustion_rate:.1f}%")
        st.divider()

        # --- 1. TREN BULANAN ---
        st.header("1. Kapan Waktu Paling Lelah di Setiap Bulannya?")
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        df_filtered['is_negative'] = df_filtered['battery_score'] < 0
        data_m = df_filtered.groupby('month_name')['is_negative'].mean() * 100
        data_m = data_m.reindex([m for m in month_order if m in selected_months])
        sns.barplot(x=data_m.index, y=data_m.values, palette="magma", ax=ax1)
        ax1.set_ylabel("Tingkat Kelelahan (%)")
        plt.xticks(rotation=45)
        st.pyplot(fig1)

        if not data_m.dropna().empty:
            max_month = data_m.idxmax()
            min_month = data_m.idxmin()
            st.markdown(f"**Keterangan:** Kelelahan tertinggi ada di bulan **{max_month}**. Sebaliknya, energi paling terjaga di bulan **{min_month}**.")
            st.success(f"✅ **Solusi:** Kurangi jadwal sosial di bulan **{max_month}** dan contoh pola hidup di bulan **{min_month}**.")
        st.divider()

        # --- 2. BATAS AMAN DURASI ---
        st.header("2. Berapa Lama Batas Aman Berinteraksi?")
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        sns.boxplot(data=df_filtered, x='battery_category', y='total_duration_minutes', palette="coolwarm", ax=ax2)
        st.pyplot(fig2)

        # Logika Sederhana mencari median durasi
        # Ambil median dari kategori yang paling lelah ('Sangat Lelah')
        subset_lelah = df_filtered[df_filtered['battery_category'] == 'Sangat Lelah']
        if not subset_lelah.empty:
            median_val = subset_lelah['total_duration_minutes'].median()
        else:
            # Jika 'Sangat Lelah' tidak terpilih di filter, ambil rata-rata umum
            median_val = df_filtered['total_duration_minutes'].median()

        st.markdown(f"**Keterangan:** Data menunjukkan kalau kondisi sangat lelah biasanya terjadi setelah beraktivitas sosial selama **{median_val:.0f} menit**.")
        st.success(f"✅ **Solusi:** Batasi interaksi harian maksimal **{median_val:.0f} menit** untuk mencegah risiko **Academic Burnout**.")
        st.divider()

        # --- 3. POLA HARIAN ---
        st.header("3. Hari Apa Energi Kamu Paling Drop?")
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        current_day_order = [d for d in day_order if d in df_filtered['day_of_week'].unique()]
        data_d = df_filtered.groupby('day_of_week')['battery_score'].mean().reindex(current_day_order)
        ax3.plot(data_d.index, data_d.values, marker='o', color='teal', linewidth=4)
        ax3.fill_between(data_d.index, data_d.values, alpha=0.2, color='teal')
        st.pyplot(fig3)

        if not data_d.dropna().empty:
            low_day = data_d.idxmin()
            st.markdown(f"**Keterangan:** Hari **{low_day}** sering jadi titik paling capek dalam seminggu.")
            st.success(f"✅ **Solusi:** Jadikan hari **{low_day}** sebagai waktu 'Me Time' untuk memulihkan energi.")
        st.divider()

        # --- 4. KORELASI INTENSITAS ---
        st.header("4. Hubungan Antara Keramaian dan Sisa Energi")
        fig4, ax4 = plt.subplots(figsize=(12, 5))
        sns.regplot(data=df_filtered, x='social_intensity_score', y='battery_score', scatter_kws={'alpha':0.2}, line_kws={'color':'red'}, ax=ax4)
        st.pyplot(fig4)
        st.markdown("**Keterangan:** Semakin ramai suasana sosial, semakin cepat baterai sosialmu habis.")
        st.success("✅ **Solusi:** Pilih lingkungan yang lebih tenang jika merasa baterai sudah mulai menipis.")
        st.divider()

        # --- 5. EFEKTIVITAS PEMBATASAN ---
        st.header("5. Apakah Membatasi Waktu Benar-benar Membantu?")
        df_filtered['group'] = np.where(df_filtered['total_duration_minutes'] < 900, 'Terkontrol (<15 jam)', 'Berisiko (>16 jam)')
        fig5, ax5 = plt.subplots(figsize=(12, 5))
        sns.barplot(data=df_filtered, x='group', y='battery_score', palette="plasma", ax=ax5)
        st.pyplot(fig5)
        st.markdown("**Keterangan:** Membatasi waktu kegiatan sangat efektif menjaga energi tetap stabil.")
        st.success("✅ **Solusi:** Jaga durasi kegiatan tetap terkontrol untuk menghindari kondisi **Academic Burnout**.")

    st.divider()
    st.info("💡 **Informasi Tambahan:** Dashboard ini mendeteksi pola kelelahanmu secara otomatis.")

except Exception as e:
    st.error(f"⚠️ Terjadi kesalahan: {e}")