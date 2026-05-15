import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# KONFIGURASI DASHBOARD
# ==========================================
st.set_page_config(page_title="Analisis Social Battery & Academic Burnout", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #fcfcfc; border-right: 1px solid #e6e6e6; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    h1, h2, h3 { color: #2c3e50; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    # Pastikan file CSV tersedia
    df = pd.read_csv("main_data_social_battre.csv")
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    bins = [-100, 0, 20, 50, 80, 200]
    labels = ['Sangat Lelah', 'Lelah', 'Cukup', 'Bagus', 'Sangat Bagus']
    df['battery_category'] = pd.cut(df['battery_score'], bins=bins, labels=labels)
    
    return df, month_order, day_order

try:
    df, month_order, day_order = load_data()
    focus_categories = ['Sangat Lelah', 'Lelah', 'Cukup']

    # ==========================================
    # LOGIKA RESET (SESSION STATE)
    # ==========================================
    # Inisialisasi state jika belum ada
    if 'selected_months' not in st.session_state:
        st.session_state.selected_months = month_order
    
    if 'selected_categories' not in st.session_state:
        st.session_state.selected_categories = focus_categories

    # ==========================================
    # SIDEBAR (PENGATURAN INSTRUMEN)
    # ==========================================
    with st.sidebar:
        st.title("🎮 Control Panel")
        st.write("Gunakan filter ini untuk menyesuaikan parameter analisis secara dinamis.")
        st.divider()

        with st.expander("📅 Periode Analisis", expanded=True):
            # Widget multiselect terhubung ke session_state via key
            st.multiselect(
                "Pilih Bulan:", 
                options=month_order, 
                key="selected_months"
            )
        
        with st.expander("⚡ Kondisi Energi", expanded=True):
            st.multiselect(
                "Level Baterai:", 
                options=focus_categories, 
                key="selected_categories"
            )
        
        st.divider()
        st.caption("🏷️ **Status Proyek**")
        st.caption("Capstone: Calm Social Battery")
        st.caption("Fokus: Academic Burnout")
        
        # Tombol Reset yang diperbaiki
        if st.button("🔄 Reset Semua Filter"):
            st.session_state.selected_months = month_order
            st.session_state.selected_categories = focus_categories
            st.rerun()

    # PROSES FILTERING DATA menggunakan nilai dari session_state
    df_filtered = df[
        (df['month_name'].isin(st.session_state.selected_months)) & 
        (df['battery_category'].isin(st.session_state.selected_categories))
    ].copy()

    # ==========================================
    # HEADER UTAMA
    # ==========================================
    st.title("🌙 Capstone Project: Calm Social Battery")
    st.markdown("### Analisis Prevalensi Social Exhaustion dan Mitigasi Risiko Academic Burnout")
    st.divider()

    if df_filtered.empty:
        st.warning("⚠️ Data tidak ditemukan. Silakan sesuaikan parameter pada filter di samping.")
    else:
        # Metrik Utama (KPI)
        avg_score = df_filtered['battery_score'].mean()
        exhaustion_rate = (df_filtered[df_filtered['battery_score'] < 0].shape[0]/len(df_filtered)*100)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Volume Observasi", f"{len(df_filtered)} Hari")
        m2.metric("Mean Battery Score", f"{avg_score:.1f}")
        m3.metric("Indeks Kelelahan Kritis", f"{exhaustion_rate:.1f}%")
        st.divider()

        # --- 1. TREN BULANAN ---
        st.header("1. Analisis Fluktuasi Kelelahan Sosial Bulanan")
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        df_filtered['is_negative'] = df_filtered['battery_score'] < 0
        # Menghitung persentase kelelahan kritis (skor < 0)
        data_m = df_filtered.groupby('month_name')['is_negative'].mean() * 100
        # Reindex agar urutan bulan sesuai kalender
        data_m = data_m.reindex([m for m in month_order if m in st.session_state.selected_months])
        
        sns.barplot(x=data_m.index, y=data_m.values, palette="magma", ax=ax1)
        ax1.set_ylabel("Persentase Kelelahan (%)")
        plt.xticks(rotation=45)
        st.pyplot(fig1)

        if not data_m.dropna().empty:
            max_month = data_m.idxmax()
            min_month = data_m.idxmin()
            st.markdown(f"""
            **Keterangan:** Grafik ini menunjukkan pola musiman kelelahan sosial. Bulan **{max_month}** diidentifikasi sebagai periode dengan tingkat kelelahan tertinggi. Sebaliknya, bulan **{min_month}** menunjukkan fase stabilisasi energi.
            """)
            st.success(f"""
            ✅ **Solusi Strategis:** Untuk memitigasi lonjakan beban di bulan **{max_month}**, pengguna perlu menerapkan *pre-emptive rest*.
            """)
        st.divider()

        # --- 2. BATAS AMAN DURASI ---
        st.header("2. Identifikasi Threshold Durasi terhadap Penurunan Energi")
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        sns.boxplot(data=df_filtered, x='battery_category', y='total_duration_minutes', 
                    order=focus_categories, palette="coolwarm", ax=ax2)
        st.pyplot(fig2)

        subset_lelah = df_filtered[df_filtered['battery_category'] == 'Sangat Lelah']
        median_val = subset_lelah['total_duration_minutes'].median() if not subset_lelah.empty else df_filtered['total_duration_minutes'].median()
        
        st.markdown(f"""
        **Keterangan:** Kondisi 'Sangat Lelah' secara konsisten muncul ketika durasi interaksi sosial mencapai median **{median_val:.0f} menit**.
        """)
        st.success(f"""
        ✅ **Solusi Strategis:** Tetapkan durasi **{median_val:.0f} menit** sebagai batas aman interaksi sosial harian.
        """)
        st.divider()

        # --- 3. POLA HARIAN (MID-WEEK CRASH) ---
        st.header("3. Analisis Siklus Mingguan dan Fenomena Titik Jenuh (Crash)")
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        current_day_order = [d for d in day_order if d in df_filtered['day_of_week'].unique()]
        data_d = df_filtered.groupby('day_of_week')['battery_score'].mean().reindex(current_day_order)
        ax3.plot(data_d.index, data_d.values, marker='o', color='teal', linewidth=4, markersize=10)
        ax3.fill_between(data_d.index, data_d.values, alpha=0.2, color='teal')
        st.pyplot(fig3)

        if not data_d.dropna().empty:
            low_day = data_d.idxmin()
            st.markdown(f"""
            **Keterangan:** Grafik ini mengungkap fenomena penurunan energi di tengah minggu, dengan hari **{low_day}** sebagai titik terendah.
            """)
            st.success(f"""
            ✅ **Solusi Strategis:** Terapkan strategi **'Micro-recovery'** pada hari {low_day}.
            """)
        st.divider()

        # --- 4. KORELASI INTENSITAS ---
        st.header("4. Signifikansi Korelasi: Intensitas Lingkungan terhadap Degradasi Energi")
        fig4, ax4 = plt.subplots(figsize=(12, 5))
        sns.regplot(data=df_filtered, x='social_intensity_score', y='battery_score', scatter_kws={'alpha':0.2}, line_kws={'color':'red'}, ax=ax4)
        st.pyplot(fig4)
        
        st.markdown("""
        **Keterangan:** Ditemukan bukti empiris bahwa intensitas lingkungan memiliki pengaruh negatif signifikan terhadap sisa energi.
        """)
        st.success("""
        ✅ **Solusi Strategis:** Terapkan teknik **'Social Pacing'** untuk mengurangi beban sensorik.
        """)
        st.divider()

        # --- 5. EFEKTIVITAS PEMBATASAN ---
        st.header("5. Uji Komparatif Manajemen Durasi sebagai Mitigasi Burnout")
        df_filtered['group'] = np.where(df_filtered['total_duration_minutes'] < 900, 'Terkontrol (<15 jam)', 'Berisiko (>16 jam)')
        fig5, ax5 = plt.subplots(figsize=(12, 5))
        sns.barplot(data=df_filtered, x='group', y='battery_score', palette="plasma", ax=ax5)
        st.pyplot(fig5)
        
        st.markdown("""
        **Keterangan:** Kelompok 'Terkontrol' memiliki resiliensi energi yang jauh lebih stabil dibandingkan kelompok 'Berisiko'.
        """)
        st.success("""
        ✅ **Solusi Strategis:** Disiplin menjaga durasi kegiatan pada zona 'Terkontrol' adalah strategi terbaik menghindari **Academic Burnout**.
        """)

    st.divider()
    st.info("💡 **Catatan Akademis:** Dashboard dirancang untuk mendeteksi pola kelelahan sosial secara proaktif.")

except Exception as e:
    st.error(f"⚠️ Terjadi kesalahan: {e}")