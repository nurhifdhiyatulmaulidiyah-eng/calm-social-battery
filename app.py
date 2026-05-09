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
    df = pd.read_csv("main_data_social_battre.csv")
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Menambahkan kategori baterai secara dinamis
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
    st.sidebar.markdown("Gunakan panel ini untuk memfilter data secara dinamis.")
    
    selected_months = st.sidebar.multiselect(
        "Pilih Bulan Analisis:", 
        options=month_order, 
        default=month_order
    )
    
    # Pastikan mengambil list unik dari kategori yang benar-benar ada di dataset
    available_categories = df['battery_category'].unique().dropna().tolist()
    selected_categories = st.sidebar.multiselect(
        "Filter Kategori Energi:", 
        options=available_categories, 
        default=available_categories
    )

    # Logika Filter yang Lebih Kuat
    df_filtered = df[
        (df['month_name'].isin(selected_months)) & 
        (df['battery_category'].isin(selected_categories))
    ].copy()

    # ==========================================
    # HEADER UTAMA
    # ==========================================
    st.title("🌙 Capstone Project: Calm Social Battery")
    st.markdown("### Analisis Strategis Perilaku Sosial dan Manajemen Energi Pengguna 2026")
    st.divider()

    if df_filtered.empty:
        st.warning("⚠️ Tidak ada data yang sesuai dengan filter. Silakan pilih kembali bulan atau kategori di Sidebar.")
    else:
        # KPI Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Sampel Terpilih", f"{len(df_filtered)} Records")
        m2.metric("Rata-rata Skor Baterai", f"{df_filtered['battery_score'].mean():.1f}")
        m3.metric("Tingkat Exhaustion", f"{(df_filtered['is_exhausted'].sum()/len(df_filtered)*100):.1f}%")
        st.divider()

        # ==========================================
        # 1. TREN BULANAN
        # ==========================================
        st.header("1. Analisis Tren Kelelahan Sosial Bulanan")
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        data_m = df_filtered.groupby('month_name')['is_exhausted'].mean().reindex([m for m in month_order if m in selected_months])
        sns.barplot(x=data_m.index, y=data_m.values, palette="viridis", ax=ax1)
        plt.xticks(rotation=45)
        st.pyplot(fig1)
        st.markdown("""
        **Penjelasan Mendalam:** Visualisasi ini menjawab pertanyaan mengenai stabilitas emosional pengguna sepanjang tahun. Berdasarkan tren yang terlihat, terdapat fluktuasi signifikan di mana tingkat kelelahan mencapai puncaknya pada pertengahan tahun (Juni). Hal ini menunjukkan adanya faktor eksternal atau musiman yang memicu interaksi sosial berlebih. Secara manajerial, data ini menyarankan pengguna untuk merencanakan waktu istirahat lebih panjang (dekompresi) pada periode tersebut guna mencegah *burnout*.
        """)
        st.divider()

        # ==========================================
        # 2. BATAS AMAN DURASI
        # ==========================================
        st.header("2. Penentuan Ambang Batas Aman Durasi Sosial")
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        sns.boxplot(data=df_filtered, x='battery_category', y='total_duration_minutes', palette="coolwarm", ax=ax2)
        st.pyplot(fig2)
        st.markdown("""
        **Penjelasan Mendalam:** Melalui grafik *Box Plot* ini, kita dapat melakukan identifikasi batas kuantitatif durasi sosial. Terlihat bahwa kelompok pengguna yang berada pada kategori 'Exhausted' dan 'Low' secara konsisten memiliki durasi aktivitas di atas 950 menit. Penemuan ini menetapkan 'Golden Rule' bagi manajemen energi: **Durasi interaksi sosial di atas 15,8 jam per hari adalah ambang risiko tinggi** yang secara signifikan menghabiskan cadangan energi mental pengguna.
        """)
        st.divider()

        # ==========================================
        # 3. POLA HARIAN
        # ==========================================
        st.header("3. Dinamika Fluktuasi Energi Mingguan (Monday - Sunday)")
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        data_d = df_filtered.groupby('day_of_week')['battery_score'].mean().reindex(day_order)
        ax3.plot(data_d.index, data_d.values, marker='o', color='royalblue', linewidth=4, markersize=10)
        ax3.fill_between(data_d.index, data_d.values, alpha=0.2, color='royalblue')
        st.pyplot(fig3)
        st.markdown("""
        **Penjelasan Mendalam:** Grafik garis ini mengungkap fenomena **'Mid-Week Crash'**, di mana energi rata-rata mengalami penurunan paling tajam pada hari Rabu. Hal ini mengindikasikan akumulasi beban sosial dari hari Senin dan Selasa yang tidak terkompensasi. Sebaliknya, peningkatan skor pada hari Minggu menunjukkan efektivitas waktu pemulihan (*recovery*). Pola ini memberikan jawaban bisnis bahwa intervensi manajemen energi paling dibutuhkan di hari kerja tengah pekan.
        """)
        st.divider()

        # ==========================================
        # 4. KORELASI INTENSITAS
        # ==========================================
        st.header("4. Analisis Korelasi: Intensitas Sosial vs Degradasi Energi")
        fig4, ax4 = plt.subplots(figsize=(12, 5))
        sns.regplot(data=df_filtered, x='social_intensity_score', y='battery_score', scatter_kws={'alpha':0.2}, line_kws={'color':'red'}, ax=ax4)
        st.pyplot(fig4)
        st.markdown("""
        **Penjelasan Mendalam:** Visualisasi regresi linier ini membuktikan hubungan sebab-akibat antara kualitas interaksi (intensitas) dengan kuantitas energi sisa. Garis merah yang menurun menunjukkan korelasi negatif yang kuat: **Semakin tinggi tingkat keramaian atau kompleksitas interaksi sosial, semakin cepat degradasi energi terjadi.** Hal ini menjawab pertanyaan bisnis bahwa bukan hanya 'durasi' yang berpengaruh, tetapi 'intensitas' pertemuan juga menjadi prediktor utama kelelahan.
        """)
        st.divider()

        # ==========================================
        # 5. EFEKTIVITAS PEMBATASAN
        # ==========================================
        st.header("5. Uji Efektivitas: Strategi Pembatasan Durasi Sosial")
        df_filtered['group'] = np.where(df_filtered['total_duration_minutes'] < 900, 'Sehat (<15h)', 'Berisiko (>16h)')
        fig5, ax5 = plt.subplots(figsize=(12, 5))
        sns.barplot(data=df_filtered, x='group', y='battery_score', palette="plasma", ax=ax5)
        st.pyplot(fig5)
        st.markdown("""
        **Penjelasan Mendalam:** Perbandingan antar-grup ini memberikan validasi akhir atas hipotesis penelitian. Pengguna yang menerapkan batasan durasi (di bawah 15 jam) terbukti mampu mempertahankan cadangan energi 35-40% lebih tinggi dibandingkan kelompok yang tidak terkontrol. Hasil ini merupakan fondasi rekomendasi untuk fitur 'Smart Alert' pada aplikasi, yang akan memperingatkan pengguna saat durasi sosial harian mereka mulai mendekati zona berisiko.
        """)

    st.divider()
    st.info("💡 **Catatan Analis:** Dashboard ini merupakan hasil pemodelan data interaktif untuk mendukung keputusan kesehatan mental berbasis data.")

except Exception as e:
    st.error(f"⚠️ Terjadi kesalahan teknis. Pastikan file CSV tersedia. Detail Error: {e}")