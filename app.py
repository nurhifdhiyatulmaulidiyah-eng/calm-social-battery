import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# KONFIGURASI DASHBOARD
# ==========================================
st.set_page_config(page_title="Social Battery & Academic Burnout Analysis", layout="wide")

# Custom CSS untuk mempercantik Sidebar
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #fcfcfc;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("main_data_social_battre.csv")
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    if 'battery_category' not in df.columns:
        df['battery_category'] = pd.cut(df['battery_score'], bins=[-100, 0, 20, 50, 80, 200], 
                                        labels=['Sangat Lelah', 'Lelah', 'Cukup', 'Bagus', 'Sangat Bagus'])
    return df, month_order, day_order

try:
    df, month_order, day_order = load_data()

    # ==========================================
    # SIDEBAR (PENGATURAN RAPI)
    # ==========================================
    with st.sidebar:
        st.title("🎮 Dashboard Controller")
        st.write("Gunakan instrumen filter di bawah untuk memvalidasi data secara dinamis.")
        st.divider()

        # Menggunakan Expander agar Sidebar tidak terlihat penuh
        with st.expander("📅 Filter Periode", expanded=True):
            selected_months = st.multiselect(
                "Pilih Bulan:", 
                options=month_order, 
                default=month_order
            )
        
        with st.expander("⚡ Filter Level Energi", expanded=True):
            available_categories = df['battery_category'].unique().dropna().tolist()
            selected_categories = st.multiselect(
                "Kondisi Baterai:", 
                options=available_categories, 
                default=available_categories
            )
        
        st.divider()
        st.caption("🏷️ **Status Proyek**")
        st.caption("Capstone: Mood Jar")
        st.caption("Fokus: Academic Burnout")
        
        if st.button("🔄 Reset Data"):
            st.rerun()

    # PROSES FILTERING
    df_filtered = df[
        (df['month_name'].isin(selected_months)) & 
        (df['battery_category'].isin(selected_categories))
    ].copy()

    # ==========================================
    # HEADER UTAMA
    # ==========================================
    st.title("🌙 Capstone Project: Calm Social Battery")
    st.markdown("### Memahami Pola Kelelahan Sosial dan Risikonya terhadap Academic Burnout")
    st.divider()

    if df_filtered.empty:
        st.warning("⚠️ Data tidak ditemukan. Silakan sesuaikan kembali filter pada Sidebar.")
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
            
            st.markdown(f"""
            **Keterangan:** Grafik di atas menunjukkan kalau tingkat kelelahan tertinggi ada di bulan **{max_month}**. Sebaliknya, energi sosial paling terjaga dan stabil ada di bulan **{min_month}**. Penurunan kelelahan di bulan {min_month} menunjukkan adanya fase pemulihan yang baik, di mana beban interaksi harian masih dalam batas aman.
            """)
            
            st.success(f"""
            ✅ **Solusi Strategis:** Untuk mencegah **Academic Burnout** di bulan **{max_month}**, penting untuk membatasi jadwal kumpul-kumpul. Kita bisa mencontoh pola hidup di bulan **{min_month}** supaya pikiran tetap segar dan tidak gampang stres dengan tugas kuliah.
            """)
        st.divider()

        # ==========================================
        # 2. BATAS AMAN DURASI
        # ==========================================
        st.header("2. Berapa Lama Batas Aman Berinteraksi?")
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        sns.boxplot(data=df_filtered, x='battery_category', y='total_duration_minutes', palette="coolwarm", ax=ax2)
        st.pyplot(fig2)

        # Logika Perbaikan: Mencari median durasi pada kategori dengan tingkat energi terendah
        if not df_filtered.empty:
            # Mengambil kategori pertama (kategori paling kiri/paling lelah)
            target_cat = df_filtered['battery_category'].cat.categories[0] 
            subset_exhausted = df_filtered[df_filtered['battery_category'] == target_cat]
            
            if not subset_exhausted.empty:
                median_dur = subset_exhausted['total_duration_minutes'].median()
            else:
                median_dur = 0
        else:
            median_dur = 0

        st.markdown(f"""
        **Keterangan:** Lewat grafik ini, kita menemukan kalau saat kondisi **{target_cat}**, durasi kegiatan sosial rata-rata berada di angka **{median_dur:.0f} menit**. Ini jadi indikator penting bahwa durasi yang berlebihan berkaitan langsung dengan habisnya baterai sosial secara drastis.
        """)
        st.success(f"✅ **Solusi Strategis:** Usahakan untuk membatasi interaksi harian agar tidak jauh melampaui **{median_dur:.0f} menit**. Pengaturan waktu yang disiplin adalah kunci utama untuk menjaga stabilitas mental dan mencegah risiko **Academic Burnout**.")

        # ==========================================
        # 3. POLA HARIAN
        # ==========================================
        st.header("3. Hari Apa Energi Kamu Paling Cepat Drop?")
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        current_day_order = [d for d in day_order if d in df_filtered['day_of_week'].unique()]
        data_d = df_filtered.groupby('day_of_week')['battery_score'].mean().reindex(current_day_order)
        
        ax3.plot(data_d.index, data_d.values, marker='o', color='teal', linewidth=4, markersize=10)
        ax3.fill_between(data_d.index, data_d.values, alpha=0.2, color='teal')
        st.pyplot(fig3)

        if not data_d.dropna().empty:
            lowest_day = data_d.idxmin()
            st.markdown(f"""
            **Keterangan:** Grafik ini menunjukkan kalau hari **{lowest_day}** sering jadi titik paling capek dalam seminggu. Penurunan ini adalah sinyal kalau beban kegiatan mulai menumpuk dan kalau dibiarkan bisa bikin semangat kuliah menurun.
            """)
            st.success(f"✅ **Solusi Strategis:** Jadikan hari **{lowest_day}** sebagai waktu untuk istirahat penuh. Hindari kegiatan sosial yang terlalu berat di hari ini untuk mencegah kelelahan mingguan dan menjauhkan risiko **Academic Burnout**.")
        st.divider()

        # ==========================================
        # 4. KORELASI INTENSITAS
        # ==========================================
        st.header("4. Hubungan Antara Keramaian dan Sisa Energi")
        fig4, ax4 = plt.subplots(figsize=(12, 5))
        sns.regplot(data=df_filtered, x='social_intensity_score', y='battery_score', scatter_kws={'alpha':0.2}, line_kws={'color':'red'}, ax=ax4)
        st.pyplot(fig4)
        
        st.markdown("""
        **Keterangan:** Garis merah di atas membuktikan kalau semakin ramai atau intens interaksi sosial yang dijalani, maka sisa energi kita akan semakin cepat habis. Artinya, suasana yang terlalu ramai berpengaruh besar pada tingkat capek yang kita rasakan.
        """)
        st.success("✅ **Solusi Strategis:** Pintar-pintarlah memilih lingkungan sosial. Membatasi diri dari tempat yang terlalu ramai secara berturut-turut adalah cara ampuh untuk menghindari **Academic Burnout**.")
        st.divider()

        # ==========================================
        # 5. EFEKTIVITAS PEMBATASAN
        # ==========================================
        st.header("5. Apakah Membatasi Waktu Benar-benar Membantu?")
        df_filtered['group'] = np.where(df_filtered['total_duration_minutes'] < 900, 'Terkontrol (<15 jam)', 'Berisiko (>16 jam)')
        fig5, ax5 = plt.subplots(figsize=(12, 5))
        sns.barplot(data=df_filtered, x='group', y='battery_score', palette="plasma", ax=ax5)
        st.pyplot(fig5)
        
        st.markdown("""
        **Keterangan:** Data membuktikan kalau membatasi durasi kegiatan di bawah 15 jam sehari sangat efektif menjaga baterai sosial tetap stabil. Kelompok yang waktunya 'Terkontrol' punya energi yang jauh lebih baik untuk tetap produktif.
        """)
        st.success("✅ **Solusi Strategis:** Batasi waktu kegiatan harian sebagai langkah utama. Menjaga durasi tetap terkontrol adalah cara paling praktis untuk menghindari stres berkepanjangan dan kondisi **Academic Burnout**.")

    st.divider()
    st.info("💡 **Informasi Tambahan:** Dashboard ini mendeteksi pola kelelahanmu secara otomatis. Gunakan solusi yang muncul sebagai panduan untuk menjaga kesehatan mental selama kuliah.")

except Exception as e:
    st.error(f"⚠️ Terjadi hambatan teknis dalam memproses data: {e}")