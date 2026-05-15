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
    st.sidebar.markdown("Filter di bawah ini akan mengubah grafik, keterangan, dan solusi secara otomatis.")
    
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
        st.warning("⚠️ Dataset tidak ditemukan. Silakan sesuaikan kembali parameter pada Sidebar.")
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

        # LOGIKA DINAMIS UNTUK KETERANGAN & SOLUSI
        if not data_m.dropna().empty:
            max_month = data_m.idxmax()
            min_month = data_m.idxmin()
            max_val = data_m.max()
            min_val = data_m.min()
            
            st.markdown(f"""
            **Keterangan:** Berdasarkan filter periode yang dipilih, tingkat kelelahan tertinggi terdeteksi pada bulan **{max_month}** ({max_val:.1f}%), yang mengindikasikan akumulasi stresor sosial maksimal. Sebaliknya, periode stabilisasi energi paling optimal terjadi pada bulan **{min_month}** dengan tingkat kelelahan hanya **{min_val:.1f}%**. Penurunan pada bulan {min_month} menunjukkan adanya fase pemulihan (*recovery*) atau berkurangnya beban interaksi yang membantu menjaga homeostasis mental pengguna.
            """)
            
            # SOLUSI DINAMIS BERDASARKAN BULAN TERTINGGI
            st.success(f"""
            ✅ **Solusi Strategis:** Untuk memitigasi lonjakan kelelahan yang terjadi pada bulan **{max_month}**, direkomendasikan implementasi *Social Distancing* temporer dan peningkatan durasi *Deep Sleep*. Belajarlah dari pola pada bulan **{min_month}** di mana energi lebih terjaga; prioritaskan kualitas interaksi di atas kuantitas untuk menghindari degradasi energi menuju fase *Academic Burnout*.
            """)
        st.divider()

        # ==========================================
        # 2. BATAS AMAN DURASI
        # ==========================================
        st.header("2. Analisis Threshold Durasi Terhadap Kondisi Academic Burnout")
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        sns.boxplot(data=df_filtered, x='battery_category', y='total_duration_minutes', palette="coolwarm", ax=ax2)
        st.pyplot(fig2)

        # KETERANGAN 2 & SOLUSI
        median_dur = df_filtered[df_filtered['battery_category'] == 'Exhausted']['total_duration_minutes'].median()
        st.markdown(f"""
        **Keterangan:** Identifikasi melalui distribusi statistik menunjukkan bahwa kelompok 'Exhausted' memiliki median durasi interaksi sebesar **{median_dur:.0f} menit**. Hal ini mengonfirmasi bahwa durasi di atas ambang batas tersebut merupakan prediktor utama terjadinya degradasi energi sosial.
        """)
        st.success(f"✅ **Solusi Strategis:** Tetapkan batasan waktu interaksi maksimal harian di bawah **{median_dur:.0f} menit**. Gunakan fitur pengingat durasi untuk mencegah eksposur sosial yang berlebihan yang dapat mempercepat transisi menuju kelelahan klinis.")
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

        # KETERANGAN 3 & SOLUSI
        if not data_d.dropna().empty:
            lowest_day = data_d.idxmin()
            st.markdown(f"""
            **Keterangan:** Grafik garis ini mengungkap pola harian di mana hari **{lowest_day}** menjadi titik energi terendah bagi pengguna. Kondisi ini mencerminkan fase jenuh psikologis akibat akumulasi beban interaksi mingguan yang tidak terkompensasi.
            """)
            st.success(f"✅ **Solusi Strategis:** Jadwalkan 'Recharge Time' atau waktu istirahat penuh pada hari **{lowest_day}**. Hindari pertemuan sosial intensitas tinggi pada hari tersebut untuk mencegah terjadinya *Mid-Week Crash* dan menjaga produktivitas akademik tetap stabil.")
        st.divider()

        # ==========================================
        # 4. KORELASI INTENSITAS
        # ==========================================
        st.header("4. Signifikansi Korelasi Intensitas")
        fig4, ax4 = plt.subplots(figsize=(12, 5))
        sns.regplot(data=df_filtered, x='social_intensity_score', y='battery_score', scatter_kws={'alpha':0.2}, line_kws={'color':'red'}, ax=ax4)
        st.pyplot(fig4)
        
        st.markdown("""
        **Keterangan:** Terdapat bukti empiris bahwa variabel intensitas sosial memiliki pengaruh negatif yang signifikan terhadap sisa cadangan energi. Semakin tinggi beban intensitas interaksi, semakin cepat laju *exhaustion* terjadi.
        """)
        st.success("✅ **Solusi Strategis:** Kurangi keterlibatan dalam interaksi sosial dengan intensitas atau keramaian tinggi secara berturut-turut. Lakukan regulasi diri dalam memilih kualitas lingkungan sosial guna menjaga stabilitas kapasitas kognitif.")
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
        **Keterangan:** Visualisasi komparatif ini menunjukkan bahwa manajemen durasi di bawah 15 jam secara signifikan mampu menekan risiko kelelahan ekstrem dibandingkan kelompok berisiko.
        """)
        st.success("✅ **Solusi Strategis:** Implementasikan strategi pembatasan waktu aktif sebagai intervensi utama. Pengguna disarankan untuk tetap berada pada pola durasi 'Terkontrol' untuk memastikan daya tahan energi mental yang berkelanjutan dalam jangka panjang.")

    st.divider()
    st.info("💡 **Catatan Akademis:** Dashboard ini mendeteksi pola kelelahan secara real-time. Gunakan solusi strategis yang muncul sebagai langkah preventif terhadap risiko Academic Burnout.")

except Exception as e:
    st.error(f"⚠️ Terjadi hambatan teknis: {e}")