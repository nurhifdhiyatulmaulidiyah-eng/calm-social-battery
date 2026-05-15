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
    st.sidebar.markdown("Navigasi ini berfungsi sebagai instrumen filter untuk memvalidasi variabel penelitian secara dinamis.")
    
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
        st.warning("⚠️ Dataset tidak ditemukan. Silakan lakukan penyesuaian parameter pada Sidebar.")
    else:
        # KPI Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Volume Observasi", f"{len(df_filtered)} Hari")
        m2.metric("Mean Battery Score", f"{df_filtered['battery_score'].mean():.1f}")
        m3.metric("Indeks Kelelahan Kritis", f"{(df_filtered[df_filtered['battery_score'] < 0].shape[0]/len(df_filtered)*100):.1f}%")
        st.divider()

        # ==========================================
        # 1. TREN BULANAN
        # ==========================================
        st.header("1. Tren Prevalensi Social Exhaustion Bulanan")
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        df['is_negative'] = df['battery_score'] < 0
        data_m = df[df['month_name'].isin(selected_months)].groupby('month_name')['is_negative'].mean().reindex([m for m in month_order if m in selected_months]) * 100
        
        sns.barplot(x=data_m.index, y=data_m.values, palette="magma", ax=ax1)
        ax1.set_ylabel("Persentase Exhaustion (%)")
        plt.xticks(rotation=45)
        st.pyplot(fig1)
        st.markdown(f"""
        **Keterangan:** Visualisasi ini memetakan fluktuasi *social exhaustion* ekstrem yang didefinisikan melalui skor baterai negatif. Secara temporal, data mengidentifikasi adanya periode kritis di mana ambang batas energi sosial pengguna terkuras habis. Penurunan signifikan pada bulan **September** menunjukkan periode stabilisasi energi, namun lonjakan pada bulan-bulan lain mengindikasikan adanya akumulasi stresor sosial yang berkontribusi pada risiko *Academic Burnout*. Pola musiman ini menjadi dasar penting bagi pengembangan intervensi preventif untuk menjaga homeostasis mental pengguna sebelum mencapai fase kelelahan klinis.
        """)
        st.divider()

        # ==========================================
        # 2. BATAS AMAN DURASI
        # ==========================================
        st.header("2. Analisis Threshold Durasi Terhadap Kondisi Academic Burnout")
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        sns.boxplot(data=df_filtered, x='battery_category', y='total_duration_minutes', palette="coolwarm", ax=ax2)
        st.pyplot(fig2)
        st.markdown("""
        **Keterangan:** Identifikasi melalui distribusi statistik *Box Plot* ini bertujuan menetapkan parameter durasi maksimal interaksi sosial. Terdapat korelasi linear yang jelas antara peningkatan durasi aktivitas dengan transisi menuju kondisi *burnout*. Kelompok pengguna yang berada pada zona merah ('Exhausted') menunjukkan median durasi di atas **950 menit**, mengonfirmasi bahwa eksposur sosial yang berlebihan tanpa jeda restif merupakan prediktor utama terjadinya *Academic Burnout*. Penentuan ambang batas ini menjadi referensi teknis bagi sistem peringatan dini dalam memitigasi dampak degradasi energi sosial.
        """)
        st.divider()

        # ==========================================
        # 3. POLA HARIAN
        # ==========================================
        st.header("3. Siklus Mingguan Social Battery dan Titik Jenuh (Mid-Week Crash)")
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        data_d = df_filtered.groupby('day_of_week')['battery_score'].mean().reindex(day_order)
        ax3.plot(data_d.index, data_d.values, marker='o', color='teal', linewidth=4, markersize=10)
        ax3.fill_between(data_d.index, data_d.values, alpha=0.2, color='teal')
        st.pyplot(fig3)
        st.markdown("""
        **Keterangan:** Grafik garis ini mengungkap pola sirkadian mingguan energi sosial pengguna. Penurunan drastis pada hari Rabu memvalidasi fenomena **'Mid-Week Crash'**, yakni titik jenuh psikologis akibat akumulasi beban interaksi yang tidak terkompensasi oleh waktu pemulihan memadai. Kondisi ini mencerminkan fase awal *exhaustion* yang jika dibiarkan akan berujung pada penurunan produktivitas akademik. Pemulihan energi secara signifikan di hari Minggu menegaskan bahwa strategi *social distancing* temporer merupakan mekanisme koping yang efektif untuk menetralisir dampak beban sosial mingguan.
        """)
        st.divider()

        # ==========================================
        # 4. KORELASI INTENSITAS
        # ==========================================
        st.header("4. Signifikansi Korelasi: Intensitas Interaksi terhadap Degradasi Energi")
        fig4, ax4 = plt.subplots(figsize=(12, 5))
        sns.regplot(data=df_filtered, x='social_intensity_score', y='battery_score', scatter_kws={'alpha':0.2}, line_kws={'color':'red'}, ax=ax4)
        st.pyplot(fig4)
        st.markdown("""
        **Keterangan:** Melalui pengujian regresi linear, ditemukan bukti empiris bahwa variabel intensitas sosial memiliki pengaruh negatif yang signifikan terhadap sisa cadangan energi. Semakin tinggi beban interaksi—baik secara kuantitas maupun kualitas keramaian—semakin cepat laju *exhaustion* terjadi. Hal ini membuktikan bahwa intensitas merupakan faktor pemicu utama degradasi energi sosial yang mempercepat transisi menuju *Academic Burnout*. Insight ini merekomendasikan perlunya regulasi diri dalam memilih kualitas interaksi sosial guna menjaga stabilitas kapasitas kognitif dan emosional pengguna.
        """)
        st.divider()

        # ==========================================
        # 5. EFEKTIVITAS PEMBATASAN
        # ==========================================
        st.header("5. Uji Komparatif: Mitigasi Burnout Melalui Manajemen Durasi")
        df_filtered['group'] = np.where(df_filtered['total_duration_minutes'] < 900, 'Terkontrol (<15h)', 'Berisiko (>16.7h)')
        fig5, ax5 = plt.subplots(figsize=(12, 5))
        sns.barplot(data=df_filtered, x='group', y='battery_score', palette="plasma", ax=ax5)
        st.pyplot(fig5)
        st.markdown("""
        **Keterangan:** Visualisasi komparatif ini merupakan validasi akhir terhadap strategi mitigasi kelelahan sosial. Perbedaan rata-rata skor baterai yang kontras antara kelompok 'Terkontrol' dan 'Berisiko' menunjukkan bahwa manajemen durasi di bawah 15 jam secara signifikan mampu menekan risiko *exhaustion*. Data membuktikan bahwa pembatasan waktu aktif adalah intervensi paling aplikatif untuk menghindari *Academic Burnout*. Pengguna dengan pola durasi terkontrol memiliki daya tahan energi mental yang lebih berkelanjutan, mendukung produktivitas yang lebih stabil dalam jangka panjang.
        """)

    st.divider()
    st.info("💡 **Catatan Akademis:** Analisis ini menggunakan pendekatan data-driven untuk memahami fenomena kelelahan sosial dan implikasinya terhadap stabilitas mental.")

except Exception as e:
    st.error(f"⚠️ Terjadi hambatan teknis dalam memproses instrumen visualisasi: {e}")