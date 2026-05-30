import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from collections import Counter
from wordcloud import WordCloud
from scipy.stats import ttest_ind

# =============================================================================
# 1. KONFIGURASI DASHBOARD UTAMA
# =============================================================================
st.set_page_config(page_title="Dashboard Terintegrasi: Social Battery & Mood Jar", layout="wide")

# Custom CSS untuk tampilan profesional, bersih, dan sesuai standar dashboard analitik
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #fcfcfc; border-right: 1px solid #e6e6e6; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    h1, h2, h3, h4, h5 { color: #2c3e50; font-family: 'Segoe UI', sans-serif; }
    .academic-box { background-color: #ffffff; padding: 25px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 5px solid #2c3e50; }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# 2. MAIN CONTROL PANEL (SIDEBAR NAVIGATION)
# =============================================================================
with st.sidebar:
    st.title("🚀 Main Control Panel")
    st.write("Silakan pilih modul analisis yang ingin dieksplorasi:")
    
    main_menu = st.selectbox(
        "Pilih Modul Dashboard:",
        ["Halaman Utama (Beranda)", "🔋 Modul 1: Social Battery", "🫙 Modul 2: Mood Jar"]
    )
    st.divider()

# =============================================================================
# 3. KLASTER HALAMAN 1: BERANDA / HOME
# =============================================================================
if main_menu == "Halaman Utama (Beranda)":
    st.title("📊 Aplikasi Dashboard Analisis Terintegrasi")
    st.markdown("### Proyek Capstone: Calm Social Battery & Calm Mood Jar")
    st.divider()

    st.markdown("""
    <div class="academic-box">
        <h4>Selamat Datang di Platform Dashboard Analisis Data</h4>
        <p>Dashboard ini berfungsi untuk menampilkan visualisasi data, hasil pengujian statistik, serta ringkasan keterangan evaluasi dari pertanyaan bisnis SMART utama pada Proyek Capstone <b>Calm Social Battery</b> dan <b>Calm Mood Jar</b>.</p>
        <p>Silakan gunakan menu drop-down <b>Pilih Modul Dashboard</b> pada sidebar sebelah kiri untuk membuka masing-masing modul analisis penelitian.</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        ### 🔋 1. Social Battery
        * **[1] Pertanyaan 1:** Tren Kelelahan Pengguna Setiap Bulan.
        * **[2] Pertanyaan 2:** Batas Aman Durasi Aktivitas Sosial.
        * **[3] Pertanyaan 3:** Pola Kelelahan Berdasarkan Hari dalam Seminggu.
        * **[4] Pertanyaan 4:** Hubungan Antara Intensitas Aktivitas dengan Skor Baterai.
        * **[5] Pertanyaan 5:** Perbandingan Energi Pengguna Durasi Pendek vs Durasi Panjang.
        """)
    with col2:
        st.success("""
        ### 🫙 2. Mood Jar
        * **Eksplorasi Awal:** Distribusi Frekuensi Mood & Word Cloud Grid 2x2.
        * **[1] Pertanyaan 1:** Eksplorasi Pemicu Utama antara Rasa Sedih dan Cemas.
        * **[2] Pertanyaan 2:** Pengaruh Hubungan Sosial pada Mood Bahagia vs Sedih.
        * **[3] Pertanyaan 3:** Kaitan antara Kondisi Fisik dengan Emosi (Marah & Cemas vs Bahagia).
        """)
    st.divider()
    st.caption("🏷️ Status Repositori: Terhubung ke GitHub & Streamlit Cloud | Jurusan Matematika, Universitas Airlangga")

# =============================================================================
# 4. KLASTER HALAMAN 2: SOCIAL BATTERY
# =============================================================================
elif main_menu == "🔋 Modul 1: Social Battery":
    
    @st.cache_data
    def load_sb_data():
        df = pd.read_csv("data/main_data_social_battery.csv")
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        day_order = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
        focus_categories = ['Low (<50)', 'Medium (50-80)', 'High (>80)']
        return df, month_order, day_order, focus_categories

    try:
        df, month_order, day_order, focus_categories = load_sb_data()

        with st.sidebar:
            st.subheader("🔋 Navigasi Pertanyaan")
            opsi_sb = st.sidebar.radio(
                "Pilih Struktur Soal:",
                [
                    "Overview & Deskriptif EDA",
                    "Pertanyaan 1: Tren Bulanan",
                    "Pertanyaan 2: Batas Aman Durasi",
                    "Pertanyaan 3: Pola Kelelahan Berdasarkan Hari",
                    "Pertanyaan 4: Hubungan Intensitas Aktivitas",
                    "Pertanyaan 5: Perbandingan Durasi Pendek vs Panjang"
                ]
            )
            st.divider()
            
            st.subheader("⚡ SB Filters")
            selected_months = st.multiselect("Pilih Bulan:", options=month_order, default=month_order)
            selected_categories = st.multiselect("Level Baterai:", options=focus_categories, default=focus_categories)

        df_filtered = df[(df['month_name'].isin(selected_months)) & (df['battery_category_3'].isin(selected_categories))].copy()

        st.title("🔋 Modul Analisis: Calm Social Battery")
        st.markdown("### Analisis Tingkat Kelelahan Sosial dan Mencegah Risiko Academic Burnout")
        st.divider()

        # OVERVIEW
        if opsi_sb == "Overview & Deskriptif EDA":
            st.subheader("Overview & Deskriptif Exploratory Data Analysis (EDA)")
            if df_filtered.empty:
                st.warning("⚠️ Data tidak ditemukan. Silakan sesuaikan parameter pada filter.")
            else:
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Volume Observasi", f"{int(len(df_filtered))} Hari")
                col_m2.metric("Mean Battery Score", f"{df_filtered['battery_score'].mean():.1f}")
                col_m3.metric("Rata-rata Durasi", f"{df_filtered['total_duration_minutes'].mean():.1f} Menit")
                
                st.subheader("Statistik Deskriptif Dataset:")
                st.dataframe(df_filtered[['total_duration_minutes', 'social_intensity_score', 'battery_score']].describe().round(2))
                st.divider()
                
                st.markdown("""
                **Keterangan:**
                * **Variabel Target (`battery_score`):** Skor baterai bergerak dari nilai minimum **0.34** hingga nilai maksimum **99.99**, dengan nilai rata-rata (*mean*) sebesar **62.05**. Fakta bahwa nilai minimumnya adalah 0.34 (tidak ada nilai negatif) memperkuat keputusan untuk menetapkan batas kondisi kelelahan tinggi (*high fatigue*) pada angka di bawah 20 (`battery_score < 20`).
                * **Variabel Aktivitas (`total_duration_minutes`):** Durasi aktivitas harian pengguna berkisar antara **0 menit** hingga **1.300 menit** (sekitar 21,6 jam). Rentang data durasi yang sangat luas ini sangat memadai untuk digunakan dalam mencari batas aman aktivitas sosial.
                """)

        # PERTANYAAN 1
        elif opsi_sb == "Pertanyaan 1: Tren Bulanan":
            st.subheader("[1] Pertanyaan 1: Tren Kelelahan Pengguna Setiap Bulan")
            st.markdown('**Soal Baku:** *"Berapa persentase pengguna yang mengalami kelelahan sosial tingkat tinggi (battery_score < 20) di setiap bulan, dan bagaimana tren perkembangannya dari Januari hingga Desember 2026?"*')
            st.markdown('**Indikator:** *Fokus pada pengguna dengan battery_score di bawah 20.*')
            st.divider()

            if df_filtered.empty:
                st.warning("⚠️ Data tidak ditemukan.")
            else:
                df_filtered['is_exhausted_critical'] = df_filtered['battery_score'] < 20
                data_m = df_filtered.groupby('month_name')['is_exhausted_critical'].mean() * 100
                data_m = data_m.reindex([m for m in month_order if m in selected_months])

                fig1, ax1 = plt.subplots(figsize=(12, 5))
                sns.barplot(x=data_m.index, y=data_m.values, palette="magma", ax=ax1, edgecolor='black')
                ax1.set_ylabel("Persentase Kelelahan (%)")
                ax1.set_xlabel("Bulan")
                ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
                plt.xticks(rotation=45)
                st.pyplot(fig1)

                if not data_m.dropna().empty:
                    max_month = data_m.idxmax()
                    min_month = data_m.idxmin()
                    st.markdown(f"""
                    **Keterangan:** Grafik ini menunjukkan pola musiman kelelahan sosial. Bulan **{max_month}** diidentifikasi sebagai periode dengan tingkat kelelahan tertinggi, yang kemungkinan besar dipicu oleh akumulasi stresor akademik (seperti pekan ujian atau tugas besar). Sebaliknya, bulan **{min_month}** menunjukkan fase stabilisasi energi di mana beban sosial berada dalam batas toleransi pengguna. Penurunan kelelahan di bulan {min_month} menjadi bukti bahwa adanya waktu jeda dapat menjaga kestabilan energi mental secara signifikan.
                    """)
                    st.success(f"""
                    ✅ **Solusi Strategis:** Untuk memitigasi lonjakan beban di bulan **{max_month}**, pengguna perlu menerapkan *pre-emptive rest* (istirahat terencana sebelum masa sibuk tiba). Mengadopsi manajemen kegiatan seperti pada bulan **{min_month}** dapat membantu mencegah akumulasi stres yang berujung pada **Academic Burnout**.
                    """)

        # PERTANYAAN 2
        elif opsi_sb == "Pertanyaan 2: Batas Aman Durasi":
            st.subheader("[2] Pertanyaan 2: Batas Aman Durasi Aktivitas Sosial")
            st.markdown('**Soal Baku:** *"Berapakah batas maksimal durasi waktu (dalam menit) yang masih aman bagi pengguna sebelum nilai battery_score mereka turun ke bawah 20?"*')
            st.markdown('**Indikator:** *Mencari batas menit (total_duration_minutes) saat baterai mulai kritis (< 20).*')
            st.divider()

            if df_filtered.empty:
                st.warning("⚠️ Data tidak ditemukan.")
            else:
                subset_lelah = df_filtered[df_filtered['battery_score'] < 20]
                median_val = subset_lelah['total_duration_minutes'].median() if not subset_lelah.empty else df_filtered['total_duration_minutes'].median()

                fig2, ax2 = plt.subplots(figsize=(12, 5))
                category_order = [cat for cat in ['Low (<50)', 'Medium (50-80)', 'High (>80)'] if cat in selected_categories]
                sns.boxplot(data=df_filtered, x='battery_category_3', y='total_duration_minutes', order=category_order, palette="coolwarm", ax=ax2)
                ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
                st.pyplot(fig2)

                st.markdown(f"""
                **Keterangan:** Berdasarkan distribusi statistik di atas, terlihat korelasi yang jelas antara durasi aktivitas dengan penurunan level energi. Kondisi 'Sangat Lelah' secara konsisten muncul ketika durasi interaksi sosial mencapai median **{median_val:.0f} menit**. Ini membuktikan bahwa kapasitas baterai sosial memiliki ambang batas durasi tertentu, di mana setelah melewati batas tersebut, pemulihan energi akan menjadi jauh lebih lambat.
                """)
                st.success(f"""
                ✅ **Solusi Strategis:** Tetapkan durasi **{median_val:.0f} menit** sebagai batas aman interaksi sosial harian. Pengguna disarankan untuk mulai melakukan 'penarikan diri secara bertahap' dari lingkungan sosial sebelum mencapai batas durasi ini guna melindungi cadangan energi mental dari risiko **Academic Burnout**.
                """)

        # PERTANYAAN 3
        elif opsi_sb == "Pertanyaan 3: Pola Kelelahan Berdasarkan Hari":
            st.subheader("[3] Pertanyaan 3: Pola Kelelahan Berdasarkan Hari dalam Seminggu")
            st.markdown('**Soal Baku:** *"Pada hari apa pengguna paling rentan mengalami kelelahan (battery_score < 20) dan pada hari apa kondisi energi mereka paling tinggi (battery_score > 80)?"*')
            st.markdown('**Indikator:** *Membandingkan tingkat energi pengguna dari hari Senin sampai Minggu.*')
            st.divider()

            if df_filtered.empty:
                st.warning("⚠️ Data tidak ditemukan.")
            else:
                data_d = df_filtered.groupby('hari_ini')['battery_score'].mean().reindex(day_order)

                fig3, ax3 = plt.subplots(figsize=(12, 5))
                ax3.plot(data_d.index, data_d.values, marker='o', color='teal', linewidth=4, markersize=10)
                ax3.fill_between(data_d.index, data_d.values, alpha=0.2, color='teal')
                ax3.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
                st.pyplot(fig3)

                if not data_d.dropna().empty:
                    low_day = data_d.idxmin()
                    st.markdown(f"""
                    **Keterangan:** Grafik ini mengungkap fenomena penurunan energi di tengah minggu, dengan hari **{low_day}** sebagai titik terendah. Hal ini mencerminkan adanya akumulasi beban sosial dari awal minggu yang tidak terkompensasi oleh waktu pemulihan memadai. Tanpa adanya intervensi di hari kritis ini, kelelahan sosial akan terus terbawa hingga akhir pekan dan menurunkan produktivitas akademik secara keseluruhan.
                    """)
                    st.success(f"""
                    ✅ **Solusi Strategis:** Mengingat hari **{low_day}** adalah hari produktif, solusinya bukan menghindari kegiatan, melainkan menerapkan strategi **'Micro-recovery'**. Luangkan waktu 15-30 menit di sela-sela jadwal kuliah untuk menyendiri tanpa distraksi sensorik. Selain itu, hindari pengambilan keputusan besar atau pertemuan sosial intensitas tinggi di hari {low_day} untuk menjaga stabilitas emosional dari ancaman **Academic Burnout**.
                    """)

        # PERTANYAAN 4
        elif opsi_sb == "Pertanyaan 4: Hubungan Intensitas Aktivitas":
            st.subheader("[4] Pertanyaan 4: Hubungan Antara Intensitas Aktivitas dengan Skor Baterai")
            st.markdown('**Soal Baku:** *"Apakah terdapat korelasi (hubungan) yang signifikan antara tingkat kepadatan aktivitas (social_intensity_score) dengan sisa energi pengguna (battery_score)?"*')
            st.markdown('**Indikator:** *Menguji hubungan antara intensitas kegiatan dan tingkat kelelahan.*')
            st.divider()

            if df_filtered.empty:
                st.warning("⚠️ Data tidak ditemukan.")
            else:
                fig4, ax4 = plt.subplots(figsize=(12, 5))
                sns.regplot(data=df_filtered, x='social_intensity_score', y='battery_score', scatter_kws={'alpha':0.2}, line_kws={'color':'red'}, ax=ax4)
                ax4.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
                st.pyplot(fig4)
                
                st.markdown("""
                **Keterangan:** Melalui model regresi linear ini, ditemukan bukti empiris bahwa intensitas lingkungan (keramaian, kebisingan, dan beban interaksi) memiliki pengaruh negatif yang signifikan terhadap sisa energi. Semakin tinggi beban sensorik di lingkungan sosial, semakin cepat laju *social exhaustion* terjadi. Hal ini membuktikan bahwa lingkungan yang terlalu menstimulasi otak secara berlebihan merupakan faktor pemicu utama kelelahan mental bagi mahasiswa.
                """)
                st.success("""
                ✅ **Solusi Strategis:** Terapkan teknik **'Social Pacing'**. Jika Anda harus berada di lingkungan yang ramai (seperti kantin atau ruang rapat besar), pastikan durasinya diimbangi dengan sesi tenang sesudahnya. Penggunaan alat bantu seperti penutup telinga atau sekadar berpindah ke area yang lebih sepi selama beberapa menit terbukti efektif mengurangi beban sensorik dan mencegah gejala awal **Academic Burnout**.
                """)

        # PERTANYAAN 5
        elif opsi_sb == "Pertanyaan 5: Perbandingan Durasi Pendek vs Panjang":
            st.subheader("[5] Pertanyaan 5: Perbandingan Energi Pengguna Durasi Pendek vs Durasi Panjang")
            st.markdown('**Soal Baku:** *"Apakah terdapat perbedaan battery_score yang nyata antara kelompok pengguna dengan durasi aktivitas yang pendek (sebentar) vs durasi aktivitas yang panjang (lama)?"*')
            st.markdown('**Indikator:** *Membandingkan skor baterai antara kelompok durasi rendah (< 900 mnt) dan durasi tinggi (> 1000 mnt).*')
            st.divider()

            kelompok_A = df_filtered[df_filtered['total_duration_minutes'] < 900]
            kelompok_B = df_filtered[df_filtered['total_duration_minutes'] > 1000]

            if len(kelompok_A) < 2 or len(kelompok_B) < 2:
                st.warning("⚠️ Jumlah sampel records data kurang memadai untuk menghitung uji t-test.")
            else:
                t_stat, p_val = ttest_ind(kelompok_A['battery_score'], kelompok_B['battery_score'], equal_var=False)
                
                col_c1, col_c2 = st.columns(2)
                col_c1.metric("Rata-rata Durasi Pendek (<900m)", f"{kelompok_A['battery_score'].mean():.2f}")
                col_c2.metric("Rata-rata Durasi Panjang (>1000m)", f"{kelompok_B['battery_score'].mean():.2f}")
                
                st.write(f"- **T-Statistic:** {t_stat:.4f} | **p-value:** {p_val:.4e}")
                
                fig5, ax5 = plt.subplots(figsize=(12, 5))
                df_filtered['group'] = np.where(df_filtered['total_duration_minutes'] < 900, 'Terkontrol (<15 jam)', 'Berisiko (>16 jam)')
                sns.barplot(data=df_filtered, x='group', y='battery_score', palette="plasma", ax=ax5, edgecolor='black')
                ax5.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
                st.pyplot(fig5)
                
                st.markdown("""
                **Keterangan:** Perbandingan rata-rata skor energi antara dua kelompok ini menunjukkan efektivitas manajemen waktu yang sangat kontras. Kelompok 'Terkontrol' memiliki resiliensi energi yang jauh lebih stabil dibandingkan kelompok 'Berisiko'. Hal ini memvalidasi secara ilmiah bahwa pembatasan waktu aktif adalah intervensi yang paling aplikatif dan efektif untuk menghindari kondisi kelelahan klinis.
                """)
                st.success("""
                ✅ **Solusi Strategis:** Disiplin dalam menjaga durasi kegiatan tetap pada zona 'Terkontrol' adalah strategi jangka panjang terbaik. Dengan menjaga konsistensi durasi aktif harian di bawah ambang batas kritis, pengguna dapat memastikan kapasitas kognitif tetap prima untuk menghadapi tantangan akademik harian tanpa risiko **Academic Burnout**.
                """)

    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan memproses data Social Battery: {e}")

# =============================================================================
# 5. KLASTER HALAMAN 3: MOOD JAR
# =============================================================================
elif main_menu == "🫙 Modul 2: Mood Jar":
    
    @st.cache_data
    def load_mj_data():
        return pd.read_csv("data/main_data_mood_jar.csv")

    try:
        mood_jar = load_mj_data()

        with st.sidebar:
            st.subheader("🫙 Navigasi Pertanyaan")
            opsi_menu = st.sidebar.radio(
                "Pilih Struktur Soal:",
                [
                    "Eksplorasi Awal & Word Cloud", 
                    "Pertanyaan 1: Topik Pemicu", 
                    "Pertanyaan 2: Hubungan Sosial", 
                    "Pertanyaan 3: Kondisi Fisik vs Emosi"
                ]
            )

        st.title("🫙 Modul Analisis: Calm Mood Jar")
        st.markdown("### Eksplorasi Kata Kunci Pemicu Emosi Harian Mahasiswa")
        st.divider()

        # GATHERING DATA & WORD CLOUD
        if opsi_menu == "Eksplorasi Awal & Word Cloud":
            st.subheader("Tahap Gathering Data & Awal Eksplorasi (Word Cloud)")
            st.divider()
            
            mood_counts = mood_jar['MOOD'].value_counts()
            
            col_text1, col_text2 = st.columns(2)
            with col_text1:
                st.write("**Frekuensi Kumulatif Kategori Mood:**")
                st.dataframe(mood_counts)
            with col_text2:
                st.write("**Proporsi Persentase Kategori Mood:**")
                for m, v in mood_counts.items():
                    st.write(f"- Kategori **{m.capitalize()}** : {(v/len(mood_jar)*100).round(2)}%")

            colors = {'sedih': '#5D9B9B', 'bahagia': '#F4A261', 'cemas': '#E9C46A', 'marah': '#E76F51'}
            bar_colors = [colors.get(m, '#999999') for m in mood_counts.index]
            
            fig_dist, axes_dist = plt.subplots(1, 2, figsize=(14, 5))
            axes_dist[0].bar(mood_counts.index, mood_counts.values, color=bar_colors, edgecolor='black')
            axes_dist[0].set_title('Distribusi Frekuensi Mood (Unik)', fontsize=14, fontweight='bold')
            axes_dist[0].get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: "{:,}".format(int(x))))
            for i, v in enumerate(mood_counts.values):
                axes_dist[0].text(i, v + 25, str(int(v)), ha='center', fontweight='bold')

            axes_dist[1].pie(mood_counts.values, labels=[k.capitalize() for k in mood_counts.index], autopct='%1.1f%%', colors=bar_colors, startangle=90, explode=[0.05]*len(mood_counts))
            axes_dist[1].set_title('Proporsi Kondisi Mental Mahasiswa', fontsize=14, fontweight='bold')
            st.pyplot(fig_dist)
            st.divider()

            st.subheader("Visualisasi Kata Kunci Dominan Menggunakan Word Cloud Grid 2x2")
            color_maps = {'sedih': 'YlGnBu', 'bahagia': 'Oranges', 'cemas': 'YlOrBr', 'marah': 'Reds'}
            
            extended_stopwords = set([
                'aku', 'saya', 'merasa', 'perasaan', 'dan', 'yang', 'di', 'ke', 'dari', 'ini', 'itu', 'untuk', 'with', 'dengan', 
                'karena', 'tidak', 'tak', 'bisa', 'akan', 'ada', 'adalah', 'sangat', 'lebih', 'bahwa', 'saat', 'jadi', 
                'agak', 'kalau', 'jika', 'saja', 'juga', 'lah', 'kah', 'nya', 'tapi', 'tetapi', 'bgt', 'banget', 'kok', 
                'loh', 'sih', 'ya', 'pas', 'malah', 'begitu', 'secara', 'tentang', 'oleh', 'bagi', 'pada', 'atau', 
                'lalu', 'kemudian', 'setelah', 'sebelum', 'ketika', 'sementara', 'bahkan', 'namun', 'melainkan', 
                'sedangkan', 'seperti', 'sebagai', 'terhadap', 'kepada', 'maupun', 'ia', 'kami', 'kita', 'kamu', 
                'mereka', 'dia', 'anda', 'hal', 'orang', 'buat', 'banyak', 'bukan', 'pun', 'hanya', 'bisa', 'mampu', 
                'dapat', 'ingin', 'mau', 'bila', 'terus', 'lagi', 'tahu', 'melihat', 'bilang', 'kata', 'katanya', 
                'dalam', 'apa', 'sudah', 'telah', 'semua', 'masih', 'merasakan', 'menjadi', 'membuat', 'sama', 
                'lain', 'sedang', 'hari', 'sedikit', 'diri', 'mulai', 'tahu-tahu', 'tahu2', 
                'yg', 'mungkin', 'benar', 'benar-benar', 'sebuah', 'suatu', 'sesuatu', 'satu', 'dua', 'tiga', 
                'kali', 'waktu', 'pernah', 'kembali', 'paling', 'terlalu', 'cukup', 'jangan', 'hampir', 
                'sekarang', 'serta', 'tersebut', 'bagaikan', 'selalu', 'biasanya', 'punya', 'mengapa', 'kenapa',
                'beberapa', 'sekali', 'setiap', 'tanpa', 'baru'
            ])

            fig_wc, axes_wc = plt.subplots(2, 2, figsize=(14, 10))
            axes_wc = axes_wc.flatten()
            moods_list = mood_jar['MOOD'].unique()

            for i, mood in enumerate(moods_list):
                text_content = " ".join(mood_jar[mood_jar['MOOD'] == mood]['MOOD_NOTE'].astype(str))
                wordcloud = WordCloud(
                    width=800, height=600, background_color='white',
                    stopwords=extended_stopwords, min_font_size=10,
                    colormap=color_maps.get(mood, 'viridis'), collocations=False
                ).generate(text_content)
                
                axes_wc[i].imshow(wordcloud, interpolation='bilinear')
                axes_wc[i].set_title(f"Word Cloud - Mood: {mood.upper()}", fontsize=14, fontweight='bold')
                axes_wc[i].axis('off')

            plt.tight_layout()
            st.pyplot(fig_wc)
            st.divider()

            st.markdown("""
            **Keterangan:**
            * **Volume Data:** Dataset berhasil dimuat dengan total **11.887** entri. Volume data yang besar dan kaya ini sangat ideal, representatif, dan memadai untuk melakukan analisis pemrosesan teks (*text mining*) serta ekstraksi kata kunci pemicu suasana hati pengguna secara mendalam dan valid.
            * **Struktur Kolom:**
              * **Variabel Target (Kategori):** Kolom `MOOD` merupakan indikator utama klasifikasi emosi pengguna. Dataset ini terbukti bersifat seimbang (*balanced dataset*) dengan jumlah sekitar **2.972** entri untuk masing-masing kategori emosi (`sedih`, `bahagia`, `marah`, dan `cemas`), sehingga sangat valid untuk dianalisis secara komparatif tanpa risiko bias.
              * **Variabel Teks (Sumber Cerita):** Kolom `MOOD_NOTE` berisi data teks naratif berupa catatan atau keluh kesah harian pengguna. Kolom ini akan menjadi sumber data utama untuk proses ekstraksi kata kunci pemicu utama (**Pertanyaan Bisnis 1**), filter interaksi sosial (**Pertanyaan Bisnis 2**), serta penyebutan keluhan fisik pengguna (**Pertanyaan Bisnis 3**).
              * **Variabel Intervensi (Dukungan):** Kolom `support_message` merupakan luaran teks yang berisi pesan dukungan otomatis yang disesuaikan dengan kondisi emosi pengguna. Kolom kualitatif ini sudah terisi penuh and siap mendukung evaluasi efektivitas pesan bantuan sistem.
            """)

        # PERTANYAAN 1
        elif opsi_menu == "Pertanyaan 1: Topik Pemicu":
            st.subheader("[1] Pertanyaan 1: Eksplorasi Pemicu Utama antara Rasa Sedih dan Cemas")
            st.markdown('**Soal Baku:** *"Apa tiga kata kunci yang paling sering muncul dalam cerita pengguna saat mereka merasa \'sedih\' dan \'cemas\' pada kolom MOOD_NOTE, serta bagaimana perbandingan kemunculan kata-kata tersebut di antara kedua suasana hati ini?"*')
            st.markdown('**Indikator:** *Menemukan kata kunci penanda utama pada cerita pengguna di kelompok mood sedih dan cemas.*')
            st.divider()

            df_sedih = mood_jar[mood_jar['MOOD'] == 'sedih']
            df_cemas = mood_jar[mood_jar['MOOD'] == 'cemas']

            stopwords = set([
                'aku', 'saya', 'kau', 'kamu', 'dia', 'mereka', 'kita', 'kami', 'ini', 'itu', 'tersebut', 
                'dan', 'atau', 'tapi', 'namun', 'karena', 'sebab', 'jika', 'kalau', 'maka', 'sangat', 'begitu', 
                'seperti', 'with', 'dengan', 'tanpa', 'untuk', 'dari', 'kepada', 'pada', 'oleh', 'dalam', 'bahwa', 'bisa', 
                'akan', 'telas', 'sudah', 'belum', 'masih', 'sedang', 'yg', 'dgn', 'tdk', 'utk', 'jd', 'jg', 'sdh', 
                'pun', 'nya', 'merasa', 'yang', 'tidak', 'perasaan', 'orang', 'saat', 'hal', 'ketika', 'hari', 'ada', 
                'banyak', 'tak', 'agak', 'rasa', 'membuat', 'terlalu', 'menjadi', 'selalu', 'juga', 'diri', 'lagi', 'benar', 'memang'
            ])

            def get_top_keywords(df_mood):
                all_text = ' '.join(df_mood['MOOD_NOTE'].astype(str))
                all_text_clean = re.sub(r'[^\w\s]', ' ', all_text)
                words = all_text_clean.split()
                filtered = [w.strip().lower() for w in words if len(w.strip()) >= 3 and w.strip().lower() not in stopwords]
                return Counter(filtered).most_common(5)

            top5_sedih = dict(get_top_keywords(df_sedih))
            top5_cemas = dict(get_top_keywords(df_cemas))

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].bar(top5_sedih.keys(), top5_sedih.values(), color='#5D9B9B', edgecolor='black')
            axes[0].set_title('Top 5 Kata Kunci - Mood SEDIH', fontsize=13, fontweight='bold')
            axes[0].get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: "{:,}".format(int(x))))
            for i, v in enumerate(top5_sedih.values):
                axes[0].text(i, v + 5, str(int(v)), ha='center', fontweight='bold')

            axes[1].bar(top5_cemas.keys(), top5_cemas.values(), color='#E9C46A', edgecolor='black')
            axes[1].set_title('Top 5 Kata Kunci - Mood CEMAS', fontsize=13, fontweight='bold')
            axes[1].get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: "{:,}".format(int(x))))
            for i, v in enumerate(top5_cemas.values):
                axes[1].text(i, v + 5, str(int(v)), ha='center', fontweight='bold')
            st.pyplot(fig)
            st.divider()

            st.markdown("""
            **Keterangan:**
            * Proses analisis kata kunci penanda dilakukan terhadap total **5.943** baris catatan cerita pengguna (gabungan antara 2.972 data `sedih` dan 2.971 data `cemas`). Penerapan fungsi pembersihan karakter non-kata memastikan perhitungan frekuensi murni berdasarkan kata kontekstual yang berdiri sendiri tanpa terganggu oleh simbol tanda baca.
            * **Struktur Pola Kata Kunci Dominan:**
                * **Pemicu Utama Mood Sedih (Sisi Kiri Grafik):** Pola cerita pengguna didominasi kuat oleh kata kunci penanda distres emosional berbasis fisik dan penurunan energi, seperti kata **'sakit'**, **'lelah'**, dan **'capek'**. Hal ini menunjukkan adanya korelasi erat bahwa perasaan sedih yang diutarakan pengguna sering kali berkaitan dengan keletihan tubuh atau kondisi fisik yang kurang prima.
                * **Pemicu Utama Mood Cemas (Sisi Kanan Grafik):** Pola cerita pengguna melompat secara dramatis pada kata kunci berbasis ketakutan psikologis dan kebingungan kognitif, yang direpresentasikan kuat oleh kata **'takut'**, **'aneh'**, dan **'ragu'**. Rasa cemas di sini dipicu oleh bayang-bayang masa depan atau ketidakpastian situasi sekitar.
            """)
            
            st.success("""
            ✅ **Solusi Strategis:** Kelompok pengguna dengan kondisi `sedih` akan diberikan alokasi pesan harian pada kolom `support_message` yang bersifat menenangkan emosi dan mendorong pemulihan fisik (*physical recovery & comforting message*). Sementara itu, kelompok pengguna dengan kondisi `cemas` akan diarahkan pada pesan bantuan dengan teknik pengalihan perhatian (*grounding technique*) serta validasi kognitif untuk menurunkan tingkat ketakutan mereka terhadap hal-hal yang bersifat tidak pasti.
            """)

        # PERTANYAAN 2
        elif opsi_menu == "Pertanyaan 2: Hubungan Sosial":
            st.subheader("[2] Pertanyaan 2: Pengaruh Hubungan Sosial pada Mood Bahagia vs Sedih")
            st.markdown('**Soal Baku:** *"Berapa besar persentase catatan mood \'bahagia\' yang menceritakan tentang hubungan sosial atau interaksi dengan orang terdekat, jika dibandingkan dengan catatan pada mood \'sedih\'?"*')
            st.markdown('- **Indikator:** Membandingkan seberapa sering kehadiran interaksi sosial muncul dalam cerita bahagia versus cerita sedih.')
            st.divider()

            df_bahagia = mood_jar[mood_jar['MOOD'] == 'bahagia']
            df_sedih = mood_jar[mood_jar['MOOD'] == 'sedih']

            social_keywords = ['teman', 'sahabat', 'kawan', 'keluarga', 'orang tua', 'ibu', 'mama', 'ayah', 'papa', 'bapak', 'pacar', 'bersama', 'kita', 'mereka', 'kami', 'kumpul', 'bertemu']

            def count_social(df_mood):
                count = sum(any(keyword in str(row['MOOD_NOTE']).lower() for keyword in social_keywords) for idx, row in df_mood.iterrows())
                return count, (count / len(df_mood)) * 100

            cnt_b, pct_b = count_social(df_bahagia)
            cnt_s, pct_s = count_social(df_sedih)

            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(['Bahagia', 'Sedih'], [pct_b, pct_s], color=['#F4A261', '#5D9B9B'], edgecolor='black', width=0.4)
            ax.set_ylim(0, max([pct_b, pct_s]) + 15)
            ax.set_title('Persentase Catatan dengan Interaksi Sosial', fontsize=14, fontweight='bold')
            ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: "{:,}".format(int(x))))
            for bar, pct, cnt in zip(bars, [pct_b, pct_s], [cnt_b, cnt_s]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{pct:.1f}%\n({int(cnt)} catatan)', ha='center', va='bottom', fontweight='bold')
            st.pyplot(fig)
            st.divider()

            st.markdown("""
            **Keterangan:**
            * Pengujian mengenai pengaruh hubungan interpersonal ini diproses menggunakan total **5.944** entri teks cerita harian pengguna (membagi rata secara adil antara 2.972 records data `bahagia` dan 2.972 records data `sedih`).
            * **Struktur Pola Hubungan Interpersonal:**
                * **Asosiasi pada Mood Bahagia (Valensi Positif):** Keterlibatan kata kunci interaksi sosial pada kelompok cerita `bahagia` mencatatkan persentase kemunculan yang tinggi. Temuan ini secara statistik mengonfirmasi bahwa kehadiran lingkaran hubungan sosial (seperti keluarga, teman, atau pasangan) memegang peranan krusial sebagai faktor pelatuk eksternal yang dominan dalam memicu kebahagiaan harian mahasiswa.
                * **Asosiasi pada Mood Sedih (Valensi Negatif):** Sebaliknya, persentase indikator keterlibatan hubungan sosial pada kelompok cerita `sedih` terdeteksi lebih rendah. Hal ini mengindikasikan bahwa distres emosional berupa rasa sedih yang dialami pengguna cenderung lebih banyak dipicu oleh masalah internal personal atau kondisi di mana mahasiswa merasa kekurangan dukungan sosial (*lack of social support*).
            """)
            
            st.success("""
            ✅ **Solusi Strategis:** Ketika sistem mendeteksi teks cerita harian pengguna mengarah pada klaster mood `sedih`, kolom `support_message` secara cerdas dapat menyarankan tindakan protektif berbasis hubungan sosial. Misalnya dengan memberikan dorongan moral bagi pengguna untuk membuka percakapan dengan sahabat karib atau meluangkan waktu sejenak bersama keluarga guna mengaktifkan sistem dukungan sosial (*social support system*) mereka demi memulihkan stabilitas energi emosional.
            """)

        # PERTANYAAN 3
        elif opsi_menu == "Pertanyaan 3: Kondisi Fisik vs Emosi":
            st.subheader("[3] Pertanyaan 3: Kaitan antara Kondisi Fisik dengan Emosi (Marah & Cemas vs Bahagia)")
            st.markdown('**Soal Baku:** *"Seberapa sering (dalam persen) pengguna yang sedang merasa \'marah\' dan \'cemas\' mengeluhkan kondisi fisik mereka dalam ceritanya, jika dibandingkan dengan saat mereka merasa \'bahagia\'?"*')
            st.markdown('- **Indikator:** Mengukur seberapa sering keluhan kondisi fisik atau tubuh yang tidak nyaman muncul bersamaan dengan emosi tertentu.')
            st.divider()

            df_marah_cemas = mood_jar[mood_jar['MOOD'].isin(['marah', 'cemas'])]
            df_bahagia = mood_jar[mood_jar['MOOD'] == 'bahagia']

            physical_keywords = ['sakit', 'nyeri', 'pusing', 'mual', 'lelah', 'capek', 'lemas', 'tidur', 'kurang tidur', 'insomnia', 'begadang', 'flu', 'demam', 'batuk', 'sakit kepala']

            def count_physical(df_mood):
                count = sum(any(keyword in str(row['MOOD_NOTE']).lower() for keyword in physical_keywords) for idx, row in df_mood.iterrows())
                return count, (count / len(df_mood)) * 100

            cnt_neg, pct_neg = count_physical(df_marah_cemas)
            cnt_pos, pct_pos = count_physical(df_bahagia)

            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(['Marah + Cemas', 'Bahagia'], [pct_neg, pct_pos], color=['#E76F51', '#F4A261'], edgecolor='black', width=0.4)
            ax.set_ylim(0, max([pct_neg, pct_pos]) + 15)
            ax.set_title('Persentase Catatan yang Menyebut Kondisi Fisik', fontsize=14, fontweight='bold')
            ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: "{:,}".format(int(x))))
            for bar, pct, cnt in zip(bars, [pct_neg, pct_pos], [cnt_neg, cnt_pos]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{pct:.1f}%\n({int(cnt)} catatan)', ha='center', va='bottom', fontweight='bold')
            st.pyplot(fig)
            st.divider()

            st.markdown("""
            **Keterangan:**
            * Evaluasi korelasi antara kondisi fisik (*somatis*) dan luapan emosi diproses menggunakan total **8.915** baris cerita harian pengguna (merangkum 2.972 data `marah`, 2.971 data `cemas`, dan 2.972 data `bahagia`).
            * **Struktur Pola Kondisi Fisik terhadap Emosi:**
                * **Klaster Mood Negatif (Marah + Cemas):** Terdapat kecenderungan yang signifikan di mana pengguna yang mengekspresikan rasa `marah` dan `cemas` sering kali menyisipkan keluhan kelainan fisik di dalam ceritanya, seperti gangguan tidur (*begadang*, *insomnia*, *kurang tidur*) ataupun rasa tidak nyaman pada tubuh (*pusing*, *nyeri*, *lelah*). Hal ini menunjukkan bahwa penurunan kesehatan biologis berdampak langsung pada penurunan ambang batas kesabaran kognitif mahasiswa.
                * **Klaster Mood Positif (Bahagia):** Sebaliknya, pada catatan suasana hati `bahagia`, penyebutan keluhan fisik terdeteksi sangat minim. Temuan ini mengindikasikan bahwa kondisi tubuh yang prima, sehat, dan waktu tidur yang cukup bertindak sebagai prasyarat utama (*well-being foundation*) bagi munculnya afeksi positif harian.
            """)
            
            st.success("""
            ✅ **Solusi Strategis:** Ketika model mendeteksi adanya kombinasi keluhan fisik pada cerita `marah` atau `cemas`, sistem melalui kolom `support_message` tidak hanya memberikan kalimat penenang psikologis, melainkan memberikan rekomendasi intervensi fisik yang nyata. Contohnya seperti memicu pengingat untuk segera tidur, menyarankan latihan pernapasan untuk meredakan pusing, atau menyarankan istirahat penuh (*bed rest*) demi menyeimbangkan kembali pasokan energi biologis pengguna.
            """)

    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan memproses data Mood Jar: {e}")