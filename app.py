import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from collections import Counter
from wordcloud import WordCloud
from scipy.stats import ttest_ind, pearsonr

# =============================================================================
# KONFIGURASI DASHBOARD UTAMA
# =============================================================================
st.set_page_config(page_title="Dashboard Terintegrasi: Social Battery & Mood Jar", layout="wide")

st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #fcfcfc; border-right: 1px solid #e6e6e6; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    h1, h2, h3, h4, h5 { color: #2c3e50; font-family: 'Segoe UI', sans-serif; }
    .academic-box { background-color: #ffffff; padding: 25px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 5px solid #2c3e50; }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# SIDEBAR NAVIGASI MODUL
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
# HALAMAN BERANDA
# =============================================================================
if main_menu == "Halaman Utama (Beranda)":
    st.title("📊 Dashboard Analisis Calm")
    st.markdown("### Proyek Capstone Calm: Social Battery & Mood Jar")
    st.divider()
    st.markdown("""
    <div class="academic-box">
        <h4>Selamat Datang di Platform Dashboard Analisis Data</h4>
        <p>Dashboard ini menampilkan visualisasi data, hasil pengujian statistik, serta ringkasan evaluasi dan solusi dari pertanyaan bisnis SMART pada <b>Proyek Capstone Calm: Social Battery dan Mood Jar</b>.</p>
        <p>Gunakan menu <b>Pilih Modul Dashboard</b> pada sidebar untuk membuka modul analisis.</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        ### 🔋 1. Social Battery
        * **[1]** Tren Kelelahan Pengguna Setiap Bulan
        * **[2]** Batas Aman Durasi Aktivitas Sosial
        * **[3]** Pola Kelelahan Berdasarkan Hari dalam Seminggu
        * **[4]** Hubungan Intensitas Aktivitas dengan Skor Baterai
        * **[5]** Perbandingan Energi Durasi Pendek vs Durasi Panjang
        """)
    with col2:
        st.success("""
        ### 🫙 2. Mood Jar
        * **Eksplorasi Awal:** Distribusi Frekuensi Mood & Word Cloud Grid 2x2
        * **[1]** Eksplorasi Pemicu Utama Sedih dan Cemas
        * **[2]** Pengaruh Hubungan Sosial pada Mood Bahagia vs Sedih
        * **[3]** Kaitan Kondisi Fisik dengan Emosi
        """)
    st.divider()
    st.caption("🏷️ Capstone Project Calm (PSU122, Coding Camp 2026)")

# =============================================================================
# MODUL 1: SOCIAL BATTERY
# =============================================================================
elif main_menu == "🔋 Modul 1: Social Battery":

    @st.cache_data
    def load_sb_data():
        df = pd.read_csv("main_data_social_battery.csv")
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if 'month_name' not in df.columns:
            df['month_name'] = df['date'].dt.month_name()
        if 'hari_ini' not in df.columns:
            pemetaan = {
                'Monday': 'Senin', 'Tuesday': 'Selasa', 'Wednesday': 'Rabu',
                'Thursday': 'Kamis', 'Friday': 'Jumat', 'Saturday': 'Sabtu', 'Sunday': 'Minggu'
            }
            df['hari_ini'] = df['date'].dt.day_name().map(pemetaan)
        if 'battery_category_3' not in df.columns:
            df['battery_category_3'] = pd.cut(
                df['battery_score'],
                bins=[0, 49.99, 80, 100],
                labels=['Low (<50)', 'Medium (50-80)', 'High (>80)']
            )
        if 'is_low_battery' not in df.columns:
            df['is_low_battery'] = df['battery_score'] < 50
        if 'is_high_battery' not in df.columns:
            df['is_high_battery'] = df['battery_score'] > 80
        if 'is_exhausted' not in df.columns:
            df['is_exhausted'] = df['battery_score'] < 20
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        day_order = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
        return df, month_order, day_order

    try:
        df, month_order, day_order = load_sb_data()

        st.title("🔋 Modul Analisis: Calm Social Battery")
        st.markdown("### Analisis Tingkat Kelelahan Sosial dan Pencegahan Risiko Academic Burnout")
        st.divider()

        # ── OVERVIEW & EDA ──────────────────────────────────────────────────
        st.subheader("📋 Overview & Deskriptif Exploratory Data Analysis (EDA)")

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Volume Observasi", f"{len(df):,} Records")
        col_m2.metric("Mean Battery Score", f"{df['battery_score'].mean():.1f}")
        col_m3.metric("Rata-rata Durasi", f"{df['total_duration_minutes'].mean():.1f} Menit")

        st.write("**Statistik Deskriptif Dataset:**")
        st.dataframe(df[['total_duration_minutes', 'social_intensity_score', 'battery_score']].describe().round(2))

        fig_eda, axes_eda = plt.subplots(1, 3, figsize=(15, 4))
        axes_eda[0].hist(df['battery_score'], bins=30, edgecolor='black', color='skyblue')
        axes_eda[0].axvline(x=50, color='orange', linestyle='--', label='Batas Low (<50)')
        axes_eda[0].axvline(x=80, color='green', linestyle='--', label='Batas High (>80)')
        axes_eda[0].set_xlabel('Battery Score')
        axes_eda[0].set_ylabel('Frekuensi')
        axes_eda[0].set_title('Distribusi Battery Score')
        axes_eda[0].legend()

        box = axes_eda[1].boxplot(df['battery_score'], vert=True, patch_artist=True)
        box['boxes'][0].set_facecolor('lightgreen')
        axes_eda[1].set_ylabel('Battery Score')
        axes_eda[1].set_title('Boxplot Battery Score')
        axes_eda[1].axhline(y=50, color='orange', linestyle='--')
        axes_eda[1].axhline(y=80, color='green', linestyle='--')

        axes_eda[2].hist(df['total_duration_minutes'], bins=30, edgecolor='black', color='coral')
        axes_eda[2].set_xlabel('Total Durasi (menit)')
        axes_eda[2].set_ylabel('Frekuensi')
        axes_eda[2].set_title('Distribusi Total Durasi Sosial')
        plt.tight_layout()
        st.pyplot(fig_eda)
        plt.close(fig_eda)

        st.markdown("""
        **Keterangan:**
        * **`battery_score`** bergerak dari minimum **0.34** hingga **99.99**, rata-rata **~62**. Kondisi kelelahan kritis ditetapkan pada `< 20`.
        * **`total_duration_minutes`** berkisar **0 – 1.300 menit** (~21,6 jam), rentang luas ini memadai untuk mencari batas aman aktivitas sosial.
        """)
        st.divider()

        # ── PERTANYAAN 1 ────────────────────────────────────────────────────
        st.subheader("📌 [1] Tren Kelelahan Pengguna Setiap Bulan")
        st.markdown('**Soal:** *"Berapa persentase pengguna yang mengalami kelelahan sosial tingkat tinggi (battery_score < 20) di setiap bulan, dan bagaimana tren perkembangannya dari Januari hingga Desember 2026?"*')
        st.markdown('**Indikator:** Fokus pada pengguna dengan battery_score di bawah 20.')

        # PERBAIKAN: gunakan is_low_battery + nunique user_id sesuai notebook asli
        monthly_stats = df.groupby('month_name', observed=False).agg(
            total_users=('user_id', 'nunique'),
            low_battery_count=('is_low_battery', 'sum')
        )
        monthly_stats['low_battery_pct'] = (
            monthly_stats['low_battery_count'] /
            monthly_stats['total_users'].replace(0, np.nan)
        ) * 100
        monthly_stats = monthly_stats.reindex(month_order)
        monthly_stats = monthly_stats.dropna(subset=['low_battery_pct'])

        if not monthly_stats.empty:
            avg_val = monthly_stats['low_battery_pct'].mean()
            max_month = monthly_stats['low_battery_pct'].idxmax()
            min_month = monthly_stats['low_battery_pct'].idxmin()

            fig1, ax1 = plt.subplots(figsize=(14, 6))
            norm_vals = monthly_stats['low_battery_pct'] / monthly_stats['low_battery_pct'].max()
            colors_p1 = plt.cm.RdYlGn_r(norm_vals)
            bars1 = ax1.bar(monthly_stats.index, monthly_stats['low_battery_pct'],
                            color=colors_p1, edgecolor='black', linewidth=1)
            ax1.axhline(y=avg_val, color='blue', linestyle='--',
                        label=f'Rata-rata ({avg_val:.2f}%)')
            ax1.set_xlabel('Bulan', fontsize=12)
            ax1.set_ylabel('Persentase Low Battery (< 50) (%)', fontsize=12)
            ax1.set_title('Tren Persentase Kelompok Low Battery Bulanan (Januari – Desember 2026)',
                          fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.set_ylim(0, monthly_stats['low_battery_pct'].max() + 10)
            for bar, pct in zip(bars1, monthly_stats['low_battery_pct']):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                         f'{pct:.2f}%', ha='center', va='bottom', fontsize=9)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)

            st.write("**Tabel Statistik Bulanan:**")
            tabel_p1 = monthly_stats[['total_users', 'low_battery_count', 'low_battery_pct']].copy()
            tabel_p1.columns = ['Total Pengguna Unik', 'Jumlah Low Battery', 'Persentase (%)']
            tabel_p1['Persentase (%)'] = tabel_p1['Persentase (%)'].round(2)
            st.dataframe(tabel_p1)

            st.markdown(f"""
            **Keterangan:** Bulan **{max_month}** memiliki tingkat kelelahan tertinggi,
            kemungkinan dipicu oleh akumulasi stresor akademik (pekan ujian atau tugas besar).
            Bulan **{min_month}** menunjukkan fase stabilisasi energi terbaik — bukti bahwa
            adanya waktu jeda dapat menjaga kestabilan energi mental secara signifikan.
            """)
            st.success(f"✅ **Solusi Strategis:** Terapkan *pre-emptive rest* sebelum bulan **{max_month}**. "
                       f"Adopsi pola manajemen kegiatan seperti bulan **{min_month}** untuk mencegah **Academic Burnout**.")
        else:
            st.warning("⚠️ Data tidak cukup untuk menampilkan grafik tren bulanan.")
        st.divider()

        # ── PERTANYAAN 2 ────────────────────────────────────────────────────
        st.subheader("📌 [2] Batas Aman Durasi Aktivitas Sosial")
        st.markdown('**Soal:** *"Berapakah batas maksimal durasi waktu (dalam menit) yang masih aman bagi pengguna sebelum nilai battery_score mereka turun ke bawah 50?"*')
        st.markdown('**Indikator:** Mencari batas menit saat baterai mulai masuk kondisi Low (< 50).')

        low_bat = df[df['battery_score'] < 50]
        normal_bat = df[df['battery_score'] >= 50]

        if not low_bat.empty and not normal_bat.empty:
            threshold_median = low_bat['total_duration_minutes'].median()
            threshold_p75 = low_bat['total_duration_minutes'].quantile(0.75)
            safe_threshold = normal_bat['total_duration_minutes'].quantile(0.25)

            col_p2a, col_p2b, col_p2c = st.columns(3)
            col_p2a.metric("Median Durasi (Low Battery)", f"{threshold_median:.0f} mnt")
            col_p2b.metric("Persentil 75 (Low Battery)", f"{threshold_p75:.0f} mnt")
            col_p2c.metric("Q25 Durasi (User Sehat)", f"{safe_threshold:.0f} mnt")

            fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
            axes2[0].hist(low_bat['total_duration_minutes'], bins=20, alpha=0.7,
                          label='Low Battery (<50)', color='red', edgecolor='black')
            axes2[0].hist(normal_bat['total_duration_minutes'], bins=20, alpha=0.7,
                          label='Normal/Sehat (>=50)', color='green', edgecolor='black')
            axes2[0].axvline(x=threshold_median, color='darkred', linestyle='--',
                             label=f'Median Low: {threshold_median:.0f} mnt')
            axes2[0].axvline(x=safe_threshold, color='darkgreen', linestyle='--',
                             label=f'Q25 Sehat: {safe_threshold:.0f} mnt')
            axes2[0].set_xlabel('Total Durasi (menit)')
            axes2[0].set_ylabel('Frekuensi')
            axes2[0].set_title('Distribusi Durasi: Low Battery vs Normal')
            axes2[0].legend()

            sns.boxplot(data=df, x='battery_category_3', y='total_duration_minutes',
                        order=['Low (<50)', 'Medium (50-80)', 'High (>80)'],
                        ax=axes2[1], palette='RdYlGn_r',
                        hue='battery_category_3', legend=False)
            axes2[1].set_xlabel('Kategori Battery Score')
            axes2[1].set_ylabel('Total Durasi (menit)')
            axes2[1].set_title('Durasi Sosial per Kategori Battery')
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

            st.markdown(f"""
            **Keterangan:** Kondisi Low Battery secara konsisten muncul saat durasi interaksi sosial mencapai
            median **{threshold_median:.0f} menit**. Ini membuktikan bahwa baterai sosial memiliki ambang batas
            durasi tertentu yang wajib dijaga agar kapasitas pemulihan tidak terganggu.
            """)
            st.success(f"✅ **Solusi Strategis:** Tetapkan **{int(threshold_median) - 1} menit/hari** sebagai "
                       f"batas aman interaksi sosial harian untuk mencegah **Academic Burnout**.")
        st.divider()

        # ── PERTANYAAN 3 ────────────────────────────────────────────────────
        st.subheader("📌 [3] Pola Kelelahan Berdasarkan Hari dalam Seminggu")
        st.markdown('**Soal:** *"Pada hari apa pengguna paling rentan mengalami kelelahan dan pada hari apa kondisi energi mereka paling tinggi?"*')
        st.markdown('**Indikator:** Membandingkan tingkat energi pengguna dari Senin sampai Minggu.')

        statistik_harian = df.groupby('hari_ini', observed=False).agg(
            rata_rata_baterai=('battery_score', 'mean'),
            median_baterai=('battery_score', 'median'),
            persen_low=('is_low_battery', 'mean'),
            persen_high=('is_high_battery', 'mean'),
            rata_rata_intensitas=('social_intensity_score', 'mean')
        ).round(4)
        statistik_harian['persen_low'] = statistik_harian['persen_low'] * 100
        statistik_harian['persen_high'] = statistik_harian['persen_high'] * 100
        valid_days = [d for d in day_order if d in statistik_harian.index]
        statistik_harian = statistik_harian.reindex(valid_days).dropna(subset=['rata_rata_baterai'])

        if not statistik_harian.empty:
            hari_terbaik = statistik_harian['rata_rata_baterai'].idxmax()
            hari_terburuk = statistik_harian['rata_rata_baterai'].idxmin()

            fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))

            warna = ['red' if h == hari_terburuk else 'green' if h == hari_terbaik else 'steelblue'
                     for h in statistik_harian.index]
            axes3[0, 0].bar(statistik_harian.index, statistik_harian['rata_rata_baterai'],
                            color=warna, edgecolor='black')
            axes3[0, 0].axhline(y=statistik_harian['rata_rata_baterai'].mean(), color='orange',
                                linestyle='--',
                                label=f"Rata-rata: {statistik_harian['rata_rata_baterai'].mean():.2f}")
            axes3[0, 0].set_ylabel('Rata-rata Skor Baterai')
            axes3[0, 0].set_title('Rata-rata Skor Baterai per Hari')
            axes3[0, 0].legend()

            axes3[0, 1].bar(statistik_harian.index, statistik_harian['persen_low'],
                            color='coral', edgecolor='black')
            axes3[0, 1].set_ylabel('Persentase Low Battery (%)')
            axes3[0, 1].set_title('Persentase Low Battery (< 50) per Hari')

            sns.boxplot(data=df, x='hari_ini', y='battery_score',
                        order=valid_days, ax=axes3[1, 0], palette='Set2',
                        hue='hari_ini', legend=False)
            axes3[1, 0].axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Batas Low (<50)')
            axes3[1, 0].axhline(y=80, color='green', linestyle='--', alpha=0.7, label='Batas High (>80)')
            axes3[1, 0].set_xlabel('Hari')
            axes3[1, 0].set_ylabel('Skor Baterai')
            axes3[1, 0].set_title('Distribusi Skor Baterai per Hari')
            axes3[1, 0].legend()

            axes3[1, 1].plot(list(statistik_harian.index), statistik_harian['rata_rata_baterai'],
                             'o-', linewidth=2, markersize=8, color='steelblue')
            axes3[1, 1].fill_between(list(statistik_harian.index),
                                     statistik_harian['rata_rata_baterai'], alpha=0.3, color='steelblue')
            for hari, nilai in statistik_harian['rata_rata_baterai'].items():
                axes3[1, 1].annotate(f'{nilai:.2f}', (hari, nilai),
                                     textcoords="offset points", xytext=(0, 10), ha='center')
            axes3[1, 1].set_ylabel('Rata-rata Skor Baterai')
            axes3[1, 1].set_title('Tren Naik-Turun Energi dalam Seminggu')

            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

            st.markdown(f"""
            **Keterangan:** Hari **{hari_terburuk}** teridentifikasi sebagai hari dengan energi terendah
            akibat akumulasi beban sosial. Hari **{hari_terbaik}** adalah hari dengan
            kondisi energi paling tinggi.
            """)
            st.success(f"✅ **Solusi Strategis:** Terapkan **'Micro-break'** di hari **{hari_terburuk}** "
                       f"— luangkan 15-30 menit menyendiri atau istirahat untuk mencegah **Academic Burnout**.")
        st.divider()

        # ── PERTANYAAN 4 ────────────────────────────────────────────────────
        st.subheader("📌 [4] Hubungan Intensitas Aktivitas dengan Skor Baterai")
        st.markdown('**Soal:** *"Apakah terdapat korelasi yang signifikan antara tingkat kepadatan aktivitas (social_intensity_score) dengan sisa energi pengguna (battery_score)?"*')
        st.markdown('**Indikator:** Menguji hubungan antara intensitas kegiatan dan tingkat kelelahan.')

        df_p4 = df[['social_intensity_score', 'battery_score']].dropna()
        if len(df_p4) >= 2:
            r_val, p_val_r = pearsonr(df_p4['social_intensity_score'], df_p4['battery_score'])

            if r_val < -0.7:
                kekuatan = "Sangat Kuat (Negatif)"
            elif r_val < -0.5:
                kekuatan = "Kuat (Negatif)"
            elif r_val < -0.3:
                kekuatan = "Sedang (Negatif)"
            elif r_val < 0:
                kekuatan = "Lemah (Negatif)"
            elif r_val < 0.3:
                kekuatan = "Lemah (Positif)"
            elif r_val < 0.5:
                kekuatan = "Sedang (Positif)"
            elif r_val < 0.7:
                kekuatan = "Kuat (Positif)"
            else:
                kekuatan = "Sangat Kuat (Positif)"

            col_p4a, col_p4b, col_p4c = st.columns(3)
            col_p4a.metric("Korelasi Pearson (r)", f"{r_val:.4f}")
            col_p4b.metric("p-value", f"{p_val_r:.4e}")
            col_p4c.metric("Kekuatan Korelasi", kekuatan)

            fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5))

            # Plot 1: Scatter Plot + Garis Tren Regresi Linear
            axes4[0].scatter(df_p4['social_intensity_score'], df_p4['battery_score'],
                             alpha=0.5, c='steelblue', edgecolors='white', linewidth=0.5)
            z = np.polyfit(df_p4['social_intensity_score'], df_p4['battery_score'], 1)
            p_line = np.poly1d(z)
            axes4[0].plot(df_p4['social_intensity_score'].sort_values(),
                          p_line(df_p4['social_intensity_score'].sort_values()),
                          "r--", linewidth=2, label=f'Garis Tren (r={r_val:.3f})')
            axes4[0].axhline(y=50, color='gray', linestyle='-', alpha=0.4, label='Batas Low (<50)')
            axes4[0].set_xlabel('Skor Intensitas Sosial')
            axes4[0].set_ylabel('Skor Baterai Sosial')
            axes4[0].set_title(f'Korelasi Intensitas vs Baterai\n(r = {r_val:.3f}, p = {p_val_r:.2e})',
                               fontweight='bold')
            axes4[0].legend()
            axes4[0].grid(True, alpha=0.3)

            # Plot 2: Hexbin Plot – Kepadatan Titik Data
            hb = axes4[1].hexbin(df_p4['social_intensity_score'], df_p4['battery_score'],
                                 gridsize=25, cmap='YlOrRd', bins='log')
            axes4[1].set_xlabel('Skor Intensitas Sosial')
            axes4[1].set_ylabel('Skor Baterai Sosial')
            axes4[1].set_title('Kepadatan Sebaran Data (Hexbin Plot)')
            plt.colorbar(hb, ax=axes4[1], label='Jumlah Data (Skala Log)')

            # Plot 3: Bar Rata-rata Baterai per Kelompok Intensitas
            df_p4_copy = df.copy()
            df_p4_copy['kelompok_intensitas'] = pd.cut(
                df_p4_copy['social_intensity_score'],
                bins=[0, 30, 50, 80],
                labels=['0-30 (Rendah)', '30-50 (Sedang)', '50-80 (Tinggi)']
            )
            stats_intensitas = (
                df_p4_copy.groupby('kelompok_intensitas', observed=False)['battery_score']
                .mean()
                .sort_index()
            )
            warna_intensitas = ['green', 'orange', 'red']
            axes4[2].bar(stats_intensitas.index, stats_intensitas.values,
                         color=warna_intensitas, edgecolor='black')
            axes4[2].set_ylabel('Rata-rata Skor Baterai')
            axes4[2].set_title('Rata-rata Skor Baterai per Tingkat Intensitas')
            axes4[2].set_ylim(0, 100)
            for i, (grup, nilai) in enumerate(stats_intensitas.items()):
                axes4[2].annotate(f'{nilai:.2f}', (i, nilai),
                                  textcoords="offset points", xytext=(0, 5),
                                  ha='center', fontweight='bold')

            plt.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)

            st.markdown(f"""
            **Keterangan:** Ditemukan bukti empiris bahwa intensitas lingkungan sosial memiliki
            pengaruh **negatif** terhadap sisa energi (r = {r_val:.4f}, {kekuatan}).
            Semakin tinggi beban sensorik, semakin cepat *social exhaustion* terjadi.
            """)
            st.success("✅ **Solusi Strategis:** Terapkan **'Social Pacing'** — jika harus di lingkungan "
                       "ramai, imbangi dengan 'Downtime' (sesi tenang) sesudahnya untuk mencegah **Academic Burnout**.")
        st.divider()

        # ── PERTANYAAN 5 ────────────────────────────────────────────────────
        st.subheader("📌 [5] Perbandingan Energi: Durasi Pendek vs Durasi Panjang")
        st.markdown('**Soal:** *"Apakah terdapat perbedaan battery_score yang nyata antara kelompok durasi pendek (< 900 mnt) vs durasi panjang (> 1000 mnt)?"*')
        st.markdown('**Indikator:** Membandingkan skor baterai antara kelompok durasi rendah dan durasi tinggi.')

        kelompok_A = df[df['total_duration_minutes'] < 900]
        kelompok_B = df[df['total_duration_minutes'] > 1000]

        if len(kelompok_A) >= 2 and len(kelompok_B) >= 2:
            t_stat, p_val_t = ttest_ind(kelompok_A['battery_score'], kelompok_B['battery_score'])
            std_gabungan = np.sqrt(
                (kelompok_A['battery_score'].std() ** 2 + kelompok_B['battery_score'].std() ** 2) / 2)
            cohens_d = (kelompok_A['battery_score'].mean() - kelompok_B['battery_score'].mean()) / std_gabungan

            col_p5a, col_p5b, col_p5c, col_p5d = st.columns(4)
            col_p5a.metric("Rata-rata Kel. A (<900 mnt)", f"{kelompok_A['battery_score'].mean():.2f}")
            col_p5b.metric("Rata-rata Kel. B (>1000 mnt)", f"{kelompok_B['battery_score'].mean():.2f}")
            col_p5c.metric("T-Statistic", f"{t_stat:.4f}")
            col_p5d.metric("p-value", f"{p_val_t:.4e}")

            persen_low_A = (kelompok_A['is_low_battery'].sum() / len(kelompok_A)) * 100
            persen_low_B = (kelompok_B['is_low_battery'].sum() / len(kelompok_B)) * 100

            fig5, axes5 = plt.subplots(2, 2, figsize=(14, 10))

            data_bp = [kelompok_A['battery_score'], kelompok_B['battery_score']]
            bp = axes5[0, 0].boxplot(data_bp, patch_artist=True,
                                     tick_labels=['Kelompok A\n(<900 mnt)', 'Kelompok B\n(>1000 mnt)'])
            bp['boxes'][0].set_facecolor('lightgreen')
            bp['boxes'][1].set_facecolor('lightcoral')
            axes5[0, 0].axhline(y=50, color='orange', linestyle='--', label='Batas Low (<50)')
            axes5[0, 0].set_ylabel('Skor Baterai')
            axes5[0, 0].legend()
            axes5[0, 0].set_title('Perbandingan Distribusi Skor Baterai')
            axes5[0, 0].grid(True, alpha=0.3)

            axes5[0, 1].hist(kelompok_A['battery_score'], bins=25, alpha=0.6,
                             label='Kel. A (<900 mnt)', color='green', edgecolor='black')
            axes5[0, 1].hist(kelompok_B['battery_score'], bins=25, alpha=0.6,
                             label='Kel. B (>1000 mnt)', color='red', edgecolor='black')
            axes5[0, 1].axvline(x=kelompok_A['battery_score'].mean(), color='darkgreen',
                                linestyle='--',
                                label=f"Rata-rata A: {kelompok_A['battery_score'].mean():.2f}")
            axes5[0, 1].axvline(x=kelompok_B['battery_score'].mean(), color='darkred',
                                linestyle='--',
                                label=f"Rata-rata B: {kelompok_B['battery_score'].mean():.2f}")
            axes5[0, 1].set_xlabel('Skor Baterai')
            axes5[0, 1].set_ylabel('Frekuensi')
            axes5[0, 1].set_title('Distribusi Skor Baterai (Overlay)')
            axes5[0, 1].legend()

            nilai_rata = [kelompok_A['battery_score'].mean(), kelompok_B['battery_score'].mean()]
            bars5 = axes5[1, 0].bar(
                ['Kelompok A\n(Durasi Aman)', 'Kelompok B\n(Durasi Berisiko)'],
                nilai_rata, color=['green', 'red'], edgecolor='black', width=0.5)
            axes5[1, 0].set_ylabel('Rata-rata Skor Baterai')
            axes5[1, 0].set_title('Perbandingan Nilai Rata-rata Skor Baterai')
            axes5[1, 0].set_ylim(0, 100)
            for bar, val in zip(bars5, nilai_rata):
                axes5[1, 0].text(bar.get_x() + bar.get_width() / 2, val + 2,
                                 f'{val:.2f}', ha='center', fontsize=12, fontweight='bold')

            normal_pct = [100 - persen_low_A, 100 - persen_low_B]
            axes5[1, 1].bar(['Kelompok A', 'Kelompok B'], normal_pct,
                            label='Normal/Sehat (>=50)', color='#4ecdc4', edgecolor='black', width=0.5)
            axes5[1, 1].bar(['Kelompok A', 'Kelompok B'], [persen_low_A, persen_low_B],
                            bottom=normal_pct, label='Low Battery (<50)',
                            color='#ff6b6b', edgecolor='black', width=0.5)
            axes5[1, 1].set_ylabel('Persentase (%)')
            axes5[1, 1].set_ylim(0, 115)
            axes5[1, 1].set_title('Komposisi Low Battery per Kelompok')
            axes5[1, 1].legend(loc='upper right')
            for i, (lp, np_) in enumerate(zip([persen_low_A, persen_low_B], normal_pct)):
                axes5[1, 1].text(i, np_ / 2, f'{np_:.1f}%', ha='center', va='center',
                                 fontsize=10, color='black', fontweight='bold')
                axes5[1, 1].text(i, np_ + lp / 2, f'{lp:.1f}%', ha='center', va='center',
                                 fontsize=10, color='white', fontweight='bold')

            plt.tight_layout()
            st.pyplot(fig5)
            plt.close(fig5)

            if p_val_t < 0.05:
                status = "**SIGNIFIKAN** (p < 0.05)"
            else:
                status = "**TIDAK SIGNIFIKAN** (p ≥ 0.05)"

            if abs(cohens_d) < 0.2:
                efek = "Sangat Kecil"
            elif abs(cohens_d) < 0.5:
                efek = "Kecil"
            elif abs(cohens_d) < 0.8:
                efek = "Sedang"
            else:
                efek = "Besar"

            st.markdown(f"""
            **Keterangan:** Perbedaan kedua kelompok bersifat {status} dengan Effect Size
            Cohen's d = **{abs(cohens_d):.3f}** ({efek}). Kelompok durasi aman memiliki
            resiliensi energi yang jauh lebih stabil dibandingkan kelompok berisiko.
            """)
            st.success("✅ **Solusi Strategis:** Disiplin menjaga durasi harian di zona 'Terkontrol' "
                       "adalah strategi jangka panjang terbaik untuk mencegah **Academic Burnout**.")
        else:
            st.warning("⚠️ Jumlah sampel kurang memadai untuk uji t-test.")

    except FileNotFoundError:
        st.error("⚠️ File `main_data_social_battery.csv` tidak ditemukan.")
    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan: {e}")

# =============================================================================
# MODUL 2: MOOD JAR
# =============================================================================
elif main_menu == "🫙 Modul 2: Mood Jar":

    @st.cache_data
    def load_mj_data():
        mood_jar = pd.read_csv("main_data_mood_jar.csv")
        mood_jar.columns = [c.strip() for c in mood_jar.columns]
        col_map = {c.upper(): c for c in mood_jar.columns}
        if 'MOOD' not in mood_jar.columns and 'MOOD' in col_map:
            mood_jar = mood_jar.rename(columns={col_map['MOOD']: 'MOOD'})
        if 'MOOD_NOTE' not in mood_jar.columns and 'MOOD_NOTE' in col_map:
            mood_jar = mood_jar.rename(columns={col_map['MOOD_NOTE']: 'MOOD_NOTE'})
        mood_jar['MOOD'] = mood_jar['MOOD'].astype(str).str.strip().str.lower()
        mood_jar['MOOD_NOTE'] = mood_jar['MOOD_NOTE'].astype(str).str.strip()
        return mood_jar

    try:
        mood_jar = load_mj_data()

        st.title("🫙 Modul Analisis: Calm Mood Jar")
        st.markdown("### Eksplorasi Kata Kunci Pemicu Emosi Harian Mahasiswa")
        st.divider()

        # ── EKSPLORASI AWAL & WORD CLOUD ─────────────────────────────────
        st.subheader("📋 Eksplorasi Awal & Word Cloud Grid 2×2")

        mood_counts = mood_jar['MOOD'].value_counts()

        col_mj1, col_mj2 = st.columns(2)
        with col_mj1:
            st.write("**Frekuensi Kumulatif Kategori Mood:**")
            st.dataframe(mood_counts.rename("Jumlah"))
        with col_mj2:
            st.write("**Proporsi Persentase Kategori Mood:**")
            for mood_name, val in mood_counts.items():
                pct = val / len(mood_jar) * 100
                st.write(f"- Kategori **{mood_name.capitalize()}** : {pct:.2f}%")

        colors_mood = {'sedih': '#5D9B9B', 'bahagia': '#F4A261', 'cemas': '#E9C46A', 'marah': '#E76F51'}
        bar_colors_mj = [colors_mood.get(m, '#999999') for m in mood_counts.index]

        fig_dist, axes_dist = plt.subplots(1, 2, figsize=(14, 5))
        axes_dist[0].bar(mood_counts.index, mood_counts.values, color=bar_colors_mj, edgecolor='black')
        axes_dist[0].set_title('Distribusi Frekuensi Mood', fontsize=14, fontweight='bold')
        axes_dist[0].set_xlabel('Kategori Mood')
        axes_dist[0].set_ylabel('Jumlah Catatan')
        axes_dist[0].get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
        for i, v in enumerate(mood_counts.values):
            axes_dist[0].text(i, v + 25, str(int(v)), ha='center', fontweight='bold')
        axes_dist[1].pie(mood_counts.values,
                         labels=[k.capitalize() for k in mood_counts.index],
                         autopct='%1.1f%%', colors=bar_colors_mj,
                         startangle=90, explode=[0.05] * len(mood_counts))
        axes_dist[1].set_title('Proporsi Kondisi Mental Mahasiswa', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_dist)
        plt.close(fig_dist)

        st.write("**Visualisasi Word Cloud Dominan (Grid 2×2):**")
        extended_stopwords = {
            'aku', 'saya', 'merasa', 'perasaan', 'dan', 'yang', 'di', 'ke', 'dari', 'ini', 'itu',
            'untuk', 'dengan', 'karena', 'tidak', 'tak', 'bisa', 'akan', 'ada', 'adalah', 'sangat',
            'lebih', 'bahwa', 'saat', 'jadi', 'agak', 'kalau', 'jika', 'saja', 'juga', 'lah', 'kah',
            'nya', 'tapi', 'tetapi', 'bgt', 'banget', 'kok', 'loh', 'sih', 'ya', 'pas', 'malah',
            'begitu', 'secara', 'tentang', 'oleh', 'bagi', 'pada', 'atau', 'lalu', 'kemudian',
            'setelah', 'sebelum', 'ketika', 'sementara', 'bahkan', 'namun', 'melainkan', 'sedangkan',
            'seperti', 'sebagai', 'terhadap', 'kepada', 'maupun', 'ia', 'kami', 'kita', 'kamu',
            'mereka', 'dia', 'anda', 'hal', 'orang', 'buat', 'banyak', 'bukan', 'pun', 'hanya',
            'mampu', 'dapat', 'ingin', 'mau', 'bila', 'terus', 'lagi', 'tahu', 'melihat', 'bilang',
            'kata', 'katanya', 'dalam', 'apa', 'sudah', 'telah', 'semua', 'masih', 'merasakan',
            'menjadi', 'membuat', 'sama', 'lain', 'sedang', 'hari', 'sedikit', 'diri', 'mulai',
            'yg', 'mungkin', 'benar', 'benar-benar', 'sebuah', 'suatu', 'sesuatu', 'satu', 'dua',
            'tiga', 'kali', 'waktu', 'pernah', 'kembali', 'paling', 'terlalu', 'cukup', 'jangan',
            'hampir', 'sekarang', 'serta', 'tersebut', 'bagaikan', 'selalu', 'biasanya', 'punya',
            'mengapa', 'kenapa', 'beberapa', 'sekali', 'setiap', 'tanpa', 'baru'
        }
        color_maps_wc = {'sedih': 'YlGnBu', 'bahagia': 'Oranges', 'cemas': 'YlOrBr', 'marah': 'Reds'}
        moods_unik = [m for m in ['sedih', 'bahagia', 'cemas', 'marah'] if m in mood_jar['MOOD'].unique()]

        if len(moods_unik) >= 1:
            n_rows = (len(moods_unik) + 1) // 2
            fig_wc, axes_wc = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))
            axes_wc = np.array(axes_wc).flatten()
            for i, mood_name in enumerate(moods_unik):
                text_content = " ".join(
                    mood_jar[mood_jar['MOOD'] == mood_name]['MOOD_NOTE'].astype(str))
                if text_content.strip():
                    wc = WordCloud(width=800, height=600, background_color='white',
                                   stopwords=extended_stopwords, min_font_size=10,
                                   colormap=color_maps_wc.get(mood_name, 'viridis'),
                                   collocations=False).generate(text_content)
                    axes_wc[i].imshow(wc, interpolation='bilinear')
                else:
                    axes_wc[i].text(0.5, 0.5, 'Data kosong', ha='center', va='center')
                axes_wc[i].set_title(f"Word Cloud — Mood: {mood_name.upper()}",
                                     fontsize=14, fontweight='bold')
                axes_wc[i].axis('off')
            for j in range(len(moods_unik), len(axes_wc)):
                axes_wc[j].axis('off')
            plt.tight_layout()
            st.pyplot(fig_wc)
            plt.close(fig_wc)

        st.markdown("""
        **Keterangan:**
        * Dataset berhasil dimuat dengan total **11.887** entri, seimbang (~2.972 per kategori mood).
        * Word Cloud menunjukkan kontras jelas: emosi negatif (`sedih`, `cemas`, `marah`) didominasi
          keluhan akademik dan kelelahan fisik; emosi positif (`bahagia`) didominasi kata kepuasan
          dan hubungan sosial.
        """)
        st.divider()

        # ── PERTANYAAN 1 ────────────────────────────────────────────────────
        st.subheader("📌 [1] Eksplorasi Pemicu Utama antara Rasa Sedih dan Cemas")
        st.markdown('**Soal:** *"Apa tiga kata kunci yang paling sering muncul dalam cerita pengguna saat merasa \'sedih\' dan \'cemas\', serta bagaimana perbandingannya?"*')
        st.markdown('**Indikator:** Menemukan kata kunci penanda utama pada cerita pengguna di kelompok mood sedih dan cemas.')

        df_sedih = mood_jar[mood_jar['MOOD'] == 'sedih']
        df_cemas = mood_jar[mood_jar['MOOD'] == 'cemas']

        stopwords_p1 = {
            'aku', 'saya', 'kau', 'kamu', 'dia', 'mereka', 'kita', 'kami', 'ini', 'itu', 'tersebut',
            'dan', 'atau', 'tapi', 'namun', 'karena', 'sebab', 'jika', 'kalau', 'maka', 'sangat',
            'begitu', 'seperti', 'sebagai', 'dengan', 'tanpa', 'untuk', 'dari', 'kepada', 'pada',
            'oleh', 'dalam', 'bahwa', 'bisa', 'akan', 'telah', 'sudah', 'belum', 'masih', 'sedang',
            'yg', 'dgn', 'tdk', 'utk', 'jd', 'jg', 'sdh', 'pun', 'nya', 'merasa', 'yang', 'tidak',
            'perasaan', 'orang', 'saat', 'hal', 'ketika', 'hari', 'ada', 'banyak', 'tak', 'agak',
            'rasa', 'membuat', 'terlalu', 'menjadi', 'selalu', 'juga', 'diri', 'lagi', 'benar',
            'memang', 'enak', 'anda', 'sedikit', 'hanya', 'lebih', 'tahu', 'semua', 'merasakan',
            'mungkin', 'sama', 'mulai', 'beberapa', 'sedih'
        }

        def get_top_keywords_mj(df_mood, top_n=5):
            all_text = ' '.join(df_mood['MOOD_NOTE'].astype(str))
            all_text_clean = re.sub(r'[^\w\s]', ' ', all_text)
            words = all_text_clean.split()
            filtered = [w.strip().lower() for w in words
                        if len(w.strip()) >= 3
                        and w.strip().lower() not in stopwords_p1
                        and not w.strip().isdigit()
                        and w.strip().lower() not in {'haha', 'hehe', 'lol', 'hmm'}]
            return dict(Counter(filtered).most_common(top_n))

        if not df_sedih.empty and not df_cemas.empty:
            top5_sedih = get_top_keywords_mj(df_sedih)
            top5_cemas = get_top_keywords_mj(df_cemas)

            fig_p1mj, axes_p1mj = plt.subplots(1, 2, figsize=(14, 5))
            axes_p1mj[0].bar(top5_sedih.keys(), top5_sedih.values(),
                             color='#5D9B9B', edgecolor='black')
            axes_p1mj[0].set_title('Top 5 Kata Kunci — Mood SEDIH', fontsize=13, fontweight='bold')
            axes_p1mj[0].set_xlabel('Kata Kunci')
            axes_p1mj[0].set_ylabel('Frekuensi')
            axes_p1mj[0].get_yaxis().set_major_formatter(
                plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
            if top5_sedih:
                max_s = max(top5_sedih.values())
                for i, v in enumerate(top5_sedih.values()):
                    axes_p1mj[0].text(i, v + max_s * 0.02, str(int(v)),
                                      ha='center', fontweight='bold')

            axes_p1mj[1].bar(top5_cemas.keys(), top5_cemas.values(),
                             color='#E9C46A', edgecolor='black')
            axes_p1mj[1].set_title('Top 5 Kata Kunci — Mood CEMAS', fontsize=13, fontweight='bold')
            axes_p1mj[1].set_xlabel('Kata Kunci')
            axes_p1mj[1].set_ylabel('Frekuensi')
            axes_p1mj[1].get_yaxis().set_major_formatter(
                plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
            if top5_cemas:
                max_c = max(top5_cemas.values())
                for i, v in enumerate(top5_cemas.values()):
                    axes_p1mj[1].text(i, v + max_c * 0.02, str(int(v)),
                                      ha='center', fontweight='bold')

            plt.tight_layout()
            st.pyplot(fig_p1mj)
            plt.close(fig_p1mj)

            top_sedih_list = list(top5_sedih.keys())
            top_cemas_list = list(top5_cemas.keys())
            st.markdown(f"""
            **Keterangan:**
            * **Mood Sedih:** Didominasi kata seperti **'{top_sedih_list[0] if top_sedih_list else '-'}'**,
              **'{top_sedih_list[1] if len(top_sedih_list) > 1 else '-'}'** → pemicu berbasis fisik/kelelahan.
            * **Mood Cemas:** Didominasi kata seperti **'{top_cemas_list[0] if top_cemas_list else '-'}'**,
              **'{top_cemas_list[1] if len(top_cemas_list) > 1 else '-'}'** → pemicu berbasis ketakutan
              psikologis & ketidakpastian.
            """)
            st.success("✅ **Solusi Strategis:** Kelompok `sedih`, `support_message` akan memberikan pesan *physical recovery* ( butuh istirahat untuk pulih) & *comforting* (dukungan). "
                       "Kelompok `cemas`, akan diberikan pesan untuk melakukan teknik *grounding* (pengalihan) dan validasi kognitif.")
        else:
            st.warning("⚠️ Data mood sedih/cemas tidak ditemukan dalam dataset.")
        st.divider()

        # ── PERTANYAAN 2 ────────────────────────────────────────────────────
        st.subheader("📌 [2] Pengaruh Hubungan Sosial pada Mood Bahagia vs Sedih")
        st.markdown('**Soal:** *"Berapa besar persentase catatan mood \'bahagia\' yang menceritakan hubungan sosial, dibandingkan dengan mood \'sedih\'?"*')
        st.markdown('**Indikator:** Membandingkan seberapa sering interaksi sosial muncul dalam cerita bahagia versus sedih.')

        df_bahagia = mood_jar[mood_jar['MOOD'] == 'bahagia']
        df_sedih_p2 = mood_jar[mood_jar['MOOD'] == 'sedih']

        social_keywords = [
            'teman', 'sahabat', 'kawan', 'keluarga', 'orang tua', 'ibu', 'mama', 'ayah', 'papa',
            'bapak', 'pacar', 'suami', 'istri', 'kekasih', 'pasangan', 'bersama', 'kita', 'mereka',
            'kami', 'kumpul', 'ngumpul', 'bertemu', 'saudara', 'kakak', 'adik', 'anak'
        ]

        def count_social_pct(df_mood):
            if len(df_mood) == 0:
                return 0, 0
            count = sum(any(kw in str(row).lower() for kw in social_keywords)
                        for row in df_mood['MOOD_NOTE'])
            return count, (count / len(df_mood)) * 100

        if not df_bahagia.empty and not df_sedih_p2.empty:
            cnt_b, pct_b = count_social_pct(df_bahagia)
            cnt_s, pct_s = count_social_pct(df_sedih_p2)

            col_p2mja, col_p2mjb = st.columns(2)
            col_p2mja.metric("Interaksi Sosial — Bahagia", f"{pct_b:.1f}% ({int(cnt_b)} catatan)")
            col_p2mjb.metric("Interaksi Sosial — Sedih", f"{pct_s:.1f}% ({int(cnt_s)} catatan)")

            fig_p2mj, ax_p2mj = plt.subplots(figsize=(10, 6))
            bars_p2 = ax_p2mj.bar(['Bahagia', 'Sedih'], [pct_b, pct_s],
                                   color=['#F4A261', '#5D9B9B'], edgecolor='black', width=0.4)
            ax_p2mj.set_ylim(0, max(pct_b, pct_s) + 15)
            ax_p2mj.set_title('Persentase Catatan dengan Interaksi Sosial',
                               fontsize=14, fontweight='bold')
            ax_p2mj.set_ylabel('Persentase (%)')
            ax_p2mj.get_yaxis().set_major_formatter(
                plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
            for bar, pct, cnt in zip(bars_p2, [pct_b, pct_s], [cnt_b, cnt_s]):
                ax_p2mj.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                              f'{pct:.1f}%\n({int(cnt)} catatan)',
                              ha='center', va='bottom', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_p2mj)
            plt.close(fig_p2mj)

            st.markdown(f"""
            **Keterangan:**
            * **Mood Bahagia ({pct_b:.1f}%):** Kehadiran lingkaran sosial (keluarga, teman, pasangan)
              merupakan faktor pelatuk eksternal dominan dalam memicu kebahagiaan mahasiswa.
            * **Mood Sedih ({pct_s:.1f}%):** Distres emosional lebih banyak dipicu masalah internal
              atau *lack of social support*.
            """)
            st.success("✅ **Solusi Strategis:** Saat sistem mendeteksi mood `sedih`, `support_message` "
                       "menyarankan pengguna membuka percakapan dengan sahabat atau meluangkan waktu bersama keluarga.")
        else:
            st.warning("⚠️ Data mood bahagia/sedih tidak ditemukan dalam dataset.")
        st.divider()

        # ── PERTANYAAN 3 ────────────────────────────────────────────────────
        st.subheader("📌 [3] Kaitan Kondisi Fisik dengan Emosi (Marah & Cemas vs Bahagia)")
        st.markdown('**Soal:** *"Seberapa sering pengguna yang sedang \'marah\' dan \'cemas\' mengeluhkan kondisi fisik, dibandingkan saat \'bahagia\'?"*')
        st.markdown('**Indikator:** Mengukur seberapa sering keluhan fisik muncul bersamaan dengan emosi tertentu.')

        df_marah_cemas = mood_jar[mood_jar['MOOD'].isin(['marah', 'cemas'])]
        df_bahagia_p3 = mood_jar[mood_jar['MOOD'] == 'bahagia']

        physical_keywords = [
            'sakit', 'nyeri', 'pusing', 'mual', 'lelah', 'capek', 'lemas', 'tidur',
            'kurang tidur', 'insomnia', 'begadang', 'flu', 'demam', 'batuk',
            'sakit kepala', 'haid', 'linu'
        ]

        def count_physical_pct(df_mood):
            if len(df_mood) == 0:
                return 0, 0
            count = sum(any(kw in str(row).lower() for kw in physical_keywords)
                        for row in df_mood['MOOD_NOTE'])
            return count, (count / len(df_mood)) * 100

        if not df_marah_cemas.empty and not df_bahagia_p3.empty:
            cnt_neg, pct_neg = count_physical_pct(df_marah_cemas)
            cnt_pos, pct_pos = count_physical_pct(df_bahagia_p3)

            col_p3mja, col_p3mjb = st.columns(2)
            col_p3mja.metric("Keluhan Fisik — Marah+Cemas", f"{pct_neg:.1f}% ({int(cnt_neg)} catatan)")
            col_p3mjb.metric("Keluhan Fisik — Bahagia", f"{pct_pos:.1f}% ({int(cnt_pos)} catatan)")

            fig_p3mj, ax_p3mj = plt.subplots(figsize=(10, 6))
            bars_p3 = ax_p3mj.bar(['Marah + Cemas', 'Bahagia'], [pct_neg, pct_pos],
                                   color=['#E76F51', '#F4A261'], edgecolor='black', width=0.4)
            ax_p3mj.set_ylim(0, max(pct_neg, pct_pos) + 15)
            ax_p3mj.set_title('Persentase Catatan yang Menyebut Kondisi Fisik',
                               fontsize=14, fontweight='bold')
            ax_p3mj.set_ylabel('Persentase (%)')
            ax_p3mj.get_yaxis().set_major_formatter(
                plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
            for bar, pct, cnt in zip(bars_p3, [pct_neg, pct_pos], [cnt_neg, cnt_pos]):
                ax_p3mj.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                              f'{pct:.1f}%\n({int(cnt)} catatan)',
                              ha='center', va='bottom', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_p3mj)
            plt.close(fig_p3mj)

            st.markdown(f"""
            **Keterangan:**
            * **Klaster Negatif — Marah + Cemas ({pct_neg:.1f}%):** Pengguna sering menyisipkan keluhan
              fisik seperti gangguan tidur (*begadang*, *insomnia*) dan rasa tidak nyaman (*pusing*, *lelah*).
              Penurunan kesehatan biologis berdampak langsung pada penurunan ambang kesabaran kognitif.
            * **Klaster Positif — Bahagia ({pct_pos:.1f}%):** Keluhan fisik sangat minim. Kondisi tubuh
              prima dan tidur cukup adalah *well-being foundation* bagi munculnya afeksi positif.
            """)
            st.success("✅ **Solusi Strategis:** Saat sistem mendeteksi keluhan fisik + mood `marah`/`cemas`, "
                       "`support_message` memberikan rekomendasi: pengingat tidur, latihan pernapasan, atau *bed rest*.")
        else:
            st.warning("⚠️ Data mood marah/cemas/bahagia tidak ditemukan dalam dataset.")

    except FileNotFoundError:
        st.error("⚠️ File `main_data_mood_jar.csv` tidak ditemukan.")
    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan: {e}")