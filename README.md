# 🌙 Capstone Project: Calm Social Battery

## Deskripsi
Proyek ini bertujuan untuk menganalisis pola kelelahan sosial (*social fatigue*) pengguna berdasarkan data aktivitas harian tahun 2026. Dashboard ini memvisualisasikan tren kelelahan bulanan, menentukan ambang batas aman durasi interaksi sosial, serta mengidentifikasi pola penurunan dan pemulihan energi mingguan (seperti fenomena *Mid-Week Crash*). Analisis ini disajikan melalui dashboard interaktif menggunakan Streamlit.

## Link Dashboard Online
Aplikasi ini telah dideploy dan dapat diakses secara publik melalui:
👉 [https://calm-social-battery-nurhifdhiyatulmaulidiyah.streamlit.app/](https://calm-social-battery-nurhifdhiyatulmaulidiyah.streamlit.app/)

## Fitur Utama Dashboard
1. **Tren Social Fatigue Bulanan:** Menampilkan persentase kelelahan tertinggi dan terendah sepanjang tahun.
2. **Batas Aman Durasi:** Visualisasi distribusi durasi aktivitas untuk menentukan batas aman (950 menit).
3. **Weekly Energy Pattern:** Grafik fluktuasi baterai energi dari Senin hingga Minggu.
4. **Korelasi Intensitas:** Hubungan antara tingkat keramaian/intensitas sosial dengan sisa baterai.
5. **Analisis Komparatif:** Perbandingan signifikan antara kelompok durasi sehat vs kelompok berisiko lelah.

## Setup Environment - Local (Conda/Python)
Jika ingin menjalankan proyek ini di komputer lokal, ikuti langkah berikut:

### 1. Buat Environment
```bash
conda create --name social-battery python=3.9
conda activate social-battery