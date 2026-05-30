# 🌙 Capstone Project Calm: Social Battery & Mood Jar

## Deskripsi
Proyek ini bertujuan untuk menganalisis pola kelelahan sosial serta karakteristik pemicu emosi harian pengguna berdasarkan data aktivitas tahun 2026. Dashboard ini memvisualisasikan tren kelelahan bulanan, menentukan ambang batas aman durasi interaksi sosial, mengidentifikasi pola penurunan energi mingguan (seperti fenomena *Mid-Week Crash*), sekaligus mengeksplorasi kata kunci tekstual pemicu suasana hati pengguna, keterlibatan hubungan interpersonal, dan manifestasi keluhan somatis. Analisis interaktif ini disajikan secara komprehensif menggunakan Streamlit.

## Link Dashboard Online
Aplikasi ini telah dideploy dan dapat diakses secara publik melalui:
👉 [https://calm-social-battery-nurhifdhiyatulmaulidiyah.streamlit.app/](https://calm-social-battery-nurhifdhiyatulmaulidiyah.streamlit.app/)

## Fitur Utama Dashboard

### 🔋 Modul 1: Social Battery
1. **Tren Social Fatigue Bulanan:** Menampilkan persentase kelelahan tertinggi dan terendah sepanjang tahun berdasarkan batas kondisi kritis (`battery_score < 20`).
2. **Batas Aman Durasi:** Visualisasi distribusi durasi aktivitas menggunakan boxplot untuk menentukan ambang batas aman menit interaksi sosial harian.
3. **Weekly Energy Pattern:** Grafik fluktuasi rata-rata baterai energi dari Senin hingga Minggu untuk mendeteksi fenomena titik jenuh tengah minggu (*Mid-Week Crash*).
4. **Korelasi Intensitas:** Model regresi linear untuk melihat hubungan antara skor intensitas kepadatan lingkungan sosial terhadap laju penipisan energi cadangan.
5. **Analisis Komparatif:** Pengujian t-test untuk melihat perbedaan rata-rata tingkat energi cadangan secara signifikan antara kelompok durasi terkontrol dengan kelompok berisiko.

### 🫙 Modul 2: Mood Jar
1. **Eksplorasi Frekuensi & Word Cloud:** Menampilkan distribusi frekuensi kumulatif kategori suasana hati harian beserta visualisasi peta kata kunci murni melalui Word Cloud Grid 2x2.
2. **Eksplorasi Pemicu Utama (Sedih vs Cemas):** Ekstraksi 5 kata kunci dominan pada cerita pengguna untuk melihat kontras pemicu kognitif ketakutan masa depan dengan duka keletihan fisik.
3. **Pengaruh Hubungan Sosial:** Perbandingan persentase rasio keterlibatan entitas interpersonal orang terdekat pada catatan suasana hati bernilai polar positif (`bahagia`) vs negatif (`sedih`).
4. **Kaitan Kondisi Fisik vs Emosi:** Pengukuran proporsi seberapa sering pengguna mengeluhkan gangguan somatosensorik tubuh biologis (seperti kurang tidur, pusing, lelah) pada kelompok emosi negatif vs emosi bahagia.

## Setup Environment - Local (Conda/Python)
Jika ingin menjalankan proyek ini di komputer lokal, ikuti langkah berikut:

### 1. Buat Environment
```bash
conda create --name calm-dashboard python=3.9
conda activate calm-dashboard