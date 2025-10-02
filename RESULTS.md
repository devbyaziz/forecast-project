# Hasil Analisis Forecasting Traffic CCTV

Laporan hasil prediksi traffic IN/OUT menggunakan model SARIMAX berdasarkan data 7 hari (6-27 September 2025).

---

## Performa Model

### **1. Model Total Traffic (Univariate)**

Model ini langsung memprediksi total traffic tanpa memisahkan IN dan OUT.

| Metric | Value | Artinya |
|--------|-------|---------|
| **MAE** | 64.30 | Prediksi meleset rata-rata 64 orang |
| **RMSE** | 85.71 | Ukuran error keseluruhan |
| **R² Score** | 0.5867 | Model bisa menjelaskan 59% pola data |

**Grafik tersedia di:**
- `results/10_forecast_vs_actual.png` - Perbandingan prediksi vs data asli
- `results/10_test_forecast_zoomed.png` - Detail test set
- `results/11_future_forecast_7days.png` - Prediksi 7 hari ke depan

---

### **2. Model IN + OUT (Multivariate)**

Model ini memprediksi traffic masuk dan keluar secara terpisah.

| Metric | IN Traffic | OUT Traffic | TOTAL Traffic |
|--------|-----------|-------------|---------------|
| **MAE** | 51.03 | 26.27 | 66.54 |
| **RMSE** | 72.31 | 35.45 | 88.04 |
| **R² Score** | 0.2021 | 0.5993 | 0.5639 |

**Temuan:**
- Model OUT jauh lebih akurat (R²=0.60) dibanding IN (R²=0.20)
- Traffic keluar punya pola yang konsisten dan mudah diprediksi
- Traffic masuk lebih acak dan susah ditebak
- Kemungkinan besar karena traffic keluar tergantung kapasitas ruangan, sementara traffic masuk dipengaruhi banyak faktor eksternal

**Grafik tersedia di:**
- `results/multivariate_test_forecast.png` - Perbandingan 3 grafik (IN/OUT/TOTAL)
- `results/multivariate_test_comparison.png` - Tampilan side-by-side
- `results/multivariate_future_forecast.png` - Prediksi 7 hari mendatang

---

## Pola Traffic yang Ditemukan

### **1. Jam-Jam Tersibuk**

Dari prediksi 7 hari ke depan, ini 10 jam dengan traffic tertinggi:

| Urutan | Jam | Rata-rata Traffic | Level |
|--------|-----|------------------|-------|
| 1 | 16:00 | 138 orang | Puncak |
| 2 | 17:00 | 135 orang | Puncak |
| 3 | 15:00 | 130 orang | Puncak |
| 4 | 18:00 | 122 orang | Tinggi |
| 5 | 14:00 | 112 orang | Tinggi |
| 6 | 19:00 | 100 orang | Sedang |
| 7 | 13:00 | 86 orang | Sedang |
| 8 | 20:00 | 69 orang | Sedang |
| 9 | 12:00 | 53 orang | Rendah |
| 10 | 21:00 | 31 orang | Rendah |

**Kesimpulan singkat:**
- Paling ramai: **Sore jam 3-6 (15:00-18:00)**
- Cukup ramai: **Jam 2 siang & 7 malam (14:00 & 19:00)**
- Sepi: **Pagi dan malam (sebelum jam 10 & setelah jam 10 malam)**

---

### **2. Sabtu/Minggu vs Hari Kerja**

| Tipe Hari | Rata-rata | Tertinggi | Terendah | Catatan |
|-----------|-----------|-----------|----------|---------|
| **Weekend** | 86 orang | 247 orang | -75 | Ramai, bisa diprediksi |
| **Weekday** | -45 orang | 162 orang | -224 | PERHATIAN: Angka tidak masuk akal |

**Perhatian penting:**
- Prediksi untuk hari kerja menghasilkan angka negatif (tidak mungkin dalam realita)
- Penyebabnya: Data training cuma 7 hari dan kebanyakan weekend
- Solusinya: Perlu kumpulkan data minimal 1-2 bulan buat dapat pola hari kerja yang benar

**Yang bisa dipercaya:**
- Data weekend (Sabtu-Minggu): Bisa dipercaya
- Data weekday (Senin-Jumat): Jangan dipercaya dulu

---

### **3. Pola dalam Sehari**

| Waktu | Level Traffic | Keterangan |
|-------|--------------|------------|
| **00:00-09:00** | Sangat sepi (0-20 orang) | Belum/sudah tutup |
| **10:00-12:00** | Mulai ramai (20-80 orang) | Baru buka, traffic naik perlahan |
| **13:00-14:00** | Ramai (80-150 orang) | Menjelang peak |
| **15:00-18:00** | **Paling ramai (130-250 orang)** | Jam puncak |
| **19:00-21:00** | Menurun (50-100 orang) | Mulai berkurang |
| **22:00-23:00** | Sepi lagi (0-40 orang) | Menjelang tutup |

---

## Analisis Akurasi Model (Test Set)

### **Periode Test:** 21 & 27 September 2025 (2 hari)

#### **5 Prediksi dengan Error Terbesar:**

| Tanggal & Jam | Data Asli | Prediksi | Selisih | % Error |
|---------------|-----------|----------|---------|---------|
| 27 Sept 19:00 | 451 orang | 225 orang | Kurang 226 | 50% |
| 27 Sept 18:00 | 485 orang | 269 orang | Kurang 216 | 45% |
| 21 Sept 13:00 | 41 orang | 254 orang | Lebih 213 | 520% |
| 21 Sept 12:00 | 0 orang | 183 orang | Lebih 183 | - |
| 21 Sept 18:00 | 238 orang | 394 orang | Lebih 156 | 66% |

**Apa yang terjadi:**

1. **Tanggal 27 Sept jam 6-7 sore** - Traffic melonjak drastis (485 orang)
   - Model cuma prediksi 270-an orang (terlalu rendah)
   - Kayaknya ada event/promo khusus yang model ga tau

2. **Tanggal 21 Sept jam 12-1 siang** - Ternyata sepi banget (cuma 0-41 orang)
   - Model malah prediksi ramai 183-254 orang (terlalu tinggi)
   - Kemungkinan tutup sementara atau lagi maintenance

**Kenapa bisa begitu:**
- Model cuma dikasih data traffic, ga tau ada event apa atau kapan tutup
- Data cuma 7 hari, belum cukup buat belajar pola outlier

---

## Prediksi 7 Hari ke Depan

### **Periode: 28 September - 4 Oktober 2025**

#### **10 Jam Paling Ramai (Prediksi):**

| Tanggal & Jam | Hari | Masuk | Keluar | Total |
|---------------|------|-------|--------|-------|
| 4 Okt 16:00 | Sabtu | 143 | 105 | 247 |
| 4 Okt 17:00 | Sabtu | 139 | 106 | 245 |
| 4 Okt 15:00 | Sabtu | 137 | 103 | 239 |
| 4 Okt 18:00 | Sabtu | 129 | 103 | 232 |
| 4 Okt 14:00 | Sabtu | 123 | 99 | 222 |
| 28 Sept 16:00 | Minggu | 126 | 94 | 220 |
| 28 Sept 17:00 | Minggu | 123 | 95 | 217 |
| 28 Sept 15:00 | Minggu | 120 | 92 | 212 |
| 4 Okt 19:00 | Sabtu | 113 | 96 | 209 |
| 28 Sept 18:00 | Minggu | 113 | 94 | 206 |

**Kesimpulan:**
- Hari paling ramai: **Sabtu 4 Oktober**
- Jam paling padat: **Sabtu jam 4 sore** (247 orang)
- Semua 10 besar terjadi di **weekend** (Sabtu/Minggu)

---

## Rekomendasi Operasional

### **1. Jadwal Staff (Weekend)**

| Jam | Prediksi Traffic | Kebutuhan Staff | Jumlah Staff |
|-----|-----------------|-----------------|--------------|
| 00:00-09:00 | 0-20 orang | Minimal | 1-2 orang |
| 10:00-12:00 | 50-100 orang | Cukup | 2-3 orang |
| 13:00-14:00 | 100-150 orang | Lumayan banyak | 3-4 orang |
| **15:00-18:00** | **150-250 orang** | **Paling banyak** | **5-6 orang** |
| 19:00-21:00 | 50-100 orang | Lumayan banyak | 3-4 orang |
| 22:00-23:00 | 0-50 orang | Cukup | 1-2 orang |

**Catatan:** Asumsi 1 staff bisa handle sekitar 50 orang

---

### **2. Perencanaan Kapasitas**

| Hal | Angka | Yang Perlu Dilakukan |
|-----|-------|---------------------|
| **Kapasitas Maksimal** | 250 orang | Pastikan ruang bisa nampung 250+ orang |
| **Rata-rata Weekend** | 86 orang | Kapasitas normal sudah cukup |
| **Jam Puncak** | 15:00-18:00 | Siapkan sumber daya tambahan |

---

### **3. Catatan Penting tentang Traffic Masuk vs Keluar**

**Traffic Masuk (IN):**
- Susah ditebak (akurasi cuma 20%)
- Bisa naik tiba-tiba kalau ada event
- Perlu dipantau real-time

**Traffic Keluar (OUT):**
- Lebih gampang diprediksi (akurasi 60%)
- Polanya konsisten
- Cukup andalkan prediksi model

**Saran:**
- Fokus pantau traffic masuk karena lebih ga terduga
- Pasang alert kalau traffic masuk tiba-tiba lewat 300 (bisa jadi ada event)
- Pakai forecast OUT sebagai baseline kapasitas keluar

---

## Keterbatasan Model

### **1. Keterbatasan Data**
- Data cuma 7 hari (168 jam) - terlalu sedikit
- Pola hari kerja belum ketangkep karena datanya kebanyakan weekend
- Model ga tau ada event khusus (promo, libur nasional, maintenance)

### **2. Masalah Prediksi**
- Prediksi hari kerja keluar angka negatif (ga mungkin dalam kenyataan)
- Kalau traffic tiba-tiba naik ekstrem (>400), model prediksinya terlalu rendah
- Kalau ternyata lagi tutup/maintenance, model tetep prediksi ramai

### **3. Seberapa Akurat Prediksinya?**
- **Paling akurat:** Weekend jam 15:00-18:00 (bisa dipercaya)
- **Lumayan akurat:** Weekend jam 10:00-14:00 & 19:00-21:00 (cukup dipercaya)
- **Ga akurat:** Hari kerja semua jam (jangan dipercaya dulu)

---

## Yang Perlu Dilakukan Selanjutnya

### **Yang Bisa Langsung Dipakai:**
1. Pakai prediksi weekend buat atur jadwal staff
2. Abaikan dulu prediksi hari kerja (belum bisa dipercaya)
3. Pantau terus traffic masuk secara real-time (lebih susah ditebak)

### **Pengumpulan Data:**
1. Kumpulkan data minimal 1-2 bulan lagi biar polanya lebih jelas
2. Catat kalau ada event khusus (promo, hari libur, tutup sementara)
3. Kalau bisa, tambahin data cuaca sama hari libur nasional

### **Perbaikan Model:**
1. Tambahin aturan: prediksi ga boleh negatif (minimal 0)
2. Coba gabungin beberapa model (SARIMAX + XGBoost)
3. Update model setiap bulan dengan data baru

---

## Lokasi File Hasil

### **Grafik/Visualisasi:**
- `results/01_data_exploration.png` - Analisis data awal
- `results/01_traffic_heatmap.png` - Heatmap traffic
- `results/04_acf_pacf.png` - Analisis ACF/PACF
- `results/multivariate_test_forecast.png` - Hasil test prediksi
- `results/multivariate_future_forecast.png` - Prediksi 7 hari

### **File Data:**
- `results/multivariate_future_forecast_7days.csv` - **File prediksi utama**
- `results/multivariate_test_predictions.csv` - Data test prediksi
- `results/multivariate_metrics.csv` - Metrik akurasi model

### **Model:**
- `results/sarimax_model.pkl` - Model yang sudah di-training (bisa dipake ulang)

---

## Pertanyaan?

Kalau ada yang mau ditanyain:
1. Cek dulu visualisasi di folder `results/`
2. Buka file CSV kalau butuh angka detailnya
3. Jalanin ulang model kalau ada data baru

---

**Laporan dibuat:** Oktober 2025
**Versi Model:** SARIMAX (1,0,1)(1,0,1,24)
**Periode Data:** 6-27 September 2025
