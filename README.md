# CCTV Traffic Forecasting dengan SARIMAX

Sistem forecasting untuk prediksi traffic CCTV menggunakan model SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables).

---

## Dataset

- **Sumber:** Data CCTV traffic (jumlah orang masuk/keluar per jam)
- **Periode:** 7 hari (6-27 September 2025)
- **Frekuensi:** Per jam (168 data points)
- **Fitur:**
  - `datetime`: Timestamp
  - `in`: Jumlah orang masuk
  - `out`: Jumlah orang keluar
  - `total_traffic`: Total traffic (in + out)
  - `hour`: Jam (0-23)
  - `day_of_week`: Hari dalam minggu (0=Senin, 6=Minggu)
  - `is_weekend`: Flag weekend (0/1)

---

## Tujuan Project

1. Forecast total traffic untuk 7 hari ke depan
2. Forecast traffic masuk (IN) dan keluar (OUT) secara terpisah
3. Identifikasi jam-jam tersibuk (peak hours)
4. Memberikan rekomendasi untuk staffing dan capacity planning

---

## Metodologi

### **Model: SARIMAX**

**Parameter:**
- **order=(1,0,1)**: Non-seasonal (p=1, d=0, q=1)
- **seasonal_order=(1,0,1,24)**: Seasonal (P=1, D=0, Q=1, s=24)

**Penjelasan Parameter:**
- `p=1`: AutoRegressive order 1 (pakai data 1 jam sebelumnya)
- `d=0`: Tidak perlu differencing (data sudah stasioner)
- `q=1`: Moving Average order 1 (pakai error 1 lag)
- `P=1`: Seasonal AR order 1
- `D=0`: Tidak perlu seasonal differencing
- `Q=1`: Seasonal MA order 1
- `s=24`: Seasonal period 24 jam (pola harian berulang setiap 24 jam)

**Cara Menentukan Parameter:**
- Analisis ACF/PACF plots (manual, lags=40)
- Test stationarity: ADF & KPSS

### **Exogenous Variables (Variabel Tambahan):**

1. `hour` - Jam (0-23)
2. `is_weekend` - Indikator weekend
3. `hour_sin` - Encoding siklik jam (sine)
4. `hour_cos` - Encoding siklik jam (cosine)
5. `day_sin` - Encoding siklik hari (sine)
6. `day_cos` - Encoding siklik hari (cosine)

**Kenapa pakai Cyclical Encoding?**
- Jam 23:00 sebenarnya dekat dengan jam 00:00, tapi kalau pakai angka biasa jaraknya 23 (jauh)
- Sin/cos encoding bikin model ngerti kalau jam 23 dan jam 0 itu berdekatan

---

## Struktur Project

```
cctv-forecast/
├── dataset/
│   ├── data_bersih.csv                          # Data mentah
│   ├── data_preprocessed_v1_original_fixed.csv  # Preprocessed (jam operasional)
│   └── data_preprocessed_v3_full_24h_fixed.csv  # Preprocessed (full 24h) [DIPAKAI]
├── results/
│   ├── 01_data_exploration.png                  # Plot eksplorasi data
│   ├── 01_traffic_heatmap.png                   # Heatmap jam vs hari
│   ├── 03_stationarity_test.png                 # Test ADF & KPSS
│   ├── 04_acf_pacf.png                          # Plot ACF/PACF
│   ├── 07_residual_analysis.png                 # Diagnostik residual
│   ├── 10_forecast_vs_actual.png                # Univariate: Aktual vs Prediksi
│   ├── 10_test_forecast_zoomed.png              # Univariate: Test set detail
│   ├── 11_future_forecast_7days.png             # Univariate: Forecast 7 hari
│   ├── multivariate_test_forecast.png           # Multivariate: IN/OUT/TOTAL
│   ├── multivariate_test_comparison.png         # Multivariate: Perbandingan
│   ├── multivariate_future_forecast.png         # Multivariate: Forecast 7 hari
│   ├── test_predictions.csv                     # Prediksi test univariate
│   ├── future_forecast_7days.csv                # Forecast univariate
│   ├── multivariate_test_predictions.csv        # Prediksi test multivariate
│   ├── multivariate_future_forecast_7days.csv   # Forecast multivariate [FILE UTAMA]
│   ├── evaluation_metrics.csv                   # Metrik univariate
│   ├── multivariate_metrics.csv                 # Metrik multivariate
│   └── sarimax_model.pkl                        # Model tersimpan
├── sarimax_forecast.py                          # Script univariate (total traffic saja)
├── sarimax_multivariate.py                      # Script multivariate (IN + OUT) [DIPAKAI]
├── requirements.txt                             # Dependencies
├── README.md                                    # Dokumentasi ini
└── RESULTS.md                                   # Hasil analisis lengkap
```

---

## Instalasi & Cara Pakai

### **1. Install Dependencies**

```bash
pip install -r requirements.txt
```

**Library yang dibutuhkan:**
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- scikit-learn
- scipy

### **2. Jalankan Forecasting**

**Model Univariate (Total Traffic saja):**
```bash
python sarimax_forecast.py
```

**Model Multivariate (IN + OUT Traffic):**
```bash
python sarimax_multivariate.py
```

### **3. Lihat Hasil**

Semua hasil tersimpan di folder `results/`:
- **Visualisasi:** File PNG
- **Prediksi:** File CSV
- **Model:** File PKL (bisa dipake ulang tanpa training lagi)

---

## Performa Model

### **Model Univariate (Total Traffic)**

| Metric | Value | Artinya |
|--------|-------|---------|
| **MAE** | 64.30 | Prediksi meleset rata-rata 64 orang |
| **RMSE** | 85.71 | Ukuran error keseluruhan |
| **MAPE** | 60.89% | Error rata-rata 60.89% (hanya dihitung dari data non-zero) |
| **R² Score** | 0.5867 | Model bisa menjelaskan 59% pola data |

**Catatan MAPE:**
- MAPE dihitung hanya untuk data dengan nilai aktual > 0
- Ini karena 50% data bernilai 0, sehingga MAPE klasik akan menghasilkan NaN
- Formula: `MAPE = mean(|actual - predicted| / actual) × 100` untuk actual > 0

### **Model Multivariate (IN + OUT)**

| Metric | IN | OUT | TOTAL |
|--------|-----|-----|-------|
| **MAE** | 51.03 | 26.27 | 66.54 |
| **RMSE** | 72.31 | 35.45 | 88.04 |
| **R² Score** | 0.2021 | 0.5993 | 0.5639 |

**Temuan:**
- Model OUT jauh lebih akurat (R²=0.60) dibanding model IN (R²=0.20)
- Traffic keluar lebih predictable dan stabil
- Traffic masuk lebih volatile dan susah diprediksi

---

## Temuan Utama

### **1. Jam Tersibuk**
- **Paling ramai:** 15:00-18:00 (sore)
- **Jam 16:00** - Traffic tertinggi (rata-rata 137 orang)
- **Jam 17:00** - Traffic tinggi (rata-rata 135 orang)

### **2. Weekend vs Hari Kerja**
- **Weekend 2x lebih ramai** dibanding hari kerja
- Rata-rata traffic weekend: 86 orang
- Rata-rata traffic weekday: -45 (PERHATIAN: prediksi negatif - tidak valid)

### **3. Pola Harian**
- **Pagi (00:00-09:00):** Traffic rendah/kosong
- **Siang-Sore (10:00-21:00):** Traffic tinggi
- **Malam (22:00-23:00):** Traffic turun drastis

### **4. Keterbatasan Model**
- Data cuma 7 hari (terlalu sedikit untuk pola hari kerja)
- 50% data bernilai 0 (jam tutup/traffic rendah) - membuat model sulit belajar pola
- Prediksi hari kerja menghasilkan angka negatif (tidak valid)
- Model cenderung underpredict saat traffic spike ekstrem (>400 orang)
- Pola seasonal 24 jam berhasil tertangkap dengan baik
- MAPE tinggi (60.89%) karena banyak data kecil - error persentase jadi besar

---

## Rekomendasi Operasional

### **Jadwal Staff:**

| Waktu | Level Traffic | Kebutuhan Staff |
|-------|--------------|--------------|
| **Sabtu/Minggu 15:00-18:00** | Puncak (200-250) | Maksimal |
| **Sabtu/Minggu 10:00-14:00** | Tinggi (100-200) | Sedang |
| **Sabtu/Minggu 19:00-21:00** | Sedang (50-100) | Sedang |
| **Pagi/Malam (00:00-09:00, 22:00-23:00)** | Rendah (<50) | Minimal |

### **Perencanaan Kapasitas:**
- **Kapasitas maksimal:** Sekitar 250 orang (Sabtu sore)
- **Kapasitas normal:** Sekitar 0-50 orang (pagi/malam)

---

## Keterbatasan & Pengembangan Kedepan

### **Keterbatasan Saat Ini:**
1. Data terlalu sedikit (7 hari) - pola belum robust
2. 50% data bernilai 0 - membuat perhitungan MAPE tidak standar
3. Prediksi bisa negatif - perlu tambah constraint (min=0)
4. Pola hari kerja belum terbentuk - butuh data lebih banyak
5. Cenderung underpredict saat spike - model terlalu konservatif
6. MAPE tinggi karena banyak nilai kecil (small denominator problem)

### **Rekomendasi Pengembangan:**
1. Kumpulkan data 4-8 minggu untuk pola yang lebih jelas
2. Tambahkan constraint: predicted_traffic >= 0
3. Pertimbangkan metrik alternatif selain MAPE untuk data dengan banyak nilai 0:
   - Symmetric MAPE (sMAPE)
   - Weighted MAPE
   - Atau fokus pada MAE/RMSE saja
4. Coba ensemble model (SARIMAX + XGBoost)
5. Tambah faktor eksternal (cuaca, event khusus, hari libur)

---

## Detail Teknis

### **Train/Test Split:**
- **Train:** 5 hari (120 jam) - 71%
- **Test:** 2 hari (48 jam) - 29%

### **Test Stationarity:**
- **ADF Test:** p-value < 0.05 (Stasioner)
- **KPSS Test:** p-value > 0.05 (Stasioner)

### **Validasi Model:**
- Analisis residual (white noise check)
- Ljung-Box test
- Q-Q plot (normality check)
- ACF of residuals

---

## Kontak & Support

Untuk pertanyaan atau issue, silakan buka issue di repository atau hubungi maintainer.

---

## License

MIT License

---

## Acknowledgments

- **statsmodels** - Implementasi SARIMAX
- **scikit-learn** - Metrics
- **matplotlib & seaborn** - Visualisasi

---

**Terakhir diupdate:** 2 Oktober 2025

### **Changelog:**
- **2 Oktober 2025:** Perbaikan perhitungan MAPE untuk menangani data dengan nilai 0 (menggunakan filter non-zero values)
