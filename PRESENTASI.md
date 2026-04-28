# Analisis ISPU DKI Jakarta — Ringkasan Presentasi

**Dataset:** Indeks Standar Pencemar Udara (ISPU) DKI Jakarta 2022–2023  
**Sumber:** data.go.id (portal data terbuka Indonesia)  
**Tools:** Python · pandas · seaborn · scikit-learn

---

## Apa itu ISPU?

ISPU (Indeks Standar Pencemar Udara) adalah angka yang menggambarkan mutu udara harian berdasarkan konsentrasi 6 polutan:

| Polutan | Simbol | Satuan |
|---|---|---|
| Partikel kasar | PM10 | μg/m³ |
| Partikel halus | **PM2.5** | μg/m³ |
| Sulfur dioksida | SO₂ | μg/m³ |
| Karbon monoksida | CO | mg/m³ |
| Ozon | O₃ | μg/m³ |
| Nitrogen dioksida | NO₂ | μg/m³ |

Nilai `max` = indeks tertinggi dari 6 polutan pada hari itu → menentukan `kategori` kualitas udara.

---

## 1. Data Understanding

**Dimensi awal: 1.825 baris × 12 kolom**

Yang perlu dicatat dari output `info()`:
- Kolom polutan bertipe `float` — karena tanda `-` dan `---` di CSV sudah otomatis dibaca sebagai `NaN`
- Nilai hilang (NaN) cukup banyak → **PM2.5: 295 baris (16%), PM10: 222 baris (12%)**
- Ada **21 baris** dengan `kategori = "TIDAK ADA DATA"` → semua sensor mati, tidak ada pengukuran

`describe()` menunjukkan:
- Rata-rata `max` ≈ **73.6** → masuk kategori SEDANG
- Nilai `max` bisa mencapai **287** → kasus ekstrem SANGAT TIDAK SEHAT

---

## 2. EDA (Exploratory Data Analysis)

### Plot 1 — Distribusi Nilai `max`
> Histogram + kurva KDE nilai indeks polutan tertinggi harian.

**Interpretasi:** Distribusi condong ke kanan (*right-skewed*) — mayoritas hari berada di nilai 50–100 (SEDANG), namun ada ekor panjang ke nilai tinggi (TIDAK SEHAT).

### Plot 2 — Jumlah Observasi per Kategori
> Bar chart frekuensi tiap kategori kualitas udara.

**Interpretasi:** **SEDANG mendominasi** (>1.000 hari), diikuti BAIK, lalu TIDAK SEHAT dan SANGAT TIDAK SEHAT sangat jarang. Artinya Jakarta sebagian besar berada di mutu udara sedang, belum bersih namun tidak selalu berbahaya.

### Plot 3 — Rata-rata PM2.5 per Stasiun
> Bar chart horizontal rata-rata PM2.5 di tiap stasiun SPKU.

**Interpretasi:** Beberapa stasiun secara konsisten mencatat PM2.5 lebih tinggi — menunjukkan ada variasi spasial polusi di DKI Jakarta. PM2.5 dipilih karena paling berbahaya (dapat masuk ke paru-paru dalam).

---

## 3. Data Cleaning

**Masalah yang ditemukan → solusi:**

| Masalah | Solusi | Alasan |
|---|---|---|
| Kolom `tanggal` bertipe string | Konversi ke `datetime` | Agar bisa ekstrak fitur waktu |
| Baris duplikat | `drop_duplicates()` | Hindari bias model |
| 21 baris "TIDAK ADA DATA" | Filter & hapus | Tidak informatif, merusak imputasi |
| NaN di kolom polutan | Imputasi **median** | Median lebih tahan terhadap outlier vs. mean |
| Spasi berlebih di nama stasiun | `str.strip()` | Menyatukan nama yang seharusnya sama |

**Output:** Dimensi berubah **(1.825 → 1.804 baris)** · Semua kolom: **0 nilai hilang**

---

## 4. Data Transformation

Tujuan: mengubah data ke format optimal untuk machine learning.

| Langkah | Teknik | Output |
|---|---|---|
| Ekstraksi waktu | `.dt.year`, `.dt.month` | Kolom `tahun`, `bulan` → tangkap pola musiman |
| Transformasi log | `np.log1p(x)` | Mengurangi skewness pada data polutan |
| Standarisasi | `StandardScaler` → mean=0, std=1 | Semua fitur berskala sama |
| One-Hot Encoding | `OneHotEncoder` | Ubah `kategori`, `stasiun`, `parameter_pencemar_kritis` ke kolom biner |

**Output:** Kolom bertambah **(12 → 31 kolom)** karena OHE mengekspansi setiap kategori.

---

## 5. Data Reduction

Setelah transformasi ada 31 fitur — banyak di antaranya berkorelasi/redundan.

### Variance Threshold
Fitur dengan varian < 0.01 dihapus (hampir konstan = tidak informatif).  
**Hasil: 31 → 28 fitur** (3 fitur dihapus)

### PCA (Principal Component Analysis)
PCA memproyeksikan 28 fitur ke dimensi baru yang memaksimalkan informasi.

| Komponen | Varians Dijelaskan | Interpretasi |
|---|---|---|
| PC1 | **~61%** | Tingkat polusi keseluruhan |
| PC2 | **~12%** | Variasi antar stasiun / musim |
| **Total** | **~73%** | 73% informasi tersimpan dalam 2 dimensi |

**Scatter plot PCA:** Titik-titik yang berdekatan = kondisi udara serupa. Pola kluster mencerminkan pengelompokan alami berdasarkan kategori kualitas udara.

---

## Kesimpulan

| Tahap | Sebelum | Sesudah |
|---|---|---|
| Raw Data | 1.825 baris, 12 kolom, banyak NaN | — |
| Cleaning | — | **1.804 baris**, 0 NaN |
| Transformation | 12 kolom | **31 kolom** (numerik semua) |
| Variance Threshold | 31 fitur | **28 fitur** |
| PCA | 28 fitur | **2 komponen** (~73% varians) |

**Temuan utama:**
- **PM2.5 adalah polutan dominan** — paling sering menjadi `parameter_pencemar_kritis`
- **Kategori SEDANG mendominasi** di DKI Jakarta sepanjang 2022–2023
- **PCA efektif** — 73% informasi terwakili dalam 2 dimensi, menandakan korelasi kuat antar polutan
- Data ini siap untuk model **Regresi Linear** (prediksi nilai `max`) atau **Regresi Logistik** (klasifikasi `kategori`)
