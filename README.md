# Sistem Pencarian Lokasi Minimarket [![Streamlit App](https://img.shields.io/badge/Streamlit-Live-red?logo=streamlit)](https://nare-sr-uas.streamlit.app/)


Sistem pencarian lokasi minimarket (Alfamart & Indomaret) berbasis web dengan evaluasi Information Retrieval menggunakan metrik MAP, Precision, dan Recall.

## Fitur Utama

- **BM25 Ranking**: Algoritma ranking berbasis TF-IDF untuk kecocokan teks
- **Haversine Distance**: Kalkulasi jarak geografis dari lokasi pengguna
- **Heuristic Ranking**: Kombinasi skor BM25, jarak, rating, dan popularitas
- **Evaluasi IR**: Precision@K, Recall@K, dan Average Precision
- **Visualisasi Peta**: Peta interaktif dengan marker lokasi
- **Filter Store**: Filter berdasarkan tipe toko (Alfamart/Indomaret)

## Struktur Project

```
project/
│
├── app.py                  # Entry point Streamlit
├── config.py               # Konfigurasi bobot dan parameter
├── requirements.txt        # Dependencies
│
├── data/
│   └── Data_Alfamart Indomaret_South Jakarta.csv
│
├── core/
│   ├── __init__.py
│   ├── preprocessing.py    # Text cleaning & normalization
│   ├── bm25_engine.py      # BM25 scoring
│   ├── geo_utils.py        # Haversine calculation
│   ├── ranking.py          # Final heuristic ranking
│   └── evaluation.py       # Precision, Recall, MAP
│
└── ui/
    ├── __init__.py
    ├── layout.py           # Layout tanpa sidebar
    ├── result_table.py     # Tabel hasil ranking
    ├── map_view.py         # Visualisasi peta
    └── metrics_view.py     # Tampilan evaluasi
```

## Cara Menjalankan

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Jalankan Aplikasi

```bash
streamlit run app.py
```

### 3. Buka Browser

Aplikasi akan otomatis terbuka di browser. Jika tidak, buka:
```
http://localhost:8501
```

## Metrik Evaluasi

### Precision@K
Mengukur ketepatan hasil:
```
Precision@K = Jumlah hasil relevan di top-K / K
```
*"Dari K hasil teratas, berapa persen yang benar-benar relevan?"*

### Recall@K
Mengukur kelengkapan hasil:
```
Recall@K = Jumlah hasil relevan di top-K / Total hasil relevan
```
*"Dari semua lokasi relevan, berapa persen yang berhasil ditemukan?"*

### Average Precision (AP)
Mengukur kualitas ranking secara keseluruhan:
```
AP = (1/|R|) × Σ P(k) × rel(k)
```
*"Seberapa baik sistem menempatkan hasil relevan di posisi atas?"*

## Konfigurasi

Edit `config.py` untuk menyesuaikan parameter:

```python
# BM25 Parameters
BM25_K1 = 1.5
BM25_B = 0.75

# Ranking Weights
WEIGHTS = {
    'bm25_score': 0.4,
    'distance_score': 0.3,
    'rating_score': 0.2,
    'popularity_score': 0.1
}

# Evaluation
DEFAULT_K = 10
```

## Ground Truth (Relevance Labeling)

Sistem menggunakan **rule-based relevance labeling**:

- **Relevan jika:**
  - Skor BM25 > 0 (kecocokan teks)
  - Jarak < 10 km (jika lokasi aktif)
  - Tipe toko sesuai query

- **Tidak relevan jika:**
  - Tidak ada kecocokan teks
  - Jarak terlalu jauh
  - Tipe toko tidak sesuai

## Justifikasi Akademik

Evaluasi ditampilkan di UI untuk memberikan:
1. **Transparansi** kualitas sistem
2. **Bukti objektif** bahwa hasil pencarian dapat diukur
3. **Standar IR** (Information Retrieval) yang teruji

## Teknologi

- **Streamlit**: Web framework
- **Pandas**: Data processing
- **Folium**: Map visualization
- **Python 3.8+**: Core language

## Lisensi

Proyek ini dibuat untuk keperluan akademik / UAS.

