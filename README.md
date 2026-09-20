# Analisis Sentimen & Identifikasi Penyebab Keluhan Pelanggan Tokopedia

> Klasifikasi sentimen (TF-IDF + SVM) dan topic modeling (NMF) pada review produk Tokopedia 2025, lalu menghubungkan topik keluhan dengan kategori produk untuk menghasilkan insight bisnis.

**Tools:** Python · pandas · scikit-learn · NLP (TF-IDF, NMF) · SciPy · Matplotlib · Seaborn · Jupyter Notebook

---

## Ringkasan Proyek

Proyek ini menganalisis 65.543 review produk Tokopedia untuk: (1) membangun model klasifikasi sentimen langsung dari teks review, (2) menemukan tema keluhan utama dari review negatif, dan (3) melihat kategori produk mana yang paling terkait dengan tiap tema keluhan.

## Business Understanding

- **Masalah:** Penjual dan platform menerima banyak review, tetapi sulit mengetahui apa yang sebenarnya membuat pelanggan kecewa, dan apakah masalahnya berbeda antarkategori produk.
- **Tujuan:** Mendeteksi review negatif secara otomatis dari teks dan mengelompokkan keluhannya ke dalam tema yang bisa ditindaklanjuti.
- **Pertanyaan utama:**
  1. Seberapa baik sentimen bisa diprediksi dari teks review (`review_text`)?
  2. Apa tema keluhan utama pada review negatif?
  3. Apakah tema keluhan berbeda antarkategori produk (`product_category`)?

## Dataset

- **Sumber:** [Tokopedia Product Reviews 2025 (Kaggle, salmanabdu)](https://www.kaggle.com/datasets/salmanabdu/tokopedia-product-reviews-2025)
- **Ukuran:** 65.543 review
- **Kolom utama yang dipakai:** `review_text`, `rating`, `sentiment_label`, `product_category`
- **Distribusi sentimen:** sangat tidak seimbang (*imbalanced*), sekitar 97,6% positive, 1,2% negative, dan 1,2% neutral (798 review negatif).
- **Catatan penting:** `sentiment_label` pada dataset ini adalah turunan langsung dari `rating` (rating 4-5 = positive, 3 = neutral, 1-2 = negative), bukan hasil analisis teks. Hal ini sudah diverifikasi pada tahap EDA.
- Data **tidak disertakan** di repo ini. Notebook memuat dataset langsung dari Kaggle memakai `kagglehub`.

## Metodologi

1. **EDA:** cek missing value, distribusi kategori, distribusi rating dan sentimen, serta verifikasi bahwa label sentimen berasal dari rating.
2. **Text cleaning:** huruf kecil, hapus URL, hapus angka dan tanda baca, rapikan spasi.
3. **Klasifikasi sentimen:** TF-IDF (5.000 fitur) dengan pembagian data 80/20 (`stratify`). Dua model dibandingkan:
   - Naive Bayes sebagai baseline
   - LinearSVC dengan `class_weight='balanced'` untuk menangani ketidakseimbangan kelas
4. **Topic modeling:** NMF dengan 5 topik pada review negatif, memakai TF-IDF dan daftar stopword bahasa Indonesia.
5. **Hubungan topik dan kategori:** tabel silang topik dominan vs kategori produk, uji chi-square, dan visualisasi heatmap.

## Hasil

### Perbandingan Model Klasifikasi

Data uji: 13.109 review (160 negative, 160 neutral, 12.789 positive).

| Model | Akurasi | F1 negative | F1 neutral | Macro F1 |
|---|---|---|---|---|
| Naive Bayes (baseline) | 97,53% | 0,02 | 0,00 | 0,34 |
| SVM (LinearSVC, balanced) | 96,32% | 0,38 | 0,19 | 0,52 |

Akurasi saja menyesatkan pada data ini: model yang selalu menebak `positive` sudah mencapai 97,56% (12.789 dari 13.109). Karena itu perbandingan dilakukan lewat F1 kelas minoritas dan macro F1.

Naive Bayes hampir hanya mengenali kelas mayoritas: recall kelas `negative` hanya 0,01 dan `neutral` 0,00. SVM dengan `class_weight='balanced'` memiliki akurasi lebih rendah, tetapi recall `negative` naik menjadi 0,41 dan F1 `negative` naik dari 0,02 menjadi 0,38, sehingga dipilih sebagai model final untuk mendeteksi keluhan pelanggan. Performa pada kelas `neutral` masih rendah (F1 0,19).

### Topik Keluhan pada Review Negatif

| Topik | Kata kunci teratas | Interpretasi |
|---|---|---|
| 1 | barang, rusak, dikirim, kualitas, cacat, lengkap, kirim, kurang, penjual, kurir | Barang rusak/cacat saat pengiriman |
| 2 | beli, pengiriman, dikirim, lama, mau, penjual, kirim, baru, padahal, kecewa | Pengiriman lambat dan kekecewaan |
| 3 | sesuai, gambar, kecil, pesan, warna, deskripsi, ternyata, kualitas, tipis, pesanan | Produk tidak sesuai deskripsi/gambar |
| 4 | bahan, bagus, kurang, ukuran, jelek, kualitas, keras, karakteristik, harga, terlalu | Kualitas bahan/material kurang baik |
| 5 | pecah, telur, busuk, telurnya, banyak, bau, sampai, pcs, baru, tolong | Produk pecah/busuk (umumnya Makanan & Minuman) |

### Hubungan Topik dan Kategori Produk

Uji chi-square: χ² = 158,42 (dof = 20), p-value ≈ 1,5 × 10⁻²³. Hubungan antara kategori produk dan topik keluhan signifikan secara statistik (p < 0,05). Catatan: 7 dari 30 sel memiliki nilai harapan di bawah 5 (kategori dengan sedikit review negatif), sehingga hasil uji perlu dibaca dengan hati-hati.

Sebaran topik dominan per kategori (jumlah review negatif):

| Kategori | T1 | T2 | T3 | T4 | T5 | Total |
|---|---|---|---|---|---|---|
| Elektronik | 5 | 14 | 1 | 3 | 1 | 24 |
| Handphone & Tablet | 6 | 47 | 1 | 8 | 0 | 62 |
| Kesehatan | 8 | 24 | 1 | 5 | 1 | 39 |
| Makanan & Minuman | 26 | 136 | 10 | 33 | 58 | 263 |
| Olahraga | 39 | 122 | 36 | 78 | 2 | 277 |
| Pertukangan | 25 | 77 | 11 | 17 | 3 | 133 |

![Heatmap topik keluhan per kategori produk](images/heatmap_topik_kategori.png)

## Temuan Utama

- Model klasifikasi sentimen (SVM) berhasil dibangun langsung dari teks review, dengan penanganan khusus untuk data yang sangat *imbalanced*.
- Topik 2 (pengiriman lambat) adalah topik dominan di semua kategori (sekitar 53% dari seluruh review negatif), sehingga masalahnya bersifat lintas kategori, bukan spesifik satu kategori.
- Topik 5 (produk pecah/busuk) sangat terkonsentrasi di Makanan & Minuman: 58 dari 65 review pada topik ini (sekitar 89%).
- Topik 4 (kualitas bahan/material) paling menonjol di Olahraga: 28% review negatif Olahraga masuk topik ini, dibanding sekitar 12-13% di kategori lain.

## Rekomendasi Bisnis

1. Prioritaskan perbaikan logistik pengiriman secara lintas kategori.
2. Terapkan SOP kemasan khusus (anti pecah, tahan suhu) untuk kategori Makanan & Minuman.
3. Tingkatkan quality control material/bahan untuk kategori Olahraga.

## Keterbatasan

- Jumlah review negatif relatif kecil (798 dari 65.543), sehingga topic modeling tidak dipecah per kategori agar hasilnya tetap andal.
- Performa model pada kelas minoritas masih rendah (F1 `negative` 0,38 dan `neutral` 0,19), jadi hasil klasifikasi sebaiknya dibaca sebagai baseline, bukan model siap produksi.
- Label sentimen berasal dari rating, sehingga model belajar memprediksi rating dari teks, bukan sentimen hasil anotasi manual.
- `review_date` hanya memiliki granularitas harian, sehingga analisis tren waktu yang lebih presisi tidak dapat dilakukan.
- Rendahnya proporsi review negatif konsisten dengan *selection bias* pada platform review: pelanggan puas cenderung lebih sering menulis review dibanding pelanggan kecewa.

## Pengembangan Lanjutan

- Normalisasi kata gaul dan singkatan (misalnya "brg", "dr", "nga") sebelum TF-IDF.
- Mencoba model berbasis bahasa Indonesia seperti IndoBERT dan membandingkannya dengan baseline SVM.
- Mencoba teknik penanganan data tidak seimbang lain (oversampling/undersampling) dan penyesuaian threshold.

## Struktur Repository

```
tokopedia-review-nlp/
├── tokopedia_sentiment_analysis.ipynb
├── images/              # grafik yang dipakai di README
├── requirements.txt
├── .gitignore
└── README.md
```

## Cara Menjalankan

1. Clone repo dan install dependensi:

```bash
git clone https://github.com/bikogabriel12-gif/tokopedia-review-nlp.git
cd tokopedia-review-nlp
pip install -r requirements.txt
```

2. Buka `tokopedia_sentiment_analysis.ipynb` dengan Jupyter Notebook atau Google Colab.
3. Jalankan semua cell. Saat `kagglehub.login()` meminta kredensial, masukkan API token Kaggle milikmu sendiri (jangan menyimpannya di dalam notebook).

## Author

**[Biko Gabriel]** · [LinkedIn](linkedin.com/in/biko-elfarol-mohammed-gabriel-b35209163) · [Kaggle](https://www.kaggle.com/bikzzz) · [GitHub](https://github.com/bikogabriel12-gif)
