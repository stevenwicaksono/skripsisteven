# Prediksi Hasil Tanaman Padi dengan Random Forest

Ini repo dari skripsiku (sudah terbit di Jurnal JTIK, Vol. 11 No. 1, 2027).
Intinya: prediksi produktivitas padi di 37 provinsi Indonesia pakai Random
Forest, lalu hasilnya dikonversi jadi produksi (ton) pakai luas panen aktual.
Seleksi fitur dan tuning hyperparameter dikerjain barengan, hal yang waktu itu
belum banyak dibahas di penelitian padi sebelumnya.

## Data

- **BPS** (2018-2024): luas panen, produktivitas, produksi padi per provinsi
- **NASA POWER API**: data cuaca (suhu, curah hujan, kelembaban, tekanan udara,
  angin, radiasi, UV) berdasarkan titik pusat tiap provinsi, diagregasi jadi
  tahunan
- Hasil gabungan: 239 baris x 32 fitur
- Split kronologis: 2018-2022 buat latih (165 baris), 2023-2024 buat uji (74 baris)

## Alur

Pakai metodologi SEMMA (Sample, Explore, Modify, Model, Assess):

1. Gabung data BPS + NASA POWER berdasarkan provinsi dan tahun
2. Bikin fitur turunan (`temp_range`, `heat_index`, `hujan_per_hari`)
3. Seleksi fitur pakai feature importance RF (threshold 0,005), 32 fitur jadi 16
4. Tuning pakai GridSearchCV + TimeSeriesSplit 5-fold (648 kombinasi)
5. Evaluasi di level produktivitas dan produksi, dibandingin sama Linear Regression

## Hasil

| Level | R² | MAPE |
|---|---|---|
| Produktivitas (test) | 75,45% | 6,95% |
| Produksi (test) | 99,79% | 6,95% |

Catatan jujur: R² produksi yang tinggi itu sebagian besar karena luas panen
aktual jadi pengali di rumus konversi. Jadi angka yang lebih mewakili
kemampuan model sebenarnya ya yang di level produktivitas.

Dibanding Linear Regression, RMSE produksi turun sekitar 77% (115.600 ton
vs 512.713 ton).

Hyperparameter terbaik: `n_estimators=100`, `max_depth=None`,
`max_features=0.5`, `min_impurity_decrease=0.0001`, `min_samples_leaf=1`,
`min_samples_split=2`.

## Keterbatasan

- Data cuma 7 tahun, learning curve-nya nunjukin model masih bisa naik kalau datanya lebih panjang
- Empat provinsi pemekaran Papua cuma punya data 2023-2024, jadi errornya tinggi (Papua Barat Daya sampai 33%)
- `Provinsi_Encoded` mendominasi feature importance (61%), jadi model banyak belajar dari "ini provinsi mana", bukan murni dari iklim

## Cara jalanin

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python main.py
```

Dibuat dengan Python 3.12, Pandas 2.2.3, NumPy 2.2.4, Scikit-learn 1.6.1.

## Sitasi

Wicaksono, S. A. P., & Akbar, M. (2027). Prediksi Hasil Tanaman Padi
Menggunakan Random Forest dengan Seleksi Fitur Berbasis Feature Importance
dan Optimasi Hyperparameter GridSearchCV. *Jurnal JTIK*, 11(1), 160-171.
https://doi.org/10.35870/jtik.v11i1.7567

## Lisensi

MIT
