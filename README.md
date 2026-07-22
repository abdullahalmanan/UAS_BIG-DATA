# 🧠 MindCheck — Aplikasi Deteksi Tingkat Stres (Growing_Stress)

Aplikasi web berbasis **Streamlit** untuk memprediksi kecenderungan peningkatan
level stres (*Growing Stress*: `Yes` / `No` / `Maybe`) menggunakan model
**Random Forest**, hasil dari UAS mata kuliah Big Data.

Dibuat oleh **Abdullah Al Manan (41523010025)** — Teknik Informatika,
Fakultas Ilmu Komputer, Universitas Mercu Buana.

---

## 📁 Struktur Folder

```
mental_health_app/
├── UAS_BigData_Mental_Health_Final.ipynb   # Notebook UAS asli — sumber pipeline & model
├── app.py                       # Aplikasi Streamlit utama
├── train_model.py                # Skrip training, disusun langsung dari kode notebook di atas
├── Mental_Health_Dataset.csv     # Dataset asli (292.364 baris)
├── mental_health_sample.csv      # Sample data untuk halaman EDA (20.000 baris, loading lebih cepat)
├── model_rf.pkl                  # Model Random Forest terlatih (hasil GridSearchCV di notebook)
├── scaler.pkl                    # StandardScaler terlatih
├── label_encoders.pkl            # LabelEncoder untuk setiap kolom kategorikal
├── fitur_terpilih.json           # Daftar 16 fitur final yang dipakai model
├── hasil_metrik_model.json       # Metrik perbandingan 5 model (Accuracy, Precision, Recall, F1)
├── .streamlit/config.toml        # Konfigurasi tema (dipaksa terang, biru-putih)
└── requirements.txt              # Daftar dependency Python
```

**Catatan penting:** `train_model.py` bukan model terpisah — kodenya diambil langsung dari
notebook `UAS_BigData_Mental_Health_Final.ipynb` (preprocessing, feature selection, dan
parameter Random Forest hasil `GridSearchCV` identik 1:1), sehingga hasil prediksi aplikasi
ini **konsisten dengan hasil evaluasi di notebook** (akurasi Random Forest ±98,92%).

## 🚀 Cara Menjalankan

1. **Install dependency** (disarankan menggunakan virtual environment):
   ```bash
   pip install -r requirements.txt
   ```

2. **(Opsional) Latih ulang model** — hanya perlu dijalankan jika ingin melatih ulang
   dari awal. File model (`.pkl`) sudah disediakan sehingga langkah ini bisa dilewati:
   ```bash
   python train_model.py
   ```

3. **Jalankan aplikasi Streamlit**:
   ```bash
   streamlit run app.py
   ```

4. Buka browser ke alamat yang muncul di terminal (biasanya `http://localhost:8501`).

## 🖥️ Fitur Aplikasi

| Halaman | Deskripsi |
|---|---|
| 🏠 Beranda | Ringkasan aplikasi, statistik utama, dan disclaimer |
| 📊 Dataset & EDA | Eksplorasi interaktif dataset (distribusi target, demografi, hubungan antar variabel) |
| 🔮 Prediksi Stres | Form input untuk memprediksi Growing_Stress secara real-time beserta probabilitas tiap kelas |
| ⚖️ Perbandingan Model | Tabel & grafik perbandingan performa Logistic Regression, KNN, Naive Bayes, Random Forest, dan XGBoost |
| ℹ️ Tentang | Informasi akademik, teknologi, dan dataset yang digunakan |

## ⚠️ Disclaimer

Aplikasi ini dibuat **khusus untuk keperluan akademik/edukatif** sebagai bagian dari
UAS mata kuliah Big Data, dan **bukan merupakan alat diagnosis klinis**. Hasil
prediksi tidak menggantikan konsultasi dengan psikolog atau tenaga profesional
kesehatan mental.

## 🔗 Keterkaitan dengan Notebook

Seluruh preprocessing (missing value handling, duplicate removal, feature
engineering `risk_score`, encoding, feature selection Chi-Square, standardisasi)
serta parameter model (hasil `GridSearchCV`) pada aplikasi ini **identik** dengan
yang digunakan pada notebook `UAS_BigData_Mental_Health.ipynb`, sehingga hasil
prediksi aplikasi konsisten dengan hasil evaluasi pada notebook tersebut.
