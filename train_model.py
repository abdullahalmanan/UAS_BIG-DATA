"""
train_model.py
================
Skrip ini disusun LANGSUNG dari kode pada notebook
`UAS_BigData_Mental_Health_Final.ipynb` (disertakan dalam folder yang sama),
mereplikasi persis seluruh preprocessing dan pemodelan pada notebook tersebut:
missing value handling, duplicate removal, feature engineering (risk_score),
label encoding, feature selection (16 fitur hasil Chi-Square), standardisasi,
stratified sampling, hingga parameter Random Forest hasil GridSearchCV
(n_estimators=100, max_depth=15, min_samples_split=5).

Artefak yang dihasilkan (model, scaler, encoder, dll.) dipakai langsung oleh
app.py sehingga hasil prediksi di aplikasi Streamlit ini KONSISTEN dengan
hasil evaluasi pada notebook.

Menjalankan: python train_model.py
"""

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

DATA_PATH = "Mental_Health_Dataset.csv"
RANDOM_STATE = 42

print("=" * 60)
print("1. Membaca & membersihkan dataset")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"Ukuran awal: {df.shape}")

# --- Missing value handling ---
df["self_employed"] = df["self_employed"].fillna(df["self_employed"].mode()[0])

# --- Duplicate removal ---
df = df.drop_duplicates()
print(f"Ukuran setelah cleaning: {df.shape}")

# --- Feature engineering: risk_score ---
def hitung_risk_score(row):
    skor = 0
    skor += 1 if row["family_history"] == "Yes" else 0
    skor += 1 if row["treatment"] == "Yes" else 0
    skor += 1 if row["Mental_Health_History"] == "Yes" else 0
    skor += 1 if row["Coping_Struggles"] == "Yes" else 0
    return skor

df["risk_score"] = df.apply(hitung_risk_score, axis=1)
df = df.drop(columns=["Timestamp"])

print("\n" + "=" * 60)
print("2. Encoding variabel kategorikal")
print("=" * 60)

df_encoded = df.copy()
kolom_kategorikal = df_encoded.select_dtypes(include="object").columns.tolist()
print("Kolom yang di-encode:", kolom_kategorikal)

label_encoders = {}
for col in kolom_kategorikal:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

print("\n" + "=" * 60)
print("3. Feature selection (fitur final, sama seperti notebook)")
print("=" * 60)

FITUR_TERPILIH = [
    "Changes_Habits", "Occupation", "Work_Interest", "Days_Indoors", "Mood_Swings",
    "Coping_Struggles", "Gender", "risk_score", "Social_Weakness", "family_history",
    "Mental_Health_History", "treatment", "care_options", "Country", "self_employed",
    "mental_health_interview",
]
print("Fitur yang digunakan:", FITUR_TERPILIH)

X = df_encoded[FITUR_TERPILIH]
y = df_encoded["Growing_Stress"]

print("\n" + "=" * 60)
print("4. Standardisasi & sampling")
print("=" * 60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

SAMPLE_SIZE = 30000
X_model, _, y_model, _ = train_test_split(
    X_scaled, y, train_size=SAMPLE_SIZE, random_state=RANDOM_STATE, stratify=y
)

X_train, X_test, y_train, y_test = train_test_split(
    X_model, y_model, test_size=0.2, random_state=RANDOM_STATE, stratify=y_model
)
print(f"Data latih: {X_train.shape}, Data uji: {X_test.shape}")

print("\n" + "=" * 60)
print("5. Melatih seluruh model (sama seperti notebook)")
print("=" * 60)

hasil_metrik = {}

def latih_dan_evaluasi(nama, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    hasil_metrik[nama] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1-Score": f1_score(y_test, y_pred, average="weighted"),
    }
    print(f"{nama:25s} -> Accuracy: {hasil_metrik[nama]['Accuracy']:.4f}")
    return model

model_lr = latih_dan_evaluasi(
    "Logistic Regression",
    LogisticRegression(solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE),
)
model_knn = latih_dan_evaluasi(
    "K-Nearest Neighbors",
    KNeighborsClassifier(n_neighbors=15, weights="distance", metric="minkowski", p=2, n_jobs=-1),
)
model_nb = latih_dan_evaluasi("Gaussian Naive Bayes", GaussianNB(var_smoothing=1e-9))

# Random Forest dengan parameter hasil GridSearchCV pada notebook
model_rf = latih_dan_evaluasi(
    "Random Forest (Tuned)",
    RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_split=5,
        random_state=RANDOM_STATE, n_jobs=-1,
    ),
)

model_xgb = latih_dan_evaluasi(
    "XGBoost",
    XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        objective="multi:softmax", num_class=3, random_state=RANDOM_STATE,
        eval_metric="mlogloss", n_jobs=-1,
    ),
)

print("\n" + "=" * 60)
print("6. Menyimpan model terbaik (Random Forest) & seluruh artefak")
print("=" * 60)

# Model final yang dipakai aplikasi: Random Forest (performa terbaik & konsisten)
model_final = model_rf

joblib.dump(model_final, "model_rf.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

with open("fitur_terpilih.json", "w") as f:
    json.dump(FITUR_TERPILIH, f)

df_metrik = pd.DataFrame(hasil_metrik).T.reset_index().rename(columns={"index": "Model"})
df_metrik.to_json("hasil_metrik_model.json", orient="records")

# Simpan sample dataset kecil untuk keperluan EDA di aplikasi (agar loading cepat)
df.sample(n=min(20000, len(df)), random_state=RANDOM_STATE).to_csv("mental_health_sample.csv", index=False)

print("\nSemua artefak berhasil disimpan:")
print("- model_rf.pkl")
print("- scaler.pkl")
print("- label_encoders.pkl")
print("- fitur_terpilih.json")
print("- hasil_metrik_model.json")
print("- mental_health_sample.csv")
print("\nSelesai.")
