import streamlit as st
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Memuat model Naive Bayes dan scaler yang telah disimpan
with open('nb_model.pkl', 'rb') as file:
    nb_model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Memuat dataset untuk evaluasi
@st.cache
def load_data():
    data = pd.read_csv('datadiabetes.csv')
    return data

data = load_data()
X = data.drop(columns=['Diabetes_012'])
y = data['Diabetes_012']

# Membagi dataset menjadi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fungsi untuk melakukan prediksi dengan model Naive Bayes
def predict_nb(features):
    # Normalisasi fitur
    features_normalized = scaler.transform([features])

    # Prediksi menggunakan model Naive Bayes
    prediction = nb_model.predict(features_normalized)

    # Menampilkan hasil prediksi
    return prediction[0]

def tampilkan_NB():
    st.title("🔮 Halaman Prediksi Risiko Diabetes")

    st.write("""
        Pada halaman ini, Anda dapat memasukkan data untuk memprediksi apakah seseorang berisiko
        menderita diabetes berdasarkan parameter tertentu.
    """)

    # Input data pengguna
    high_bp = st.radio("Tekanan Darah Tinggi (HighBP)", [0, 1], index=0)
    high_chol = st.radio("Kolesterol Tinggi (HighChol)", [0, 1], index=0)
    chol_check = st.radio("Pemeriksaan Kolesterol (CholCheck)", [0, 1], index=0)
    bmi = st.number_input("Indeks Massa Tubuh (BMI)", min_value=0.0, max_value=500.0, value=25.0)
    smoker = st.radio("Pernah Merokok (Smoker)", [0, 1], index=0)
    stroke = st.radio("Pernah Mengalami Stroke (Stroke)", [0, 1], index=0)
    heart_disease = st.radio("Penyakit Jantung atau Serangan (HeartDiseaseorAttack)", [0, 1], index=0)
    phys_activity = st.radio("Aktivitas Fisik (PhysActivity)", [0, 1], index=1)
    fruits = st.radio("Konsumsi Buah (Fruits)", [0, 1], index=0)
    veggies = st.radio("Konsumsi Sayuran (Veggies)", [0, 1], index=0)
    alcohol = st.radio("Konsumsi Alkohol Berat (HvyAlcoholConsump)", [0, 1], index=0)
    healthcare = st.radio("Memiliki Asuransi Kesehatan (AnyHealthcare)", [0, 1], index=1)
    no_doc_cost = st.radio("Tidak Dapat Mengakses Dokter (NoDocbcCost)", [0, 1], index=0)
    gen_health = st.number_input("Kesehatan Umum (GenHlth)", min_value=0.0, max_value=500.0, value=25.0)
    ment_health = st.number_input("Kesehatan Mental (MentHlth)", min_value=0, max_value=500, value=0)
    phys_health = st.number_input("Kesehatan Fisik (PhysHlth)", min_value=0, max_value=500, value=0)
    diff_walk = st.radio("Kesulitan Berjalan (DiffWalk)", [0, 1], index=0)
    sex = st.radio("Jenis Kelamin (Sex)", [0, 1], index=0)
    age = st.slider("Usia", min_value=18, max_value=120, value=25)
    education = st.number_input("Tingkat Pendidikan (Education)", min_value=0, max_value=100, value=5)
    income = st.number_input("Pendapatan (Income)", min_value=0, max_value=100, value=50)

    # Tombol untuk memproses prediksi
    if st.button("Prediksi"):
        # Fitur input
        features = [high_bp, high_chol, chol_check, bmi, smoker, stroke, heart_disease, phys_activity,
                    fruits, veggies, alcohol, healthcare, no_doc_cost, gen_health, ment_health, phys_health,
                    diff_walk, sex, age, education, income]

        # Prediksi menggunakan model Naive Bayes
        pred_nb = predict_nb(features)

        # Menampilkan hasil prediksi
        st.subheader("Hasil Prediksi")
        st.write(f"**Model Naive Bayes:** {'Diabetes' if pred_nb == 1 else 'Tidak Diabetes'}")

        # Rekomendasi berdasarkan hasil
        if pred_nb == 1:
            st.warning("Anda berisiko terkena diabetes. Disarankan untuk segera berkonsultasi dengan dokter.")
        else:
            st.success("Anda memiliki risiko rendah terkena diabetes. Tetap jaga kesehatan!")

        # Evaluasi model pada data uji
        y_pred_test = nb_model.predict(scaler.transform(X_test))
        acc = accuracy_score(y_test, y_pred_test)
        prec = precision_score(y_test, y_pred_test)
        rec = recall_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)

        st.header("📊 Evaluasi Model")
        st.write(f"**Akurasi:** {acc:.2f}")
        st.write(f"**Precision:** {prec:.2f}")
        st.write(f"**Recall:** {rec:.2f}")
        st.write(f"**F1-Score:** {f1:.2f}")
