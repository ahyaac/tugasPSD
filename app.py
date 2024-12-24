import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Konfigurasi Halaman (Harus di awal skrip)
st.set_page_config(
    page_title="Aplikasi Diabetes",
    page_icon="🏠",
    layout="wide",
)

# Menambahkan folder yang berisi prediksi.py ke sys.path
sys.path.append('C:/ann')  # Pastikan folder ini berisi file prediksi.py
from prediksi_ann import tampilkan_ANN  # Mengimpor fungsi dari prediksi.py
from prediksi_knn import tampilkan_KNN  # Mengimpor fungsi dari prediksi.py
from prediksi_nb import tampilkan_NB  # Mengimpor fungsi dari prediksi.py

# Fungsi untuk memuat data dengan cache
@st.cache_data
def load_data():
    data = pd.read_csv('datadiabetes.csv')  # Pastikan file ada di direktori kerja
    return data

# Sidebar untuk Navigasi
with st.sidebar:
    st.title("Navigasi")
    halaman = st.radio(
        "Pilih Halaman:",
        ("Home", "ANN Penceptron", "KNN", "Naive Bayes")
    )

# Halaman Home
if halaman == "Home":
    st.title("🏠 Selamat Datang di Aplikasi Diabetes")
    st.write("""
        **Diabetes** adalah penyakit kronis yang terjadi ketika tubuh tidak dapat
        menghasilkan insulin yang cukup atau tidak dapat menggunakan insulin secara efektif. 
        Insulin adalah hormon yang mengatur kadar gula dalam darah. Berikut adalah beberapa 
        fakta tentang diabetes:
        - Diabetes dapat menyebabkan komplikasi serius seperti penyakit jantung, kebutaan, gagal ginjal, dan amputasi.
        - Ada dua jenis utama diabetes: **Tipe 1** dan **Tipe 2**.
        - Gaya hidup sehat, termasuk diet yang baik dan olahraga teratur, dapat membantu mengelola diabetes.
    """)

    # Menampilkan Dataset
    st.title("📊 Dataset Diabetes")
    st.write("""Berikut adalah dataset yang digunakan untuk menganalisis faktor-faktor yang dapat mempengaruhi risiko diabetes.""")

    # Memuat dataset
    data = load_data()

    # Menampilkan dataset
    st.dataframe(data.head())  # Menampilkan 5 baris pertama dari dataset

    # Opsi untuk menampilkan lebih banyak data
    if st.checkbox("Tampilkan Dataset Lengkap"):
        st.write(data)  # Menampilkan dataset lengkap jika checkbox dicentang

    # Menampilkan opsi untuk menampilkan statistik deskriptif
    if st.checkbox("Tampilkan Statistik Deskriptif"):
        st.write("Statistik Deskriptif Dataset:")
        st.dataframe(data.describe())  # Menampilkan informasi statistik deskriptif

    # Menggabungkan 'Pre-Diabetes' (1) dan 'Diabetes' (2) menjadi satu kategori 'Diabetes' (1.0)
    data['Diabetes_012'] = data['Diabetes_012'].replace({2: 1}).astype(float)

    # Membuat tabel distribusi berdasarkan usia dan kategori diabetes
    distribusi = pd.crosstab(data['Age'], data['Diabetes_012'], dropna=False)

    # Pastikan kolom target diisi dengan 0 dan 1 jika tidak ada data
    distribusi = distribusi.reindex(columns=[0, 1], fill_value=0)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))

    # Menentukan lebar dan offset bar
    bar_width = 0.35
    x = range(len(distribusi.index))

    # Membuat bar plot dengan offset
    ax.bar([i - bar_width / 2 for i in x], distribusi[0], bar_width, color='blue', label='Non-Diabetes', alpha=0.7)
    ax.bar([i + bar_width / 2 for i in x], distribusi[1], bar_width, color='orange', label='Diabetes', alpha=0.7)

    # Menambahkan judul, label, dan legenda
    ax.set_title("Distribusi Data Usia terhadap Target Diabetes", fontsize=16)
    ax.set_xlabel("Usia", fontsize=14)
    ax.set_ylabel("Jumlah", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(distribusi.index, rotation=45, fontsize=12)
    ax.legend(title="Klasifikasi Diabetes", fontsize=12)

    # Menampilkan grafik di Streamlit
    st.pyplot(fig)

    # Membuat data untuk diagram lingkaran
    diabetes_counts = data['Diabetes_012'].value_counts().reindex([0, 1], fill_value=0)
    labels = ['Non-Diabetes', 'Diabetes']
    colors = ['blue', 'orange']
    sizes = diabetes_counts.values

    # Membuat diagram lingkaran
    fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
    ax_pie.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',  # Menampilkan persentase
        startangle=90,  # Memutar posisi awal
        wedgeprops={'edgecolor': 'black'}  # Memberikan garis tepi pada slice
    )
    ax_pie.set_title("Distribusi Persentase Diabetes vs Non-Diabetes", fontsize=16)

    # Menampilkan diagram lingkaran di Streamlit
    st.pyplot(fig_pie)

    # Subheader untuk Preprocessing
    st.subheader("Preprocessing")

    # Menambahkan nomor urut dan penjelasan
    st.markdown("### 1. Mendeteksi Missing Value")

    # Menghitung jumlah missing value di setiap kolom
    missing_values = data.isnull().sum()

    # Membuat DataFrame untuk ditampilkan
    missing_table = pd.DataFrame({
        'Kolom': missing_values.index,
        'Jumlah Missing Value': missing_values.values
    })

    # Menampilkan tabel lengkap dengan nama kolom
    st.table(missing_table.set_index('Kolom'))

    # Memisahkan fitur (X) dan target (y)
    X = data.drop(columns=['Diabetes_012'])  # Hanya kolom fitur
    y = data['Diabetes_012']  # Hanya kolom target

    # Inisialisasi MinMaxScaler
    scaler = MinMaxScaler()

    # Normalisasi hanya pada kolom fitur (X)
    X_normalized = scaler.fit_transform(X)

    # Konversi kembali hasil normalisasi menjadi DataFrame
    X_normalized_df = pd.DataFrame(X_normalized, columns=X.columns)

    # Menampilkan hasil normalisasi
    st.subheader("### 2. Hasil Normalisasi Fitur")
    st.dataframe(X_normalized_df.head())  
    if st.checkbox("Tampilkan Semua Data yang Dinormalisasi"):
        st.dataframe(X_normalized_df)  

    st.subheader("### 3. Pembagian Dataset")
    # Membagi dataset menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Konversi data latih dan uji ke DataFrame agar lebih mudah dibaca
    X_train_df = pd.DataFrame(X_train, columns=X.columns)
    X_test_df = pd.DataFrame(X_test, columns=X.columns)

    # Menampilkan data latih
    st.markdown("**Data Latih (Training Data)**")
    st.write(f"Jumlah data latih: {X_train_df.shape[0]} baris, {X_train_df.shape[1]} kolom")
    st.dataframe(X_train_df)

    # Menampilkan data uji
    st.markdown("**Data Uji (Testing Data)**")
    st.write(f"Jumlah data uji: {X_test_df.shape[0]} baris, {X_test_df.shape[1]} kolom")
    st.dataframe(X_test_df)

    st.subheader("Modeling")
    
    # Model Perceptron, KNN, dan Naive Bayes
    models = {
        'Perceptron': Perceptron(),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB()
    }

    results = {}

    for model_name, model in models.items():
        st.subheader(f"### Evaluasi Model {model_name}")
        
        # Melatih model
        model.fit(X_train, y_train)
        
        # Prediksi
        y_pred = model.predict(X_test)
        
        # Menghitung evaluasi
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # Menampilkan hasil
        st.write(f"**Akurasi**: {accuracy:.2f}")
        st.write(f"**Precision**: {precision:.2f}")
        st.write(f"**Recall**: {recall:.2f}")
        st.write(f"**F1-Score**: {f1:.2f}")
        
        # Menampilkan confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Diabetes', 'Diabetes'])
        disp.plot(cmap=plt.cm.Blues)
        st.pyplot(plt.gcf())
        
        # Menyimpan hasil evaluasi
        results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # Menampilkan perbandingan model
    st.subheader("Perbandingan Model")
    results_df = pd.DataFrame(results).T
    st.dataframe(results_df)
    
    # Evaluasi Perceptron dengan epoch 4
    model_perceptron_4 = Perceptron(max_iter=4, random_state=42)
    model_perceptron_4.fit(X_train, y_train)
    y_pred_4 = model_perceptron_4.predict(X_test)

    accuracy_4 = accuracy_score(y_test, y_pred_4)
    precision_4 = precision_score(y_test, y_pred_4)
    recall_4 = recall_score(y_test, y_pred_4)
    f1_4 = f1_score(y_test, y_pred_4)
    cm_4 = confusion_matrix(y_test, y_pred_4)

    st.subheader("### Evaluasi Model Perceptron (Epoch 4)")
    st.write(f"**Akurasi**: {accuracy_4:.2f}")
    st.write(f"**Precision**: {precision_4:.2f}")
    st.write(f"**Recall**: {recall_4:.2f}")
    st.write(f"**F1-Score**: {f1_4:.2f}")

    disp_4 = ConfusionMatrixDisplay(confusion_matrix=cm_4, display_labels=['Non-Diabetes', 'Diabetes'])
    disp_4.plot(cmap=plt.cm.Blues)
    st.pyplot(plt.gcf())

    # Evaluasi Perceptron dengan epoch 8
    model_perceptron_8 = Perceptron(max_iter=8, random_state=42)
    model_perceptron_8.fit(X_train, y_train)
    y_pred_8 = model_perceptron_8.predict(X_test)

    accuracy_8 = accuracy_score(y_test, y_pred_8)
    precision_8 = precision_score(y_test, y_pred_8)
    recall_8 = recall_score(y_test, y_pred_8)
    f1_8 = f1_score(y_test, y_pred_8)
    cm_8 = confusion_matrix(y_test, y_pred_8)

    st.subheader("### Evaluasi Model Perceptron (Epoch 8)")
    st.write(f"**Akurasi**: {accuracy_8:.2f}")
    st.write(f"**Precision**: {precision_8:.2f}")
    st.write(f"**Recall**: {recall_8:.2f}")
    st.write(f"**F1-Score**: {f1_8:.2f}")

    disp_8 = ConfusionMatrixDisplay(confusion_matrix=cm_8, display_labels=['Non-Diabetes', 'Diabetes'])
    disp_8.plot(cmap=plt.cm.Blues)
    st.pyplot(plt.gcf())

    # Menyimpan hasil perbandingan Perceptron dengan epoch 4 dan 8
    results_perceptron = {
        'epoch 4': {'accuracy': accuracy_4, 'precision': precision_4, 'recall': recall_4, 'f1': f1_4},
        'epoch 8': {'accuracy': accuracy_8, 'precision': precision_8, 'recall': recall_8, 'f1': f1_8}
    }
    
    # Menampilkan perbandingan Perceptron epoch 4 dan 8
    st.subheader("Perbandingan Model Perceptron Epoch 4 dan 8")
    results_perceptron_df = pd.DataFrame(results_perceptron).T
    st.dataframe(results_perceptron_df)

elif halaman == "ANN Penceptron":
    tampilkan_ANN()  # Memanggil fungsi dari prediksi.py
elif halaman == "KNN":
    tampilkan_KNN()  # Memanggil fungsi dari prediksi.py
elif halaman == "Naive Bayes":
    tampilkan_NB()  # Memanggil fungsi dari prediksi.py
