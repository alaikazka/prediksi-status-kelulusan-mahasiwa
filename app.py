import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Prediksi Status Kelulusan Mahasiswa", layout="centered")

st.title("Prediksi Kelulusan Status Mahasiswa")
st.write("Aplikasi ini menggunakan algoritma **Support Vector Machine (SVM)** untuk memprediksi apakah mahasiswa akan Lulus, Dropout, atau Masih Aktif.")

@st.cache_resource
def load_model():
    try:
        model = joblib.load('svm_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

target_mapping = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
warna_hasil = {'Dropout': 'red', 'Enrolled': 'orange', 'Graduate': 'green'}

# INPUT
st.subheader("Masukkan Data Mahasiswa")

col1, col2 = st.columns(2)

with col1:
    tuition_fees = st.selectbox("Status Pembayaran SPP", [1, 0], format_func=lambda x: "Lancar" if x == 1 else "Menunggak")
    scholarship = st.selectbox("Penerima Beasiswa", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    age = st.number_input("Umur saat Mendaftar", min_value=15, max_value=100, value=20)

with col2:
    sem1_approved = st.number_input("SKS Lulus (Semester 1)", min_value=0, max_value=30, value=5)
    sem1_grade = st.number_input("Nilai Rata-rata (Semester 1)", min_value=0.0, max_value=20.0, value=12.0)
    sem2_approved = st.number_input("SKS Lulus (Semester 2)", min_value=0, max_value=30, value=5)
    sem2_grade = st.number_input("Nilai Rata-rata (Semester 2)", min_value=0.0, max_value=20.0, value=12.0)

if st.button("Prediksi Status"):
    if model:
        input_data = pd.DataFrame({
            'Tuition fees up to date': [tuition_fees],
            'Scholarship holder': [scholarship],
            'Curricular units 1st sem (approved)': [sem1_approved],
            'Curricular units 1st sem (grade)': [sem1_grade],
            'Curricular units 2nd sem (approved)': [sem2_approved],
            'Curricular units 2nd sem (grade)': [sem2_grade],
            'Age at enrollment': [age]
        })

        # Prediksi
        prediction_index = model.predict(input_data)[0]
        prediction_label = target_mapping.get(prediction_index, "Unknown")
        probability = model.predict_proba(input_data).max()

        # Hasil
        st.markdown("---")
        st.subheader("Hasil Prediksi:")
        
        color = warna_hasil.get(prediction_label, 'blue')
        st.markdown(f"<h2 style='text-align: center; color: {color};'>{prediction_label}</h2>", unsafe_allow_html=True)
        st.write(f"Tingkat Keyakinan Model: **{probability*100:.2f}%**")
        
        # Saran
        if prediction_label == 'Dropout':
            st.warning("Mahasiswa ini berisiko tinggi putus kuliah. Disarankan untuk memberikan bimbingan konseling segera.")
        elif prediction_label == 'Enrolled':
            st.info("Mahasiswa ini masih aktif namun perlu dipantau perkembangannya.")
        else:
            st.success("Mahasiswa ini memiliki performa yang baik dan diprediksi akan lulus.")
    else:
        st.error("Model belum dimuat. Pastikan file 'svm_model.pkl' ada di repository.")
