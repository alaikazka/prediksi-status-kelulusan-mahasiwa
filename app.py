import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Konfigurasi Halaman
st.set_page_config(
    page_title="Prediksi Status Kelulusan Mahasiswa",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Judul dan Deskripsi
st.title("🎓 Prediksi Status Mahasiswa")
st.markdown("""
Aplikasi ini menggunakan algoritma **Support Vector Machine (SVM)** untuk memprediksi apakah mahasiswa akan **Lulus (Graduate)** atau **Dropout**.
""")

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        # Mencoba memuat model
        if os.path.exists('svm_model.pkl'):
            model = joblib.load('svm_model.pkl')
            return model
        else:
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- MAPPING TARGET (PENTING) ---
# Berdasarkan urutan abjad (Dropout, Graduate):
# 0 = Dropout
# 1 = Graduate
# Sesuaikan jika output di notebook Anda berbeda.
target_mapping = {0: 'Dropout', 1: 'Graduate'}
warna_hasil = {'Dropout': '#ff4b4b', 'Graduate': '#21c354'} # Merah untuk Dropout, Hijau untuk Graduate

if not model:
    st.warning("⚠️ File 'svm_model.pkl' tidak ditemukan. Silakan upload file model hasil training ulang (retrain).")

# --- INPUT USER ---
st.markdown("---")
st.subheader("📝 Masukkan Data Mahasiswa")

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Data Keuangan & Pribadi**")
        tuition_fees = st.selectbox(
            "Status Pembayaran SPP", 
            [1, 0], 
            format_func=lambda x: "Lancar (Up to date)" if x == 1 else "Menunggak"
        )
        scholarship = st.selectbox(
            "Penerima Beasiswa", 
            [1, 0], 
            format_func=lambda x: "Ya" if x == 1 else "Tidak"
        )
        # Range Umur: 17 - 70
        age = st.number_input(
            "Umur saat Mendaftar", 
            min_value=17, 
            max_value=70, 
            value=19,
            help="Rentang usia valid: 17 - 70 tahun"
        )

    with col2:
        st.markdown("**Data Akademik**")
        # Range SKS: 0 - 17
        sem1_approved = st.number_input(
            "SKS Lulus (Semester 1)", 
            min_value=0, 
            max_value=17, 
            value=5,
            help="Jumlah SKS yang berhasil lulus di semester 1 (Maks: 17)"
        )
        sem1_grade = st.number_input(
            "Nilai Rata-rata (Semester 1)", 
            min_value=0.0, 
            max_value=20.0, 
            value=12.0,
            step=0.1
        )
        # Range SKS: 0 - 17
        sem2_approved = st.number_input(
            "SKS Lulus (Semester 2)", 
            min_value=0, 
            max_value=17, 
            value=5,
            help="Jumlah SKS yang berhasil lulus di semester 2 (Maks: 17)"
        )
        sem2_grade = st.number_input(
            "Nilai Rata-rata (Semester 2)", 
            min_value=0.0, 
            max_value=20.0, 
            value=12.0,
            step=0.1
        )

# --- TOMBOL PREDIKSI ---
if st.button("🔍 Prediksi Status", type="primary"):
    if model:
        # Membuat DataFrame sesuai input 7 fitur
        input_data = pd.DataFrame({
            'Tuition fees up to date': [tuition_fees],
            'Scholarship holder': [scholarship],
            'Curricular units 1st sem (approved)': [sem1_approved],
            'Curricular units 1st sem (grade)': [sem1_grade],
            'Curricular units 2nd sem (approved)': [sem2_approved],
            'Curricular units 2nd sem (grade)': [sem2_grade],
            'Age at enrollment': [age]
        })

        try:
            # Melakukan prediksi
            prediction_index = model.predict(input_data)[0]
            prediction_label = target_mapping.get(prediction_index, "Unknown")
            
            # Mendapatkan probabilitas (Keyakinan Model)
            try:
                probs = model.predict_proba(input_data)
                # probs[0] adalah array probabilitas untuk setiap kelas [prob_dropout, prob_graduate]
                probability = probs.max()
                confidence_text = f"Tingkat Keyakinan Model: **{probability*100:.2f}%**"
            except:
                probability = None
                confidence_text = ""

            # --- TAMPILAN HASIL ---
            st.markdown("---")
            st.subheader("Hasil Analisis")
            
            result_col1, result_col2 = st.columns([1, 2])
            
            with result_col1:
                color = warna_hasil.get(prediction_label, 'grey')
                st.markdown(
                    f"""
                    <div style="
                        background-color: {color};
                        padding: 20px;
                        border-radius: 10px;
                        text-align: center;
                        color: white;
                        font-weight: bold;
                        font-size: 24px;
                        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                        {prediction_label}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            with result_col2:
                if confidence_text:
                    st.write(confidence_text)
                
                # Logika Saran (Hanya Dropout dan Graduate)
                if prediction_label == 'Dropout':
                    st.error("⚠️ **Peringatan:** Mahasiswa ini terdeteksi memiliki risiko tinggi untuk Dropout. Disarankan untuk segera melakukan intervensi akademik atau konseling.")
                elif prediction_label == 'Graduate':
                    st.success("✅ **Prospek Positif:** Mahasiswa ini diprediksi akan Lulus. Pertahankan performa akademik saat ini.")
                else:
                    st.write("Hasil prediksi tidak dapat dikategorikan.")
        
        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi. Detail: {e}")
    else:
        st.error("Model belum dimuat. Pastikan Anda telah melakukan training ulang di notebook dan menyimpan file 'svm_model.pkl'.")

# Footer
st.markdown("---")
st.caption("Sistem Prediksi Kelulusan Mahasiswa (Support Vector Machine)")
