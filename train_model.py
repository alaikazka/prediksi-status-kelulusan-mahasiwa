import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import joblib

# 1. Load Data
# Pastikan file dataset.csv ada di folder yang sama
df = pd.read_csv('dataset.csv')

# 2. Seleksi Fitur Utama (Agar input di web tidak terlalu banyak)
# Kita ambil fitur yang paling berkorelasi dengan kelulusan
selected_features = [
    'Tuition fees up to date',             # Status pembayaran SPP
    'Scholarship holder',                  # Penerima Beasiswa
    'Curricular units 1st sem (approved)', # SKS Lulus Sem 1
    'Curricular units 1st sem (grade)',    # Nilai Rata-rata Sem 1
    'Curricular units 2nd sem (approved)', # SKS Lulus Sem 2
    'Curricular units 2nd sem (grade)',    # Nilai Rata-rata Sem 2
    'Age at enrollment'                    # Umur saat mendaftar
]

X = df[selected_features]
y = df['Target']

# 3. Encoding Target (Dropout, Enrolled, Graduate -> 0, 1, 2)
# Ini penting agar prediksi bisa diterjemahkan kembali nanti
le = LabelEncoder()
y = le.fit_transform(y)

# Simpan urutan kelas untuk dipakai di app.py
# Biasanya: 0=Dropout, 1=Enrolled, 2=Graduate (Cek output print di bawah)
print("Mapping Target:", dict(zip(le.classes_, le.transform(le.classes_))))

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Membuat Pipeline (StandardScaler + SVM)
# SVM SANGAT butuh scaling (StandardScaler) agar akurasinya bagus
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
])

# 6. Latih Model
print("Sedang melatih model SVM...")
pipeline.fit(X_train, y_train)
print("Pelatihan selesai!")

# 7. Simpan Model
joblib.dump(pipeline, 'svm_model.pkl')
print("Model berhasil disimpan sebagai 'svm_model.pkl'")
