import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import joblib

df = pd.read_csv('dataset.csv')

selected_features = [
    'Tuition fees up to date',             # Status pembayaran
    'Scholarship holder',                  # Penerima Beasiswa
    'Curricular units 1st sem (approved)', # SKS Lulus Sem 1
    'Curricular units 1st sem (grade)',    # Nilai Rata-rata Sem 1
    'Curricular units 2nd sem (approved)', # SKS Lulus Sem 2
    'Curricular units 2nd sem (grade)',    # Nilai Rata-rata Sem 2
    'Age at enrollment'                    # Umur saat mendaftar
]

X = df[selected_features]
y = df['Target']

le = LabelEncoder()
y = le.fit_transform(y)

print("Mapping Target:", dict(zip(le.classes_, le.transform(le.classes_))))

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Pipeline (StandardScaler + SVM)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
])

# 6. train model
pipeline.fit(X_train, y_train)


joblib.dump(pipeline, 'svm_model.pkl')
print("Model disimpan sebagai 'svm_model.pkl'")
