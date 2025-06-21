import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

# Load data
df = pd.read_csv('simulasi_kemacetan_numerik 1.csv', delimiter=';')

# Misalnya kita asumsikan kolom 'Rata-rata kecepatan' adalah target klasifikasi kemacetan
# 1 = Cepat (tidak macet), 2 = Sedang, 3 = Lambat (macet)
X = df.drop(columns=['Unnamed: 0', 'Rata-rata kecepatan'])
y = df['Rata-rata kecepatan']

# Bagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat model Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Prediksi dan evaluasi
y_pred = model.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred))
print("Laporan Klasifikasi:\n", classification_report(y_test, y_pred))

# Contoh prediksi probabilitas dengan kondisi baru
# Format: [Waktu, Hari, Cuaca, Volume kendaraan, Kendaraan besar, Lampu lalu lintas, Kecelakaan, ...]
contoh_kondisi = [[3, 1, 2, 3, 1, 0, 1, 0, 1, 1, 0, 1, 3, 2]]  # Isi sesuai urutan kolom input
prob = model.predict_proba(contoh_kondisi)
print("Probabilitas Kemacetan (Cepat, Sedang, Lambat):", prob)

