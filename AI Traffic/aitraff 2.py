import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

# Load data
df = pd.read_csv('simulasi_kemacetan_numerik 1.csv', delimiter=';')

# Misalnya kita asumsikan kolom 'Rata-rata kecepatan' adalah target
X = df.drop(columns=['Unnamed: 0', 'Rata-rata kecepatan'])
y = df['Rata-rata kecepatan']

# Bagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat model Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Evaluasi dengan data testing
y_pred = model.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred))
print("Laporan Klasifikasi:\n", classification_report(y_test, y_pred))

# Cross-validation (5-fold)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("Akurasi tiap lipatan cross-validation:", cv_scores)
print("Akurasi rata-rata cross-validation:", cv_scores.mean())

# Prediksi probabilitas untuk data baru
contoh_kondisi = [[3, 5, 2, 3, 1, 1, 1, 1, 0, 1, 1, 0, 3, 2]]  # Sesuai urutan kolom
prob = model.predict_proba(contoh_kondisi)
print("Probabilitas Kemacetan (Cepat, Sedang, Lambat):", prob)
