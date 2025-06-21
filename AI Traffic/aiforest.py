import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load data
df = pd.read_csv('simulasi_kemacetan_numerik 1.csv', delimiter=';')

# Pisahkan fitur dan target
X = df.drop(columns=['Unnamed: 0', 'Rata-rata kecepatan'])
y = df['Rata-rata kecepatan']

# Bagi data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Buat model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred))
print("Laporan Klasifikasi:\n", classification_report(y_test, y_pred))

# Validasi silang
scores = cross_val_score(model, X, y, cv=5)
print("Akurasi rata-rata cross-validation:", scores.mean())

# Prediksi probabilitas dari kondisi baru
contoh = [[3, 1, 2, 3, 1, 0, 1, 0, 1, 1, 0, 1, 3, 2]]
prob = model.predict_proba(contoh)
print("Probabilitas Kemacetan (Cepat, Sedang, Lambat):", prob)
