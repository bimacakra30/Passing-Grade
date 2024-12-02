import pandas as pd
from sklearn.linear_model import LinearRegression  # Import Linear Regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle

# Membaca dataset
try:
    df = pd.read_csv('passing-grade.csv')
    df = df.dropna(subset=['RATAAN', 'S.BAKU', 'MIN'])
except FileNotFoundError:
    print("File 'passing-grade.csv' tidak ditemukan. Pastikan file tersebut ada.")
    exit()

# Membagi data menjadi fitur dan target
X = df[['RATAAN', 'S.BAKU']]
y = df['MIN']

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model menggunakan Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Mengevaluasi model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Menyimpan model ke dalam file
with open('passing_grade_model.sav', 'wb') as file:
    pickle.dump(model, file)
    print("Model berhasil disimpan sebagai 'passing_grade_model.sav'.")
