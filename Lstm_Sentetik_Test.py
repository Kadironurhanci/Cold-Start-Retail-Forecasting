import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

print("SENTETİK VERİ LSTM TESTİ")

# 1. Sentetik Veriyi Oku
df_syn = pd.read_csv('Akilli_Sentetik_Hobbies.csv', sep=';')

# İlk ürünü seçiyoruz (Örn: HOBBIES_1_001_CA_1_validation_SYN)
urun_id = df_syn.iloc[0]['item_id']
print(f" Test Edilen Sentetik Ürün: {urun_id}")

# Sadece d_1'den d_1913'e kadar olan verileri al
satislar = df_syn.iloc[0, 1:].values.astype(float)

# 2. Veri Ön İşleme
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(satislar.reshape(-1, 1))

# Pencereleme (Son 28 güne bak)
def create_sequences(data, seq_length=28):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 28
X, y = create_sequences(scaled_data, seq_length)

# %80 Eğitim, %20 Test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 3. Model Kurulumu ve Erken Durdurma
lstm_model = Sequential([
    Input(shape=(seq_length, 1)),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')

erken_durma = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 4. Eğitim Başlatılmsı
print("\n Eğitim başlatılıyor...")
history = lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[erken_durma], verbose=1)

# 5. Grafik Çizimi
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Eğitim Kaybı (Train Loss)')
plt.plot(history.history['val_loss'], label='Test Kaybı (Validation Loss)')
plt.title(f'LSTM Testi - {urun_id}')
plt.xlabel('Epochs')
plt.ylabel('Hata (MSE)')
plt.legend()
plt.grid(True)
plt.show()

print(" Test Tamamlandı!")