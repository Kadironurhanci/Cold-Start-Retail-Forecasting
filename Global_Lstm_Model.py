import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. Klasör ve Masaüstü Hazırlığı
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "LSTM_Sonuclari")
if not os.path.exists(desktop_path):
    os.makedirs(desktop_path)

print("--- 🚀 GLOBAL LSTM EĞİTİMİ BAŞLIYOR ---")

# 2. Veri Yükleme (Noktalı olan teknik dosyayı kullanıyoruz!)
print("Devasa veri seti yükleniyor...")
df = pd.read_csv('Sentetik_Hobbies_Final_Dataset.csv', sep=';')

# 3. Veri Ön İşleme (Preprocessing)
# Sadece sayısal sütunlarla çalışacağız
data = df[['sales', 'sell_price']].values

# Normalizasyon (LSTM 0-1 arası veriyi çok sever)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Eğitim ve Test Ayırımı (%80 Eğitim, %20 Test)
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# 4. Pencereleme (Look-back) Fonksiyonu
# Son 28 güne bakıp 29. günü tahmin etmeyi öğrenecek (M5 standardı)
def create_sequences(data, window_size=28):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size, :]) # Tüm özellikleri al (sales + price)
        y.append(data[i+window_size, 0])    # Sadece satış adedini tahmin et
    return np.array(X), np.array(y)

window = 28
print(f"Sekanslar oluşturuluyor (Pencere boyutu: {window})...")
X_train, y_train = create_sequences(train_data, window)
X_test, y_test = create_sequences(test_data, window)

# 5. LSTM Mimarisini İnşa Etme
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2), # Overfitting'i engellemek için %20 nöron öldürme
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1) # Tahmin sonucu (satış adedi)
])

model.compile(optimizer='adam', loss='mse')

# 6. Model Eğitimi (Epochs)
print("Motorlar çalıştırılıyor... Eğitim başladı.")
# EarlyStopping: Eğer model öğrenmeyi bırakırsa eğitimi vaktinde keser (Overfitting koruması)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=20, # Şimdilik 20 yeterli, duruma göre artırırız
    batch_size=1024, # Global model olduğu için büyük batch hızı artırır
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# 7. 📈 GRAFİK ŞÖLENİ VE KAYDETME


# GRAFİK 1: Öğrenme Eğrisi (Loss Curve)
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Eğitim Kaybı (Train Loss)', color='blue')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı (Val Loss)', color='red')
plt.title('LSTM Öğrenme Eğrisi (Overfitting Kontrolü)')
plt.xlabel('Epoch Sayısı')
plt.ylabel('Hata (MSE)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(desktop_path, "1_Ogrenme_Egrisi.png"))
plt.close()

# GRAFİK 2: Tahmin vs Gerçek (Son 200 gün örneği)
predictions = model.predict(X_test)
plt.figure(figsize=(15, 7))
plt.plot(y_test[-200:], label='Gerçek Sentetik Satış', color='green', alpha=0.7)
plt.plot(predictions[-200:], label='LSTM Tahmini', color='orange', linestyle='--')
plt.title('Son 200 Gün: Gerçek Değerler vs LSTM Tahminleri')
plt.xlabel('Zaman')
plt.ylabel('Normalize Satış')
plt.legend()
plt.savefig(os.path.join(desktop_path, "2_Tahmin_Karsilastirma.png"))
plt.close()

print(f"\n✅ EĞİTİM TAMAMLANDI!")
