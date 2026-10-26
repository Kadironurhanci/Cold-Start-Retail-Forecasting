import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

print("--- 🚀 SİSTEM KONTROLÜ ---")
print(f"TensorFlow Sürümü: {tf.__version__}")
print("İşlemci (CPU) üzerinden eğitim başlatılıyor...\n")

# 1. VERİYİ YÜKLEME VE FİLTRELEME
print("Veri seti yükleniyor... (Bu işlem dosya boyutuna göre 10-15 saniye sürebilir)")
df = pd.read_csv('sales_train_validation.csv')

#CA_1 mağazasındaki HOBBIES_1_002 ürününü seçiyoruz
hedef_urun_id = 'HOBBIES_1_002_CA_1_validation'
urun_verisi = df[df['id'] == hedef_urun_id]

if urun_verisi.empty:
    print(f"Hata: {hedef_urun_id} bulunamadı! Lütfen dosya adını kontrol et.")
    exit()


gun_sutunlari = [col for col in df.columns if col.startswith('d_')]
gercek_satislar = urun_verisi[gun_sutunlari].values[0].astype(float)

print(f"✅ Veri çekildi! Toplam gün sayısı: {len(gercek_satislar)}")

#veri ön işleme
#scalerizasyon
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(gercek_satislar.reshape(-1, 1))

#ZAMAN SERİSİ PENCERELEME
#Modele "Son 28 güne bakarak 29. günü tahmin et" diyeceğiz
def create_sequences(data, seq_length=28):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 28
X, y = create_sequences(scaled_data, seq_length)

#%80 Eğitim, %20 Test olarak veriyi ayırıyoruz
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Eğitim Verisi Boyutu: {X_train.shape} | Test Verisi Boyutu: {X_test.shape}\n")

#MODEL MİMARİSİNİ GÜÇLENDİRME
#Keras 3 Standartlarına Uygun Model
lstm_model = Sequential([
    Input(shape=(seq_length, 1)), # Uyarıyı kaldıran yeni sistem bu
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')

#earlystopping sensörü
#patience sabır demek tamam mı patience miktarı kadar iyileşmezse duruyor best weight aralarından en iyisini seçiyor vay aq
erken_durma = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)


lstm_history = lstm_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[erken_durma], # Sensörü devreye soktuk
    verbose=1
)
print("\nİLK EĞİTİM BAŞARIYLA TAMAMLANDI!")