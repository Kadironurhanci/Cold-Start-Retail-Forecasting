import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

print("--- 🚀 GLOBAL LSTM MASTER PLAN BAŞLIYOR ---")

# =====================================================================
# 🎛️ KONTROL PANELİ (HİPERPARAMETRELER VE VERİ AYARLARI)
# =====================================================================
EPOCH_SAYISI = 25          # Modelin veriyi kaç kere baştan sona okuyacağı
BATCH_SIZE = 512           # Tek seferde hafızaya alınacak veri boyutu
WINDOW_SIZE = 28           # Kaç geçmiş güne bakıp tahmine karar verilecek?
TEST_GUN_SAYISI = 30       # Son kaç günü test (görünmez) olarak ayıralım?

# Örneklem (Sampling) Ayarları
HOBBIES_1_ADET = 50        # Hobbies 1'den kaç ürün seçilecek?
HOBBIES_2_ADET = 50        # Hobbies 2'den kaç ürün seçilecek?
# Toplam 100 ürünle (veya kendi gireceğin sayıyla) mikro evren yaratıyoruz
# =====================================================================

# 1. Klasör Hazırlığı
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "LSTM_Master_Sonuclari")
if not os.path.exists(desktop_path):
    os.makedirs(desktop_path)

# 2. Veri Yükleme ve Örneklem Seçimi (Sampling)
print(f"Veri yükleniyor ve H1'den {HOBBIES_1_ADET}, H2'den {HOBBIES_2_ADET} ürün seçiliyor...")
df = pd.read_csv('Sentetik_Hobbies_Final_Dataset.csv', sep=';')

# Ürünleri filtrele
h1_urunleri = df[df['item_id'].str.contains('HOBBIES_1')]['item_id'].unique()[:HOBBIES_1_ADET]
h2_urunleri = df[df['item_id'].str.contains('HOBBIES_2')]['item_id'].unique()[:HOBBIES_2_ADET]

secilen_urunler = list(h1_urunleri) + list(h2_urunleri)
df_secili = df[df['item_id'].isin(secilen_urunler)]

print(f"Seçilen toplam ürün sayısı: {len(secilen_urunler)}")

# 3. Veriyi Modele Hazırlama (Ölçekleme)
data_sales = df_secili[['sales']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_sales = scaler.fit_transform(data_sales)

# Eğitim ve Test olarak bölme (Son 30 gün teste gidecek şekilde ayarlanır)
train_size = len(scaled_sales) - (len(secilen_urunler) * TEST_GUN_SAYISI)
train_data = scaled_sales[:train_size]
test_data = scaled_sales[train_size:]

# Zaman Serisi Sekansları Oluşturma
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data, WINDOW_SIZE)
X_test, y_test = create_sequences(test_data, WINDOW_SIZE)

# 4. LSTM Mimarisi
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# 5. Eğitim (Epochs) ve Erken Durdurma
early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

print(f"Model {EPOCH_SAYISI} epoch üzerinden eğitime başlıyor...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCH_SAYISI,
    batch_size=BATCH_SIZE,
    validation_split=0.1, # Eğitim verisinin %10'unu anlık test için kullan
    callbacks=[early_stop],
    verbose=1
)

# 6. TAHMİN (PREDICTION) VE GERİ DÖNÜŞTÜRME
predictions_scaled = model.predict(X_test)
# Ölçeklenmiş (0-1 arası) veriyi tekrar gerçek satış adetlerine (1, 2, 5 gibi) çeviriyoruz
predictions = scaler.inverse_transform(predictions_scaled)
actuals = scaler.inverse_transform(y_test)

# 7. ORTALAMA ALMA (MAKRO GRAFİK İÇİN)
# 100 ürünün test günleri boyunca yaptığı satışların genel ortalamasını alıyoruz
# Bu, "Kategorinin genel trendini doğru tahmin edebildik mi?" sorusunun cevabıdır.
makro_tahminler = predictions.mean()
makro_gercekler = actuals.mean()

# 8. 💾 PROTOBUF (PB) FORMATINDA KAYDETME (Production Ready)
pb_kayit_yolu = os.path.join(desktop_path, "LSTM_SavedModel_Protobuf")
model.export(pb_kayit_yolu)
print(f"\n📦 EFSANE! Modelin mimarisi ve ağırlıkları PROTOBUF (.pb) formatında şuraya kaydedildi: {pb_kayit_yolu}")

# B) Python İçi Analiz Formatı (.keras)
keras_kayit_yolu = os.path.join(desktop_path, "LSTM_Model.keras")
model.save(keras_kayit_yolu)
print(f"🐍 Model Python uyumlu (.keras) formatında kaydedildi.")

# 9. 📈 GRAFİK ŞÖLENİ
print("📊 Grafikler üretiliyor...")

# Grafik 1: Öğrenme Eğrisi (Epoch Performansı)
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Eğitim Kaybı (Train Loss)')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı (Val Loss)')
plt.title(f'LSTM Öğrenme Eğrisi ({EPOCH_SAYISI} Epoch)')
plt.xlabel('Epoch')
plt.ylabel('Hata (MSE)')
plt.legend()
plt.savefig(os.path.join(desktop_path, "1_Epoch_Ogrenme_Egrisi.png"))
plt.close()

# Grafik 2: Makro Bakış (Agrega Edilmiş Ortalamalar)
plt.figure(figsize=(12, 6))
plt.plot(actuals[:200], label='Gerçek Satışlar', color='blue', alpha=0.6)
plt.plot(predictions[:200], label='LSTM Tahminleri', color='orange', linestyle='dashed')
plt.title(f'Seçilen {len(secilen_urunler)} Ürünün Makro Tahmin Grafiği (Son 200 Kayıt)')
plt.xlabel('Zaman Çizelgesi')
plt.ylabel('Satış Adedi')
plt.legend()
plt.savefig(os.path.join(desktop_path, "2_Makro_Ortalama_Tahmin.png"))
plt.close()

print(f"\n✅ TEBRİKLER HOCAM! Masaüstündeki 'LSTM_Master_Sonuclari' klasörüne bakabilirsin.")
print("Tezin kalbi başarıyla kodlandı ve atıyor! 🚀")