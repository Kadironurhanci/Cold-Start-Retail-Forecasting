import pandas as pd
import numpy as np
import os
import pickle
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# 🛑 ADIM 1: REPRODUCIBILITY (TEKRARLANABİLİRLİK)
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except:
    print("⚠️ Deteminizm modu bu TF sürümünde desteklenmiyor, normal devam ediliyor.")

print("\n" + "=" * 50)
print("--- 🛡️ GRU ACADEMIC GOLD STANDARD MOTORU BAŞLATILDI ---")
print("=" * 50)

# 2. Yollar
proje_klasoru = os.path.dirname(os.path.abspath(__file__))
veri_klasoru = os.path.join(proje_klasoru, "Tez_Veri_Setleri")
model_klasoru = os.path.join(proje_klasoru, "Tez_Modelleri")

if not os.path.exists(model_klasoru):
    os.makedirs(model_klasoru)

egitim_dosyalari = {
    'Baseline_0': '2_Train_Baseline_0_Sentetik.csv',
    'Mix_25': '3_Train_Mix_25.csv',
    'Mix_50': '4_Train_Mix_50.csv',
    'Mix_100': '5_Train_Mix_100.csv'
}

ZAMAN_PENCERESI = 30

# --- EĞİTİM ANA DÖNGÜSÜ ---
for model_adi, dosya_adi in egitim_dosyalari.items():
    dosya_yolu = os.path.join(veri_klasoru, dosya_adi)

    if not os.path.exists(dosya_yolu):
        print(f"\n❌ DOSYA EKSİK: {dosya_adi} bulunamadı, atlanıyor...")
        continue

    print(f"\n📂 {model_adi} İşleniyor: {dosya_adi}")
    df = pd.read_csv(dosya_yolu, sep=';', low_memory=False)

    urunler = df['item_id'].unique()
    train_only_sales_pool = []

    # 🛑 ADIM 2: TAM HİJYENİK SCALER (SADECE TRAIN KISMINA FİT)
    for urun in tqdm(urunler, desc="Scaler Havuzu Hazırlanıyor"):
        urun_sales = df[df['item_id'] == urun]['sales'].values.reshape(-1, 1)
        split_idx = int(len(urun_sales) * 0.8)
        train_only_sales_pool.append(urun_sales[:split_idx])

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(np.vstack(train_only_sales_pool))

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []

    # 🛑 ADIM 3: ÜRÜN BAZLI TEMPORAL SPLIT
    for urun in tqdm(urunler, desc=f"{model_adi} Sequenceler"):
        urun_sales = df[df['item_id'] == urun]['sales'].values.reshape(-1, 1)
        scaled_urun_sales = scaler.transform(urun_sales)

        split_idx = int(len(scaled_urun_sales) * 0.8)

        # Train
        for i in range(ZAMAN_PENCERESI, split_idx):
            X_train_list.append(scaled_urun_sales[i - ZAMAN_PENCERESI: i])
            y_train_list.append(scaled_urun_sales[i][0])

        # Validation
        for i in range(split_idx, len(scaled_urun_sales)):
            X_val_list.append(scaled_urun_sales[i - ZAMAN_PENCERESI: i])
            y_val_list.append(scaled_urun_sales[i][0])

    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    X_val = np.array(X_val_list)
    y_val = np.array(y_val_list)

    # 🧠 GRU MİMARİSİ
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(ZAMAN_PENCERESI, 1)),
        Dropout(0.2),
        GRU(32, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mae')

    # 🛑 ADIM 4: GELİŞMİŞ CALLBACKS
    model_kayit_yolu = os.path.join(model_klasoru, f"GRU_Model_{model_adi}.keras")
    checkpoint = ModelCheckpoint(filepath=model_kayit_yolu, monitor='val_loss', save_best_only=True, verbose=0)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print(f"🔥 {model_adi} Eğitimi Başlıyor (Epoch: 20)...")
    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=64,
        validation_data=(X_val, y_val),
        shuffle=False,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    # Scaler Kaydı
    with open(os.path.join(model_klasoru, f"GRU_Scaler_{model_adi}.pkl"), 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ {model_adi} TAMAMLANDI.")

print("\n" + "=" * 50)
print("🏆 TEZ İÇİN KURŞUN GEÇİRMEZ GRU MODELLERİ TAMAMLANDI!")
print("=" * 50)