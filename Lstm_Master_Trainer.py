import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

print("-4'LÜ HİBRİT LSTM EĞİTİM MOTORU (LEAKAGE-FREE VERSİYON)-")
proje_yolu = os.getcwd()

veri_klasoru = os.path.join(proje_yolu, "Tez_Veri_Setleri")
model_klasoru = os.path.join(proje_yolu, "Tez_Modelleri")

if not os.path.exists(model_klasoru):
    os.makedirs(model_klasoru)

egitim_setleri = {
    "Baseline_0": "2_Train_Baseline_0_Sentetik.csv",
    "Mix_25": "3_Train_Mix_25.csv",
    "Mix_50": "4_Train_Mix_50.csv",
    "Mix_100": "5_Train_Mix_100.csv"
}
#parametre ayarlarını yaptığımız yer hocam
WINDOW_SIZE = 30
EPOCHS = 50
BATCH_SIZE = 64

for model_adi, dosya_adi in egitim_setleri.items():
    print(f"\n[{model_adi}] Modeli eğitiliyor. Veri: {dosya_adi}")

    dosya_yolu = os.path.join(veri_klasoru, dosya_adi)
    if not os.path.exists(dosya_yolu):
        continue

    df = pd.read_csv(dosya_yolu, sep=';')

    #data leakage önleme yeri
    #Her ürünün satış geçmişinin ilk %80 i Train, son %20si Validation olacak
    train_data_list = []
    val_data_list = []

    urunler = df['item_id'].unique()

    for urun in urunler:
        urun_df = df[df['item_id'] == urun].copy()
        satislar = urun_df[['sales']].values

        split_idx = int(len(satislar) * 0.8)
        train_data_list.append(satislar[:split_idx])
        val_data_list.append(satislar[split_idx:])

    #train val değerlerini birleştir kral
    train_raw_all = np.vstack(train_data_list)
    val_raw_all = np.vstack(val_data_list)

    #HELAL SCALER EĞİTİMİ (Sadece Train'i görüyor!)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_raw_all)  # Validation'ı asla görmedi!

    X_train, y_train, X_val, y_val = [], [], [], []

    # 3. FRANKENSTEIN HATASINI ÇÖZME: Sekansları Ürün Bazlı Oluşturma
    for i, urun in enumerate(urunler):
        train_satis = train_data_list[i]
        val_satis = val_data_list[i]

        train_scaled = scaler.transform(train_satis)
        val_scaled = scaler.transform(val_satis)

        #Train Sekansları
        for j in range(len(train_scaled) - WINDOW_SIZE):
            X_train.append(train_scaled[j:j + WINDOW_SIZE, 0])
            y_train.append(train_scaled[j + WINDOW_SIZE, 0])

        #Validation Sekansları
        for j in range(len(val_scaled) - WINDOW_SIZE):
            X_val.append(val_scaled[j:j + WINDOW_SIZE, 0])
            y_val.append(val_scaled[j + WINDOW_SIZE, 0])

    X_train = np.array(X_train).reshape(-1, WINDOW_SIZE, 1)
    y_train = np.array(y_train)
    X_val = np.array(X_val).reshape(-1, WINDOW_SIZE, 1)
    y_val = np.array(y_val)

    #LSTM MİMARİSİ
    model = Sequential([
        Input(shape=(WINDOW_SIZE, 1)),
        LSTM(64, return_sequences=True),
        LSTM(32, return_sequences=False),
        Dense(16),  # Eski relu buradan silindi
        LeakyReLU(alpha=0.1),  # ASLA ÖLMEYEN ZOMBİ NÖRON EKLENDİ
        Dense(1)
    ])

    # Hızı 0.005'ten 0.001'e çektik grafik güzel gözükecek acele yok
    custom_optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=custom_optimizer, loss='mae')

    # Early Stop'u SADECE BİR KERE ve doğru yerde tanımlıyoruz
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    print(f"{model_adi} motoru çalıştırıldı. Bekleyiniz...")

    # fit işlemi yukarıda earlystopu görecek
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1,
        shuffle=False  #Zaman serisi olduğu için karıştırmıyoruz
    )

    #Kayıt İşlemleri
    model.save(os.path.join(model_klasoru, f"LSTM_Model_{model_adi}.keras"))
    with open(os.path.join(model_klasoru, f"Scaler_{model_adi}.pkl"), 'wb') as f:
        pickle.dump(scaler, f)

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss (Eğitim Kaybı)')
    plt.plot(history.history['val_loss'], label='Val Loss (Doğrulama Kaybı)')
    plt.title(f'{model_adi} - Öğrenme Eğrisi (Leakage-Free)')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_klasoru, f"Grafik_{model_adi}.png"))
    plt.close()

    print(f"✅ {model_adi} temiz bir şekilde tamamlandı!")

print("\nTÜM EĞİTİMLER KUSURSUZ BİR ŞEKİLDE TAMAMLANDI!")