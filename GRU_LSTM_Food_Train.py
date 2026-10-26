import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
import warnings

warnings.filterwarnings('ignore')

print("--- GEMINI SAMPİYONLAR LİGİ: HATASIZ EGİTİM MOTORU BASLİYOR ---")

# 1. Klasör Ayarları
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

models_save_dir = os.path.join(PROJECT_DIR, "Egitilmis_Modeller_Gemini")
graphs_save_dir = os.path.join(PROJECT_DIR, "Tez_Grafikleri_Gemini")

if not os.path.exists(models_save_dir):
    os.makedirs(models_save_dir)
if not os.path.exists(graphs_save_dir):
    os.makedirs(graphs_save_dir)

# 2. Hiperparametreler
WINDOW_SIZE = 30
EPOCHS = 20
BATCH_SIZE = 64

def prepare_multivariate_data(file_path):
    print(f"\n Veri yukleniyor: {os.path.basename(file_path)}...")
    df = pd.read_csv(file_path, sep=';')

    df['day_num'] = df['day'].str.extract('(\d+)').astype(int)

    # 1. MUDAHALE: Kimlik cakismasini engelle (Sentetiklere SYN takisi ekle)
    print(" Kimlik cakismasi engelleniyor (Sentetikler ayriliyor)...")
    mask = df.duplicated(subset=['item_id', 'day_num'], keep='first')
    df.loc[mask, 'item_id'] = df.loc[mask, 'item_id'].astype(str) + '_SYN'

    df = df.sort_values(by=['item_id', 'day_num']).reset_index(drop=True)

    # 2. MUDAHALE: Aykiri degerleri tirpanla
    print(" Aykiri deger tirpani calisiyor (Outlier sinirlandirmasi)...")
    upper_limit = df['sales'].quantile(0.99)
    df['sales'] = df['sales'].clip(lower=0, upper=upper_limit)

    scaler = MinMaxScaler()
    df[['sales_scaled', 'price_scaled']] = scaler.fit_transform(df[['sales', 'sell_price']])

    X, y = [], []

    print(" Zaman pencereleri olusturuluyor...")
    for item, group in df.groupby('item_id'):
        values = group[['sales_scaled', 'price_scaled']].values
        target = group['sales_scaled'].values

        for i in range(WINDOW_SIZE, len(values)):
            X.append(values[i - WINDOW_SIZE:i])
            y.append(target[i])

    # 3. MUDAHALE: Verileri karistir (Shuffle)
    print(" Veriler homojen sekilde karistiriliyor...")
    X_shuffled, y_shuffled = shuffle(np.array(X), np.array(y), random_state=42)

    return X_shuffled, y_shuffled

def build_and_train_model(X_train, y_train, model_type, file_name):
    print(f"\n Model Insasi: {model_type} basliyor...")

    model = Sequential()
    input_shape = (X_train.shape[1], X_train.shape[2])

    if model_type == 'LSTM':
        model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
    elif model_type == 'GRU':
        model.add(GRU(64, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(GRU(32))

    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    print(f" Egitim Ateslendi! (Max {EPOCHS} Epoch) - {file_name}")
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_split=0.2, callbacks=[early_stop], verbose=1)

    base_name = file_name.replace(".csv", "")

    model.save(os.path.join(models_save_dir, f"{base_name}_{model_type}.keras"))

    val_loss_list = history.history['val_loss']
    best_epoch_index = np.argmin(val_loss_list)
    best_epoch_num = best_epoch_index + 1
    best_val_loss = val_loss_list[best_epoch_index]

    epochs_range = range(1, len(val_loss_list) + 1)

    # 4. MUDAHALE: Kirmizi cizgili sov grafigi
    plt.figure(figsize=(10, 6))

    plt.plot(epochs_range, history.history['loss'], label='Egitim Hatasi (Train Loss)', color='#1f77b4', linewidth=2)
    plt.plot(epochs_range, val_loss_list, label='Dogrulama Hatasi (Val Loss)', color='#ff7f0e', linewidth=2, linestyle='--')

    plt.axvline(x=best_epoch_num, color='red', linestyle=':', linewidth=2, label=f'Erken Durdurma (Optimum: Epoch {best_epoch_num})')
    plt.scatter(best_epoch_num, best_val_loss, color='red', s=100, zorder=5)

    plt.title(f"{base_name} - {model_type} Ogrenme Egrisi", fontsize=14)
    plt.xlabel('Epoch Sayisi', fontsize=12)
    plt.ylabel('Hata Miktari (MSE)', fontsize=12)
    plt.xticks(epochs_range)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    graph_path = os.path.join(graphs_save_dir, f"{base_name}_{model_type}_Loss_EarlyStop.png")
    plt.savefig(graph_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f" {model_type} Modeli ve grafigi kaydedildi!")

# Gemini hedefleri icin ozel liste (Llama dosyalariyla karismamasi icin acikca belirttik)
target_files = [
    '1_Food_Train_Baseline_0_Sentetik.csv',
    '2_Food_Train_Mix_25.csv',
    '3_Food_Train_Mix_50.csv',
    '4_Food_Train_Mix_100.csv'
]

available_files = [f for f in target_files if f in os.listdir(PROJECT_DIR)]

if not available_files:
    print(f" UYARI: Hedef Gemini CSV dosyalari bulunamadi!")
else:
    for file_name in available_files:
        file_path = os.path.join(PROJECT_DIR, file_name)

        X, y = prepare_multivariate_data(file_path)
        print(f"\n Veri Seti Ozeti: {X.shape[0]} adet egitim ornegi uretildi.")

        build_and_train_model(X, y, 'LSTM', file_name)
        build_and_train_model(X, y, 'GRU', file_name)

    print(f"\n BUTUN ISLEM TAMAMLANDI! Grafikler ve modeller seni bekliyor.")