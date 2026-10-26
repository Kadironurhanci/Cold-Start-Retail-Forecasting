import pandas as pd
import numpy as np
import os
import pickle
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm


def calculate_smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)


print("\n" + "=" * 60)
print("-TEZ MASTER LSTM vs GRU ---")
print("=" * 60)

proje_klasoru = os.path.dirname(os.path.abspath(__file__))
model_klasoru = os.path.join(proje_klasoru, "Tez_Modelleri")
test_dosya_yolu = os.path.join(proje_klasoru, "1_Test_Set_100_Gercek.csv")

# Veriyi Oku (Noktalı virgül ve kendi başlıklarıyla)
df_test = pd.read_csv(test_dosya_yolu, sep=';')
unique_items = df_test['item_id'].unique()

ZAMAN_PENCERESI = 30
mimariler = ['LSTM', 'GRU']
model_isimleri = ['Baseline_0', 'Mix_25', 'Mix_50', 'Mix_100']
master_sonuclar = []

# Önce LSTM sonra GRU
for mimari in mimariler:
    for m_adi in model_isimleri:
        model_path = os.path.join(model_klasoru, f"{mimari}_Model_{m_adi}.keras")
        scaler_path = os.path.join(model_klasoru, f"Scaler_{m_adi}.pkl")

        if not os.path.exists(model_path):
            print(f"⚠️ {mimari} {m_adi} bulunamadı, atlanıyor...")
            continue

        print(f"\n🥊 RİNGE ÇIKAN: {mimari} - {m_adi}")
        model = tf.keras.models.load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        y_true_all, y_pred_all = [], []

        for item_id in tqdm(unique_items, desc=f"{mimari} {m_adi} Tahmini"):
            item_data = df_test[df_test['item_id'] == item_id]
            sales_history = item_data['sales'].values.astype(float).reshape(-1, 1)

            if len(sales_history) < ZAMAN_PENCERESI + 1:
                continue

            scaled_sales = scaler.transform(sales_history)

            for i in range(len(scaled_sales) - 100, len(scaled_sales)):
                X_input = scaled_sales[i - ZAMAN_PENCERESI: i].reshape(1, ZAMAN_PENCERESI, 1)
                pred_scaled = model.predict(X_input, verbose=0)

                true_val = scaler.inverse_transform(scaled_sales[i].reshape(-1, 1))[0][0]
                pred_val = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]

                y_true_all.append(true_val)
                y_pred_all.append(max(0, pred_val))

        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)

        rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
        mae = mean_absolute_error(y_true_all, y_pred_all)
        smape_val = calculate_smape(y_true_all, y_pred_all)

        master_sonuclar.append({
            'Mimari': mimari,
            'Model_Tipi': m_adi,
            'RMSE': round(rmse, 4),
            'MAE': round(mae, 4),
            'sMAPE': round(smape_val, 2)
        })

# tablomu göster kral yorma
master_df = pd.DataFrame(master_sonuclar)
print("\n" + "=" * 60)
print("TEZ karşılaştırma tablosu (LSTM vs GRU)")
print(master_df.to_string(index=False))
print("=" * 60)

master_df.to_csv('MASTER_Tez_Karsilastirma.csv', index=False, sep=';')
print("✅ Tüm sonuçlar 'MASTER_Tez_Karsilastirma.csv' olarak kaydedildi!")