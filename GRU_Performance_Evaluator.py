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
print("--- 🏁 GRU FINAL EVALUATOR (DİKEY VERİ UYUMLU) ---")
print("=" * 60)

proje_klasoru = os.path.dirname(os.path.abspath(__file__))
model_klasoru = os.path.join(proje_klasoru, "Tez_Modelleri")
test_dosya_yolu = os.path.join(proje_klasoru, "1_Test_Set_100_Gercek.csv")

# 1. Veriyi Oku (Sütun isimleri yoksa biz veriyoruz)
# 1. Veriyi Oku (Noktalı virgül ile ve kendi başlıklarını kullanarak)
df_test = pd.read_csv(test_dosya_yolu, sep=';')

# 2. Benzersiz ürünlerin listesini al (Sütun adı senin dosyanda 'item_id' imiş)
unique_items = df_test['item_id'].unique()
print(f"📦 Toplam {len(unique_items)} test ürünü bulundu.")

ZAMAN_PENCERESI = 30
model_isimleri = ['Baseline_0', 'Mix_25', 'Mix_50', 'Mix_100']
final_sonuclar = []

for m_adi in model_isimleri:
    model_path = os.path.join(model_klasoru, f"GRU_Model_{m_adi}.keras")
    scaler_path = os.path.join(model_klasoru, f"GRU_Scaler_{m_adi}.pkl")

    if not os.path.exists(model_path):
        continue

    print(f"\n🧠 {m_adi} test ediliyor...")
    model = tf.keras.models.load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    y_true_all = []
    y_pred_all = []

    # 🚀 Dikey Veri İşleme Mantığı
    for item_id in tqdm(unique_items, desc=f"Tahmin: {m_adi}"):

        # Sadece bu ürüne ait satırları filtrele
        item_data = df_test[df_test['item_id'] == item_id]

        # Gün sırasının bozulmadığından emin olmak için direkt sırayla satışları al
        # Senin yapında 3. sütun (index 2) satış adetidir
        sales_history = item_data['sales'].values.astype(float).reshape(-1, 1)

        if len(sales_history) < ZAMAN_PENCERESI + 1:
            continue  # Veri çok kısaysa atla

        scaled_sales = scaler.transform(sales_history)

        # Son 100 günlük veriyi test et
        for i in range(len(scaled_sales) - 100, len(scaled_sales)):
            X_input = scaled_sales[i - ZAMAN_PENCERESI: i].reshape(1, ZAMAN_PENCERESI, 1)

            pred_scaled = model.predict(X_input, verbose=0)

            true_val = scaler.inverse_transform(scaled_sales[i].reshape(-1, 1))[0][0]
            pred_val = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]

            y_true_all.append(true_val)
            y_pred_all.append(max(0, pred_val))

    # Metrikleri Hesapla
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    mae = mean_absolute_error(y_true_all, y_pred_all)
    smape_val = calculate_smape(y_true_all, y_pred_all)

    final_sonuclar.append({
        'Model': m_adi,
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'sMAPE': round(smape_val, 2)
    })

# Final Raporu Kaydet
sonuc_df = pd.DataFrame(final_sonuclar)
print("\n" + "=" * 60)
print("📊 TEZ FINAL PERFORMANS TABLOSU (Hobbies)")
print(sonuc_df.to_string(index=False))
print("=" * 60)
sonuc_df.to_csv('GRU_Final_Tez_Sonuclari_Dikey.csv', index=False, sep=';')