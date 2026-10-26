import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import warnings

warnings.filterwarnings('ignore')

print("--- TEST VE DEGERLENDIRME MOTORU BASLIYOR ---")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Modellerin oldugu klasoru Llama veya Gemini hangisini test ediyorsan ona gore ayarla
models_dir = os.path.join(PROJECT_DIR, "Egitilmis_Modeller_Llama")
graphs_save_dir = os.path.join(PROJECT_DIR, "Tez_Grafikleri_Llama")

if not os.path.exists(graphs_save_dir):
    os.makedirs(graphs_save_dir)

WINDOW_SIZE = 30


def calculate_smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    mask = denominator != 0
    smape = np.zeros_like(y_true, dtype=float)
    smape[mask] = diff[mask] / denominator[mask]
    return np.mean(smape) * 100


def prepare_test_data(file_path):
    print(f"Test verisi yukleniyor: {os.path.basename(file_path)}...")
    df = pd.read_csv(file_path, sep=';')

    df['day_num'] = df['day'].str.extract('(\d+)').astype(int)
    df = df.sort_values(by=['item_id', 'day_num']).reset_index(drop=True)

    df['sales'] = pd.to_numeric(df['sales'], errors='coerce').fillna(0)
    df['sell_price'] = pd.to_numeric(df['sell_price'], errors='coerce').fillna(0)

    scaler_sales = MinMaxScaler()
    scaler_price = MinMaxScaler()

    df['sales_scaled'] = scaler_sales.fit_transform(df[['sales']])
    df['price_scaled'] = scaler_price.fit_transform(df[['sell_price']])

    X, y_true_scaled, y_true_real = [], [], []

    for item, group in df.groupby('item_id'):
        values = group[['sales_scaled', 'price_scaled']].values
        target_scaled = group['sales_scaled'].values
        target_real = group['sales'].values

        for i in range(WINDOW_SIZE, len(values)):
            X.append(values[i - WINDOW_SIZE:i])
            y_true_scaled.append(target_scaled[i])
            y_true_real.append(target_real[i])

    return np.array(X), np.array(y_true_real), scaler_sales


test_file_name = '1_Test_Set_100_Gercek.csv'
test_file_path = os.path.join(PROJECT_DIR, test_file_name)

if not os.path.exists(test_file_path):
    print(f"UYARI: Test dosyasi bulunamadi: {test_file_path}")
else:
    X_test, y_test_real, scaler = prepare_test_data(test_file_path)
    print(f"Test verisi hazir! Toplam test ornegi: {X_test.shape[0]}")

    results = []

    # Sadece .keras uzantili egitilmis modelleri bul
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.keras')]
    model_files.sort()

    if not model_files:
        print("UYARI: Test edilecek egitilmis model bulunamadi!")
    else:
        for model_file in model_files:
            print(f"\nModel test ediliyor: {model_file}")
            model = load_model(os.path.join(models_dir, model_file))

            # Tahminleri uret
            predictions_scaled = model.predict(X_test, verbose=0)

            # Tahminleri gercek satis rakamlarina cevir
            predictions_real = scaler.inverse_transform(predictions_scaled).flatten()

            # Satis eksi olamayacagi icin negatifleri sifirla
            predictions_real = np.clip(predictions_real, a_min=0, a_max=None)

            # Metrik hesaplamalari
            rmse = np.sqrt(mean_squared_error(y_test_real, predictions_real))
            mse = mean_squared_error(y_test_real, predictions_real)
            mape = mean_absolute_percentage_error(y_test_real, predictions_real) * 100
            smape = calculate_smape(y_test_real, predictions_real)
            r2 = r2_score(y_test_real, predictions_real)

            results.append({
                "Model Adi": model_file.replace(".keras", ""),
                "RMSE": round(rmse, 4),
                "MSE": round(mse, 4),
                "MAPE (%)": round(mape, 2),
                "sMAPE (%)": round(smape, 2),
                "R-Kare": round(r2, 4)
            })

        df_results = pd.DataFrame(results)
        print("\n--- TEST SONUCLARI ---")
        print(df_results)

        # Tabloyu PNG olarak cizme islemi
        fig, ax = plt.subplots(figsize=(14, len(df_results) * 0.6 + 1.5))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=df_results.values,
                         colLabels=df_results.columns,
                         cellLoc='center',
                         loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.8)

        # Baslik satirini yesil ve beyaz yazi yap
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(color='white', weight='bold')
                cell.set_facecolor('#4CAF50')
            else:
                cell.set_facecolor('#f9f9f9' if row % 2 == 0 else 'white')

        plt.title("LSTM ve GRU Test Verisi Hata Metrikleri", fontsize=16, pad=20)

        table_img_path = os.path.join(graphs_save_dir, "Test_Sonuclari_Tablosu.png")
        plt.savefig(table_img_path, bbox_inches='tight', dpi=300)
        plt.close()

        print(f"\nISLEM TAMAM! Tablo PNG olarak kaydedildi: {table_img_path}")