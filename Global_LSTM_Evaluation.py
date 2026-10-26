import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import random

print("--- 🚀 ULTRA HIZLI ROLLING FORECAST & AYLIK/YILLIK RAPORLAMA BAŞLIYOR ---")

# 1. Klasör Hazırlığı
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "Tez_Final_Grafikleri")
if not os.path.exists(desktop_path):
    os.makedirs(desktop_path)

# 2. Veri ve Model Yükleme
df = pd.read_csv('Sentetik_Hobbies_Final_Dataset.csv', sep=';')
df['d_num'] = df['day'].str.replace('d_', '').astype(int)

model_path = os.path.join(os.path.expanduser("~"), "Desktop", "LSTM_Master_Sonuclari", "LSTM_Model.keras")
model = load_model(model_path)

scaler = MinMaxScaler(feature_range=(0, 1))
df['sales_scaled'] = scaler.fit_transform(df[['sales']].values)

# 3. Ürün Seçimi
h1_urunleri = df[df['item_id'].str.contains('HOBBIES_1')]['item_id'].unique()[-70:]
h2_urunleri = df[df['item_id'].str.contains('HOBBIES_2')]['item_id'].unique()[-30:]
hedef_urunler = list(h1_urunleri) + list(h2_urunleri)

print(f"Toplam {len(hedef_urunler)} ürün için analiz başlıyor. (Hızlı Motor Devrede ⚡)")

TOHUM_SURESI = 30
PENCERE = 28

# Raporlama Listeleri
aylik_rapor = []
yillik_rapor = []

# Rastgele 5 ürünün detaylı grafiği için seçim
grafik_cizilecek_urunler = random.sample(hedef_urunler, 5)

for i, urun in enumerate(hedef_urunler, 1):
    urun_df = df[df['item_id'] == urun].sort_values(by='d_num')
    aktif_df = urun_df[urun_df['sell_price'] > 0].copy()

    if len(aktif_df) <= TOHUM_SURESI:
        continue

    tohum_scaled = aktif_df.iloc[:TOHUM_SURESI]['sales_scaled'].values
    gercek_gelecek_adet = aktif_df.iloc[TOHUM_SURESI:]['sales'].values

    current_window = list(tohum_scaled[-PENCERE:])
    tahmin_edilen_gelecek_tam_sayi = []

    # ⚡ HIZLANDIRILMIŞ TAHMİN DÖNGÜSÜ ⚡
    for _ in range(len(gercek_gelecek_adet)):
        input_seq = np.array(current_window).reshape(1, PENCERE, 1)
        input_tensor = tf.convert_to_tensor(input_seq, dtype=tf.float32)

        # model.predict() yerine alt seviye çağrı (Kat kat hızlıdır!)
        pred_scaled = model(input_tensor, training=False).numpy()[0][0]

        # Ölçeği geri çevir
        beklenen_deger_float = scaler.inverse_transform([[pred_scaled]])[0][0]

        # 🎲 POISSON SİHRİ: Beklenen değeri alıp tam sayı zar atıyoruz (Gerçekçi zıplamalar)
        # Negatif olmaması için max(0, x) kullanıyoruz
        tahmin_adet = np.random.poisson(lam=max(0, beklenen_deger_float))
        tahmin_edilen_gelecek_tam_sayi.append(tahmin_adet)

        # Döngüyü beslemek için Poisson'lu sayıyı tekrar 0-1 arasına scale edip pencereye ekle
        yeni_scaled = scaler.transform([[tahmin_adet]])[0][0]
        current_window.append(yeni_scaled)
        current_window.pop(0)

    # --- 📊 AYLIK VE YILLIK PERFORMANS HESAPLAMA ---
    tahmin_dizisi = np.array(tahmin_edilen_gelecek_tam_sayi)
    gercek_dizisi = np.array(gercek_gelecek_adet)

    # Sadece ilk ayı (30 gün) ve ilk yılı (365 gün) kıyaslayalım (Ömrü yetiyorsa)
    if len(gercek_dizisi) >= 30:
        aylik_gercek = np.sum(gercek_dizisi[:30])
        aylik_tahmin = np.sum(tahmin_dizisi[:30])
        aylik_hata_orani = abs(aylik_gercek - aylik_tahmin) / max(1, aylik_gercek)  # Sıfıra bölme hatasını önle
        aylik_dogruluk = max(0, (1 - aylik_hata_orani)) * 100

        aylik_rapor.append({
            'Urun_ID': urun,
            'Ilk_1_Ay_Gercek_Satis': aylik_gercek,
            'Ilk_1_Ay_Tahmin_Satis': aylik_tahmin,
            'Aylik_Dogruluk_Orani(%)': round(aylik_dogruluk, 2)
        })

    if len(gercek_dizisi) >= 365:
        yillik_gercek = np.sum(gercek_dizisi[:365])
        yillik_tahmin = np.sum(tahmin_dizisi[:365])
        yillik_hata_orani = abs(yillik_gercek - yillik_tahmin) / max(1, yillik_gercek)
        yillik_dogruluk = max(0, (1 - yillik_hata_orani)) * 100

        yillik_rapor.append({
            'Urun_ID': urun,
            'Ilk_1_Yil_Gercek_Satis': yillik_gercek,
            'Ilk_1_Yil_Tahmin_Satis': yillik_tahmin,
            'Yillik_Dogruluk_Orani(%)': round(yillik_dogruluk, 2)
        })

    print(f"[{i}/{len(hedef_urunler)}] {urun} tamamlandı. (Hızlandırılmış)")

    # Eğer rastgele seçilen 5 üründen biriyse grafiğini çiz
    if urun in grafik_cizilecek_urunler:
        plt.figure(figsize=(15, 6))
        # Gerçekçi görünmesi için sadece ilk 150 günü çizelim (Karmaşayı önler)
        cizim_siniri = min(150, len(gercek_dizisi))
        plt.plot(gercek_dizisi[:cizim_siniri], label='Gerçek Satış', color='green', alpha=0.6, linewidth=2)
        plt.plot(tahmin_dizisi[:cizim_siniri], label='LSTM + Poisson Tahmini', color='orange', linestyle='--',
                 linewidth=2)
        plt.title(f'Günlük Satış Dinamikleri (İlk 150 Gün): {urun}')
        plt.xlabel('Zaman (Günler)')
        plt.ylabel('Satış Adedi (Tam Sayı)')
        plt.legend()
        plt.savefig(os.path.join(desktop_path, f"Grafik_{urun}.png"))
        plt.close()

# 6. 💾 EXCEL (CSV) RAPORLARINI KAYDET
print("\nExcel raporları oluşturuluyor...")

df_aylik = pd.DataFrame(aylik_rapor)
df_yillik = pd.DataFrame(yillik_rapor)

aylik_yol = os.path.join(desktop_path, "Aylik_Basari_Raporu.csv")
yillik_yol = os.path.join(desktop_path, "Yillik_Basari_Raporu.csv")

df_aylik.to_csv(aylik_yol, index=False, sep=';')
df_yillik.to_csv(yillik_yol, index=False, sep=';')

print(f"\n✅ İŞLEM KUSURSUZ TAMAMLANDI!")
print(f"Genel Aylık Doğruluk Ortalaması: %{round(df_aylik['Aylik_Dogruluk_Orani(%)'].mean(), 2)}")
if not df_yillik.empty:
    print(f"Genel Yıllık Doğruluk Ortalaması: %{round(df_yillik['Yillik_Dogruluk_Orani(%)'].mean(), 2)}")
print(f"Raporlar ve grafikler '{desktop_path}' klasörüne kaydedildi.")