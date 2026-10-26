import pandas as pd
import time
import re
from google import genai
import os
import random

# 1. AYARLAR
API_KEY = ""
client = genai.Client(api_key=API_KEY)
# Modeli daha hızlı ve stabil olan 1.5 Flash'a çekelim, lite bazen çok takılıyor
MODEL_ID = "gemini-3.1-flash-lite-preview"

dosya_adi = 'Akilli_Sentetik_Hobbies.csv'

print("SENTETİK VERİNİN BAŞLATILDIĞI YER ")
profil_df = pd.read_csv('Hobbies_Urun_Profilleri.csv', sep=';')

# 2. KALINAN YERİ BULMA (RESUME ÖZELLİĞİ)
tamamlananlar = []
if os.path.exists(dosya_adi):
    mevcut_df = pd.read_csv(dosya_adi, sep=';')
    tamamlananlar = mevcut_df['item_id'].tolist()
    print(f"📂 Dosya bulundu! {len(tamamlananlar)} ürün zaten üretilmiş. Kaldığımız yerden devam ediyoruz...")
else:
    print("📂 Yeni dosya oluşturulacak. Sıfırdan başlanıyor...")

# 3. ÜRETİM DÖNGÜSÜ
for index, row in profil_df.iterrows():
    urun_id = row['urun_id']
    hedef_id = urun_id + "_SYN"

    # ÜRÜN VARSA ATLA !
    if hedef_id in tamamlananlar:
        continue

    dogum_gunu = int(row['dogum_gunu'])
    aktif_gun = int(row['aktif_gun_sayisi'])
    sifir_oran = float(str(row['sifir_orani_%']).replace(',', '.'))
    max_satis = int(row['max_gunluk_satis'])
    ort_satis = float(str(row['ort_satis_adeti']).replace(',', '.'))

    if aktif_gun < 10:
        continue

    yeni_dogum_gunu = random.randint(1, 1500)
    yeni_aktif_gun = 1913 - yeni_dogum_gunu + 1

    print(f"📦 {hedef_id} üretiliyor... (Yeni Doğum: {yeni_dogum_gunu}. gün)")

    basarili = False
    deneme_sayisi = 0

    while not basarili:
        deneme_sayisi += 1

        try:
            prompt = f"""
            Sen istatistiksel bir zaman serisi simülatörüsün. {hedef_id} adlı ürün için {yeni_aktif_gun} adet günlük satış verisi üret.
            KESİN KURALLAR:
            1. Ürettiğin {yeni_aktif_gun} sayının yaklaşık %{sifir_oran} kadarı "0" olmalıdır.
            2. Satış olan günlerde satışlar GENELLİKLE 1 veya 2 olmalıdır (Ortalama {ort_satis}).
            3. ÇOK NADİREN satışlar yükselebilir ancak {max_satis} sayısını geçemez.
            4. Sadece virgülle ayrılmış tam sayılar ver (Örn: 0,0,1,0...).
            5. Toplam sayı adedi TAM OLARAK {yeni_aktif_gun} olmalıdır.
            """

            response = client.models.generate_content(model=MODEL_ID, contents=prompt)
            clean_text = re.sub(r'[^0-9,]', '', response.text.strip())
            gemini_verisi = [s for s in clean_text.split(',') if s]

            if len(gemini_verisi) > (yeni_aktif_gun * 0.8):
                baslangic_sifirlari = ['0'] * (yeni_dogum_gunu - 1)
                tam_seri = baslangic_sifirlari + gemini_verisi

                tam_seri = tam_seri[:1913]
                while len(tam_seri) < 1913:
                    tam_seri.append('0')

                # SİHİR BURADA: ANINDA KAYDETME (Append Mode)
                row_data = {'item_id': hedef_id}
                for i in range(1, 1914):
                    row_data[f'd_{i}'] = tam_seri[i - 1]

                df_tek = pd.DataFrame([row_data])

                # Dosya yoksa başlıklarla oluştur, varsa altına ekle
                if not os.path.exists(dosya_adi):
                    df_tek.to_csv(dosya_adi, index=False, sep=';')
                else:
                    df_tek.to_csv(dosya_adi, mode='a', header=False, index=False, sep=';')

                basarili = True
                print(f"  Başarılı ve Kaydedildi! (4 SANİYE BEKLENECEK)")
                time.sleep(4)  # API ENGELİNE TAKILMAMAK İÇİN SLEEP KOMUTU

            else:
                print(f"    Eksik veri ({len(gemini_verisi)}), tekrar deneniyor.")
                time.sleep(2)

        except Exception as e:
            hata_mesaji = str(e).lower()
            # GELEN HATANIN NE OLDUĞUNU YAZDIRACAĞIM KOD Kİ ENGELE Mİ TAKILDIM YOKSA API Mİ YOĞN BİLEYİM
            print(f"\n  GOOGLE'DAN GELEN GERÇEK HATA: {e}\n")

            if "429" in hata_mesaji or "quota" in hata_mesaji:
                print("  Limit aşıldı! 1 dakika bekleniyor...")
                time.sleep(60)
            else:
                print(f" Diğer Hata! 10 saniye bekleniyor...")
                time.sleep(10)