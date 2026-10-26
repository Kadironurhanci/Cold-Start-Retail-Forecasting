import pandas as pd
import numpy as np
import os, re, time, random
from google import genai
from google.genai import types

# 🛑 1. API & REPRODUCIBILITY SETUP
API_KEY = ""
client = genai.Client(api_key=API_KEY)
PRIMARY_MODEL = 'gemini-3.1-flash-lite-preview'

# 2. Üretim Emrini Yükle
emri_df = pd.read_csv('Llama_Gida_Uretim_Emri_400.csv', sep=';')
dosya_adi = 'Sentetik_Gida_Verileri_Gemini_Full.csv'

# 🛑 3. PERFORMANS FIX: Mevcut ID'leri set olarak hafızaya al
existing_ids = set()
if os.path.exists(dosya_adi) and os.path.getsize(dosya_adi) > 0:
    try:
        check_df = pd.read_csv(dosya_adi, sep=';', usecols=['item_id'])
        existing_ids = set(check_df['item_id'].astype(str).unique())
    except:
        print("⚠️ Dosya okunamadı, sıfırdan başlanıyor.")

print(f"🚀 Gıda Fabrikası (Elite Edition) Başladı! Hedef: {len(emri_df)} ürün.")

for index, row in emri_df.iterrows():
    hedef_id = str(row['urun_id'])

    # Zaten üretilmişse vakit kaybetme
    if hedef_id in existing_ids:
        continue

    dogum_gunu = int(row['dogum_gunu'])
    hedef_sifir = row['hedef_sifir_orani']
    ort_satis = row['ort_satis_adeti']
    max_satis = row['max_gunluk_satis']

    # M5 zaman ölçeği: 1913 gün
    aktif_gun_sayisi = max(0, min(1913, 1913 - dogum_gunu + 1))
    print(f"\n📦 {hedef_id} işleniyor... (Doğum: d_{dogum_gunu}, Aktif: {aktif_gun_sayisi} gün)")

    llama_verisi = []

    while len(llama_verisi) < aktif_gun_sayisi:
        eksik = aktif_gun_sayisi - len(llama_verisi)
        istenecek = min(eksik, 400)  # Batch size 400 idealdir

        wait_time = 5  # Hata durumunda bekleme süresi
        for attempt in range(3):
            try:
                prompt = f"""
                Sen Walmart Gıda Reyonu Müdürüsün. Bana tam olarak {istenecek} adet günlük satış sayısı üret.
                Kurallar:
                - Ürün ortalaması {ort_satis}, maksimum satışı {max_satis} olmalı.
                - Sıfır (satışsız gün) oranı yaklaşık %{hedef_sifir} olmalı.
                - Gıda ürünü olduğu için hafta sonları hafif artışlar yap.
                FORMAT: Sadece sayıları virgülle ayırarak yaz. Metin ekleme.
                """

                # Yeni SDK: generate_content
                response = client.models.generate_content(
                    model=PRIMARY_MODEL,
                    contents=prompt
                )

                content = response.text

                # 🛑 4. REGEX FIX: Sadece rakamları güvenli şekilde ayıkla
                parca = re.findall(r'\d+', content)

                if len(parca) >= (istenecek * 0.85):  # Verinin büyük kısmı geldiyse kabul
                    llama_verisi.extend(parca)
                    print(f"   [+] {len(parca)} gün eklendi. ({len(llama_verisi)}/{aktif_gun_sayisi})")
                    break
                else:
                    print(f"   ⚠️ Eksik veri ({len(parca)}/{istenecek}), tekrar deneniyor...")
                    time.sleep(wait_time)
            except Exception as e:
                print(f"   🚨 Deneme {attempt + 1} Hatası: {e}")
                time.sleep(wait_time)
                wait_time *= 2  # Exponential backoff

    # 🛑 5. VERİ TİPİ VE HİJYEN FIX
    # Stringleri integer yapıyoruz (ML pipeline hatasız çalışsın diye)
    aktif_seri = [int(x) for x in llama_verisi[:aktif_gun_sayisi]]

    # 🛑 6. DOĞUM GÜNÜ ÖNCESİ SIFIRLAR (Kankanın istediği tam hijyen)
    tam_seri = ([0] * (dogum_gunu - 1)) + aktif_seri

    # 1913'e tam eşitleme
    tam_seri = tam_seri[:1913]
    while len(tam_seri) < 1913:
        tam_seri.append(0)

    # 🛑 7. GÜVENLİ CSV KAYDI (Append Mode)
    row_dict = {'item_id': hedef_id}
    for i in range(1, 1914):
        row_dict[f'd_{i}'] = tam_seri[i - 1]

    # Dosya varsa header ekleme
    file_exists = os.path.exists(dosya_adi) and os.path.getsize(dosya_adi) > 0
    pd.DataFrame([row_dict]).to_csv(
        dosya_adi,
        mode='a',
        header=not file_exists,
        index=False,
        sep=';'
    )
    print(f"   💾 {hedef_id} kaydedildi. {index + 1}/{len(emri_df)}")

    # Rate limit (Ücretsiz tier için mola)
    time.sleep(3)

print("\n" + "=" * 50)
print("🏆 TEZ İÇİN GIDA VERİLERİ (GEMINI NEW SDK) TAMAMLANDI!")
print("=" * 50)