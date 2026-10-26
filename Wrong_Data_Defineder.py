import pandas as pd
import numpy as np

print("--- 🔍 Broken Data Defineder ---")
df = pd.read_csv('Akilli_Sentetik_Hobbies.csv', sep=';')

bozuk_urunler = []

# check all columns
for index, row in df.iterrows():
    urun_id = row['item_id']
    # d_1'den d_1913'e kadar olan satışları al
    satislar = row.iloc[1:].values

    # Satışların içinde 50'den büyük GİRİLMEZ bir sayı var mı?
    # (Hobbies kategorisinde bir günde 50'den fazla satış imkansızdır)
    if any(pd.to_numeric(s, errors='coerce') > 50 for s in satislar):
        bozuk_urunler.append(urun_id)

print("--- ACCURACY INCREASER ---")

# Dosyayı okuyoruz
df = pd.read_csv('Akilli_Sentetik_Hobbies.csv', sep=';')

# Sadece günlerin olduğu (d_1, d_2...) sütunları seçiyoruz
gun_kolonlari = df.columns[1:]

print("HATALI FORMATI 0 A DÖNÜŞTRDM ")

# Bütün gün kolonlarını sayıya çevir, harf/bozuk format varsa direkt NaN (boş) yap
df[gun_kolonlari] = df[gun_kolonlari].apply(pd.to_numeric, errors='coerce')

# Eğer bir hücre NaN (bozuk) ise veya 50'den büyükse (imkansız satış), onu 0 yap. Değilse kendi değerini koru.
df[gun_kolonlari] = df[gun_kolonlari].where((df[gun_kolonlari] <= 50) & (df[gun_kolonlari].notna()), 0)

# Dosyayı aynı isimle, iyileşmiş haliyle kaydediyoruz
df.to_csv('Akilli_Sentetik_Hobbies.csv', index=False, sep=';')

print("✅ Bitti! ALL DATA CLEANED.")

print(f"\n Toplam {len(df)} ürün içinde {len(bozuk_urunler)} adet ürün BOZUK üretilmiş!")

if len(bozuk_urunler) > 0:
    print("Bozuk Ürünlerin Listesi:")
    for u in bozuk_urunler:
        print(f"- {u}")
else:
    print("✅ Bütün veriler SAĞLAMLAŞTIRILDI !")