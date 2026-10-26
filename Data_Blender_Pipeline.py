import pandas as pd
import numpy as np
import os

print("--- 🏗️ VERİ İZOLASYON VE BLENDER OPERASYONU BAŞLIYOR ---")

# 1. Dosya Yolları
sales_file = 'sales_train_validation.csv'
calendar_file = 'calendar.csv'
prices_file = 'sell_prices.csv'
sentetik_file = 'Sentetik_Hobbies_Final_Dataset.csv'

# 2. Sentetik Veriyi Yükle ve Hedefleri Belirle
print("Sentetik veriler analiz ediliyor...")
df_syn = pd.read_csv(sentetik_file, sep=';')
syn_items = sorted(df_syn['item_id'].unique())

# Gerçek ID'leri bul (Sonundaki '_SYN' takısını atarak)
# Not: id'ler genelde 'HOBBIES_1_001_CA_1_validation' formatındadır
real_items = [x.replace('_SYN', '') for x in syn_items]

# H1 ve H2 olarak ayır
h1_real = [x for x in real_items if 'HOBBIES_1' in x]
h2_real = [x for x in real_items if 'HOBBIES_2' in x]

# 3. KARANTİNA (Test Seti Ayırımı - Asla eğitime girmeyecekler)
# Son 70 H1 ve Son 30 H2
test_h1 = h1_real[-70:]
test_h2 = h2_real[-30:]
test_ids = test_h1 + test_h2

# Eğitim Havuzu (Geriye kalan 465 gerçek ürün)
train_h1_real = h1_real[:-70]
train_h2_real = h2_real[:-30]
train_real_ids = train_h1_real + train_h2_real

print(f"Karantinaya Alınan Test Ürünü: {len(test_ids)} (70 H1, 30 H2)")
print(f"Eğitim Havuzunda Kalan Gerçek Ürün: {len(train_real_ids)}")

# 4. Orijinal M5 Verisinden Sadece Hedef 565 Ürünü Çek
print("M5 Orijinal verisinden gerçek satışlar cımbızlanıyor (Biraz sürebilir)...")
df_sales = pd.read_csv(sales_file)
df_target_real = df_sales[df_sales['id'].isin(real_items)].copy()

print("Gerçek veri Long Format'a (Melt) çevriliyor...")
id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
df_melted = df_target_real.melt(id_vars=id_vars, var_name='day', value_name='sales')

print("Gerçek veriye Takvim ve Fiyatlar ekleniyor...")
df_calendar = pd.read_csv(calendar_file)[['d', 'wm_yr_wk']]
df_prices = pd.read_csv(prices_file)

df_merged = df_melted.merge(df_calendar, left_on='day', right_on='d', how='left')
df_real_final = df_merged.merge(df_prices, on=['item_id', 'store_id', 'wm_yr_wk'], how='left')

# Sütunları sentetik veriyle aynı formata getir
df_real_final = df_real_final[['id', 'day', 'sales', 'sell_price']].rename(columns={'id': 'item_id'})
df_real_final['sell_price'] = df_real_final['sell_price'].fillna(0)

# 5. DOSYALARI OLUŞTURMA VE BÖLME
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "Tez_Veri_Setleri")
if not os.path.exists(desktop_path):
    os.makedirs(desktop_path)

# A) TEST SETİ OLUŞTUR (Sadece 100 gerçek ürün)
df_test = df_real_final[df_real_final['item_id'].isin(test_ids)]
df_test.to_csv(os.path.join(desktop_path, "1_Test_Set_100_Gercek.csv"), index=False, sep=';')
print("✅ 1_Test_Set_100_Gercek.csv oluşturuldu.")

# B) BASELINE TRAIN (Model 1 - Sadece 465 Gerçek Ürün)
df_train_baseline = df_real_final[df_real_final['item_id'].isin(train_real_ids)]
df_train_baseline.to_csv(os.path.join(desktop_path, "2_Train_Baseline_0_Sentetik.csv"), index=False, sep=';')
print("✅ 2_Train_Baseline_0_Sentetik.csv oluşturuldu.")

# C) VERİ ÇOĞALTMA (Data Augmentation) İÇİN SENTETİK SEÇİMİ
# Sadece Eğitim havuzundaki gerçek ürünlerin sentetik karşılıklarını alabiliriz
train_h1_syn = [x + '_SYN' for x in train_h1_real]
train_h2_syn = [x + '_SYN' for x in train_h2_real]

# %25 Miktar: 86 H1, 30 H2
syn_25_ids = train_h1_syn[:86] + train_h2_syn[:30]
df_syn_25 = df_syn[df_syn['item_id'].isin(syn_25_ids)]
df_train_25 = pd.concat([df_train_baseline, df_syn_25], ignore_index=True)
df_train_25.to_csv(os.path.join(desktop_path, "3_Train_Mix_25.csv"), index=False, sep=';')
print("✅ 3_Train_Mix_25.csv oluşturuldu.")

# %50 Miktar: 173 H1, 60 H2
syn_50_ids = train_h1_syn[:173] + train_h2_syn[:60]
df_syn_50 = df_syn[df_syn['item_id'].isin(syn_50_ids)]
df_train_50 = pd.concat([df_train_baseline, df_syn_50], ignore_index=True)
df_train_50.to_csv(os.path.join(desktop_path, "4_Train_Mix_50.csv"), index=False, sep=';')
print("✅ 4_Train_Mix_50.csv oluşturuldu.")

# %100 Miktar: Tamamı (346 H1, 119 H2)
syn_100_ids = train_h1_syn + train_h2_syn
df_syn_100 = df_syn[df_syn['item_id'].isin(syn_100_ids)]
df_train_100 = pd.concat([df_train_baseline, df_syn_100], ignore_index=True)
df_train_100.to_csv(os.path.join(desktop_path, "5_Train_Mix_100.csv"), index=False, sep=';')
print("✅ 5_Train_Mix_100.csv oluşturuldu.")

print("\n🚀 BÜTÜN ALTYAPI HAZIR! Tez_Veri_Setleri klasörünü kontrol et.")