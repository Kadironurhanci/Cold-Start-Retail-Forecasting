import pandas as pd
import numpy as np
import os

print("--- 🏗️ KESİN ÇÖZÜM: GERÇEK AUGMENTATION (VERİ KATLAMA) ---")

# 1. Dosya Yolları
real_file = 'Food_Gercek_Satislar_Train_400.csv'
syn_file = 'Sentetik_Gida_Verileri_Gemini_Full.csv'
calendar_file = 'calendar.csv'
prices_file = 'sell_prices.csv'

# 2. DOSYALARI YÜKLE VE SÜTUNLARI TEMİZLE
print("Dosyalar yükleniyor (Otomatik ayraç ve UTF-8 kalkanı devrede)...")
df_real_wide = pd.read_csv(real_file, sep=None, engine='python', encoding='utf-8-sig')
df_syn_wide = pd.read_csv(syn_file, sep=None, engine='python', encoding='utf-8-sig')
df_calendar = pd.read_csv(calendar_file, sep=None, engine='python', encoding='utf-8-sig')
df_prices = pd.read_csv(prices_file, sep=None, engine='python', encoding='utf-8-sig')

df_real_wide.columns = df_real_wide.columns.str.strip().str.lower()
df_syn_wide.columns = df_syn_wide.columns.str.strip().str.lower()
df_calendar.columns = df_calendar.columns.str.strip().str.lower()
df_prices.columns = df_prices.columns.str.strip().str.lower()

df_calendar_sub = df_calendar[['d', 'wm_yr_wk']]


# 3. İŞTE HAYAT KURTARAN DÖNÜŞÜM MOTORU (İkisini de Dikey Formata Çevirecek)
def process_to_long_format(df_wide):
    # d_ ile başlamayanları ID kabul et ve günleri alt alta diz
    id_vars = [col for col in df_wide.columns if not col.startswith('d_')]
    df_melt = df_wide.melt(id_vars=id_vars, var_name='day', value_name='sales')

    # ZIRH: Hangi sütunda o uzun 'FOODS_1_020_TX_3' yazısı var onu bulalım
    base_col = 'id' if 'id' in df_melt.columns else ('item_id' if 'item_id' in df_melt.columns else df_melt.columns[0])

    # item_id ve store_id yarat (Güvenli Parçalama)
    if 'item_id' not in df_melt.columns:
        df_melt['item_id'] = df_melt[base_col].astype(str).apply(lambda x: "_".join(x.split("_")[:3]))

    if 'store_id' not in df_melt.columns:
        # Uzunluk kontrolü yapıyoruz ki sentetik veride kısa isim varsa patlamasın
        df_melt['store_id'] = df_melt[base_col].astype(str).apply(
            lambda x: "_".join(x.split("_")[3:5]) if len(x.split("_")) >= 5 else "TX_1"
        )  # Eğer store bulamazsa varsayılan olarak TX_1 atar, birleştirme çökmez.

    # Takvim ve Fiyat ekle
    df_merged = df_melt.merge(df_calendar_sub, left_on='day', right_on='d', how='left')
    df_final = df_merged.merge(df_prices, on=['item_id', 'store_id', 'wm_yr_wk'], how='left')

    # Sadece modelin ihtiyacı olanları al ve temizle
    df_clean = df_final[['item_id', 'day', 'sales', 'sell_price']].copy()
    df_clean['sell_price'] = df_clean['sell_price'].fillna(0)

    return df_clean


# 4. GERÇEK VE SENTETİK VERİLERİ FABRİKAYA SOKALIM
print("Gerçek 400 ürün dikey (long) formata çevriliyor...")
df_train_baseline = process_to_long_format(df_real_wide)

print("Sentetik 400 ürün dikey (long) formata çevriliyor (Önceki hatamız burasıydı!)...")
df_syn_long = process_to_long_format(df_syn_wide)
syn_unique_items = sorted(df_syn_long['item_id'].unique())

# 5. DOSYALARI OLUŞTURMA VE YAZMA (ARTIK GERÇEKTEN ÇOĞALACAK)
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "Tez_Modelleri_V2")
if not os.path.exists(desktop_path):
    os.makedirs(desktop_path)

# A) BASELINE TRAIN
print("⏳ 1. Dosya: Baseline diske yazılıyor (~765 Bin Satır)...")
df_train_baseline.to_csv(os.path.join(desktop_path, "1_Food_Train_Baseline_0_Sentetik.csv"), index=False, sep=';')
print("✅ 1_Food_Train_Baseline_0_Sentetik.csv oluşturuldu.")

# B) %25 MİKTAR
print("⏳ 2. Dosya: %25 Sentetik veri ekleniyor (~950 Bin Satır olacak)...")
syn_25_ids = syn_unique_items[:100]
df_syn_25 = df_syn_long[df_syn_long['item_id'].isin(syn_25_ids)]
df_train_25 = pd.concat([df_train_baseline, df_syn_25], ignore_index=True)
df_train_25.to_csv(os.path.join(desktop_path, "2_Food_Train_Mix_25.csv"), index=False, sep=';')
print("✅ 2_Food_Train_Mix_25.csv oluşturuldu.")

# C) %50 MİKTAR
print("⏳ 3. Dosya: %50 Sentetik veri ekleniyor (~1.1 Milyon Satır olacak)...")
syn_50_ids = syn_unique_items[:200]
df_syn_50 = df_syn_long[df_syn_long['item_id'].isin(syn_50_ids)]
df_train_50 = pd.concat([df_train_baseline, df_syn_50], ignore_index=True)
df_train_50.to_csv(os.path.join(desktop_path, "3_Food_Train_Mix_50.csv"), index=False, sep=';')
print("✅ 3_Food_Train_Mix_50.csv oluşturuldu.")

# D) %100 MİKTAR
print("⏳ 4. Dosya: %100 Sentetik veri ekleniyor (~1.5 Milyon Satır olacak)...")
df_train_100 = pd.concat([df_train_baseline, df_syn_long], ignore_index=True)
df_train_100.to_csv(os.path.join(desktop_path, "4_Food_Train_Mix_100.csv"), index=False, sep=';')
print("✅ 4_Food_Train_Mix_100.csv oluşturuldu.")

print("\n🚀 BÜTÜN DOSYALAR GERÇEK ANLAMDA ÇOĞALTILDI! Kanka klasörü kontrol et ve satır sayılarına bak, bu sefer jilet!")