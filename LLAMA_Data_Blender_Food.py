import pandas as pd
import numpy as np
import os

print("--- 🏗️ FOOD VERİ BLENDER VE AUGMENTATION OPERASYONU BAŞLIYOR ---")

# 1. Dosya Yolları (Karantinaya alınmış test hariç, doğrudan senin 400'lük dosyaların)
real_file = 'Food_Gercek_Satislar_Train_400.csv'
syn_file = 'Sentetik_Gida_Verileri_Llama.csv'
calendar_file = 'calendar.csv'
prices_file = 'sell_prices.csv'

# 2. Otomatik Ayraç ve UTF-8 Zırhı İle Dosyaları Yükleme
print("Dosyalar yükleniyor ve temizleniyor...")
df_real_wide = pd.read_csv(real_file, sep=None, engine='python', encoding='utf-8-sig')
df_syn_wide = pd.read_csv(syn_file, sep=None, engine='python', encoding='utf-8-sig')
df_calendar = pd.read_csv(calendar_file, sep=None, engine='python', encoding='utf-8-sig')
df_prices = pd.read_csv(prices_file, sep=None, engine='python', encoding='utf-8-sig')

# Sütun isimlerindeki gizli boşlukları kes ve küçük harfe zorla
df_real_wide.columns = df_real_wide.columns.str.strip().str.lower()
df_syn_wide.columns = df_syn_wide.columns.str.strip().str.lower()
df_calendar.columns = df_calendar.columns.str.strip().str.lower()
df_prices.columns = df_prices.columns.str.strip().str.lower()

df_calendar_sub = df_calendar[['d', 'wm_yr_wk']]


# 3. Long Format (Melt) ve Fiyat/Takvim Birleştirme Motoru
def process_to_long_format(df_wide):
    # Günleri alt alta diz
    id_vars = [col for col in df_wide.columns if not col.startswith('d_')]
    df_melt = df_wide.melt(id_vars=id_vars, var_name='day', value_name='sales')

    # item_id ve store_id parçalama zırhı
    base_col = 'id' if 'id' in df_melt.columns else ('item_id' if 'item_id' in df_melt.columns else df_melt.columns[0])

    if 'item_id' not in df_melt.columns:
        df_melt['item_id'] = df_melt[base_col].astype(str).apply(lambda x: "_".join(x.split("_")[:3]))

    if 'store_id' not in df_melt.columns:
        df_melt['store_id'] = df_melt[base_col].astype(str).apply(
            lambda x: "_".join(x.split("_")[3:5]) if len(x.split("_")) >= 5 else "TX_1"
        )

    # Takvim ve Fiyatları ekle
    df_merged = df_melt.merge(df_calendar_sub, left_on='day', right_on='d', how='left')
    df_final = df_merged.merge(df_prices, on=['item_id', 'store_id', 'wm_yr_wk'], how='left')

    # Modelin ihtiyacı olan nihai sütunlar
    df_clean = df_final[['item_id', 'day', 'sales', 'sell_price']].copy()
    df_clean['sell_price'] = df_clean['sell_price'].fillna(0)

    return df_clean


# 4. Gerçek ve Sentetik Verileri İşleme Sokma
print("Gerçek 400 ürün dikey (long) formata çevriliyor...")
df_train_baseline = process_to_long_format(df_real_wide)

print("Sentetik 400 ürün dikey (long) formata çevriliyor...")
df_syn_long = process_to_long_format(df_syn_wide)

# Sentetik ürünlerin benzersiz ID'lerini sıralı şekilde al
syn_unique_items = sorted(df_syn_long['item_id'].unique())

# 5. Dosyaları Oluşturma ve Proje Klasörüne Yazma
# Dosya yollarını doğrudan kodun çalıştığı PyCharm klasörüne ayarlıyoruz
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

print("⏳ 1. Dosya: Baseline diske yazılıyor...")
df_train_baseline.to_csv(os.path.join(PROJECT_DIR, "1_Food_Train_Baseline_0_Sentetik.csv"), index=False, sep=';')
print("✅ 1_Food_Train_Baseline_0_Sentetik.csv oluşturuldu.")

print("⏳ 2. Dosya: %25 Sentetik veri ekleniyor...")
# 400 ürünün %25'i olan ilk 100 sentetik ürünü alıyoruz
syn_25_ids = syn_unique_items[:100]
df_syn_25 = df_syn_long[df_syn_long['item_id'].isin(syn_25_ids)]
df_train_25 = pd.concat([df_train_baseline, df_syn_25], ignore_index=True)
df_train_25.to_csv(os.path.join(PROJECT_DIR, "2_Food_Train_Llama_Mix_25.csv"), index=False, sep=';')
print("✅ 2_Food_Train_Llama_Mix_25.csv oluşturuldu.")

print("⏳ 3. Dosya: %50 Sentetik veri ekleniyor...")
# 400 ürünün %50'si olan ilk 200 sentetik ürünü alıyoruz
syn_50_ids = syn_unique_items[:200]
df_syn_50 = df_syn_long[df_syn_long['item_id'].isin(syn_50_ids)]
df_train_50 = pd.concat([df_train_baseline, df_syn_50], ignore_index=True)
df_train_50.to_csv(os.path.join(PROJECT_DIR, "3_Food_Train_Llama_Mix_50.csv"), index=False, sep=';')
print("✅ 3_Food_Train_Llama_Mix_50.csv oluşturuldu.")

print("⏳ 4. Dosya: %100 Sentetik veri ekleniyor...")
# Bütün sentetik veriyi ekliyoruz
df_train_100 = pd.concat([df_train_baseline, df_syn_long], ignore_index=True)
df_train_100.to_csv(os.path.join(PROJECT_DIR, "4_Food_Train_Llama_Mix_100.csv"), index=False, sep=';')
print("✅ 4_Food_Train_Llama_Mix_100.csv oluşturuldu.")

print("\n🚀 BÜTÜN ALTYAPI HAZIR! Proje klasörünü kontrol et, model eğitimine geçebiliriz.")