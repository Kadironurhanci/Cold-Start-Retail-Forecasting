import pandas as pd
import numpy as np
import os

print("m5 veri setinden ihtiyacımız olanı çekme işi ")

# 1. dosya yolu tanımlama
sales_file = 'sales_train_validation.csv'
calendar_file = 'calendar.csv'
prices_file = 'sell_prices.csv'

# 2. satış verilerini getir
print("Satış verisi yükleniyor")
df_sales = pd.read_csv(sales_file)

# hobbies kısımlarını alalım
df_hobbies = df_sales[df_sales['dept_id'].isin(['HOBBIES_1', 'HOBBIES_2'])].copy()
print(f"tamamdır : {len(df_hobbies)} adet gerçek Hobbies ürünü bulundu.")

# 3. melt işlemi
# day sütununa indirgicez ki her ürünü model farklı sanmasın
print("Veri 'Long Format'a dönüştürülüyor (Melt işlemi)...")
id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
df_melted = df_hobbies.melt(id_vars=id_vars, var_name='day', value_name='sales')

# 4. Fiyatları ve Takvimi Giydirme
print("Takvim ve Fiyat bilgileri birleştiriliyor...")
df_calendar = pd.read_csv(calendar_file)[['d', 'wm_yr_wk']]
df_prices = pd.read_csv(prices_file)

# Önce takvimle birleştir (wm_yr_wk bilgisini almak için)
df_merged = df_melted.merge(df_calendar, left_on='day', right_on='d', how='left')

# Sonra fiyatlarla birleştir (item_id, store_id ve wm_yr_wk üzerinden)
df_final = df_merged.merge(df_prices, on=['item_id', 'store_id', 'wm_yr_wk'], how='left')

# 5. Temizlik ve Kayıt
# Sadece ihtiyacımız olan sütunları tutalım (Sentetik veriyle aynı olsun)
cols_to_keep = ['item_id', 'day', 'sales', 'sell_price']
df_final = df_final[cols_to_keep]

# Fiyatı olmayan (henüz rafa çıkmamış) ise 0 yapıcam
df_final['sell_price'] = df_final['sell_price'].fillna(0)

# Masaüstüne Kaydet
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
output_path = os.path.join(desktop_path, "Gercek_Hobbies_Dataset.csv")

print(f"Final dosyası oluşturuluyor...")
df_final.to_csv(output_path, index=False, sep=';')

print(f"\n OPERASYON BAŞARILI!")
print(f"Gerçek Hobbies verisi (Melt edilmiş halde) şuraya kaydedildi:\n{output_path}")