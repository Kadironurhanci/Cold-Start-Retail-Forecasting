import pandas as pd
import numpy as np

print("- FİYAT EŞLEŞTİRME VE VERİ BİRLEŞTİRME BAŞLIYOR -")

# 1. Dosyaları YÜKLEYECEİZ
print("FİYAT BELİRRLEME DOSYALARINI YÜKLEME ZAMANI")
df_syn = pd.read_csv('Sentetik_Hobbies_Processed.csv', sep=';')
df_cal = pd.read_csv('calendar.csv')
df_prices = pd.read_csv('sell_prices.csv')

# 2. Veriyi MODELLERİN okuyabileceği FORMAT a çekiyorum
print("Veri formatı LSTM için optimize ediliyor (Wide to Long)...")
df_long = pd.melt(df_syn, id_vars=['item_id'], var_name='d', value_name='sales')

# 3. takvimle birleştiriyoruz ki olası hata yapmasın program
print("Takvim ve hafta bilgileri ekleniyor...")
df_cal_min = df_cal[['d', 'wm_yr_wk', 'event_name_1']]
df_merged = pd.merge(df_long, df_cal_min, on='d', how='left')

# 4. fiyatları ürünlerle birleştiriyoruz
print("Gerçek satış fiyatları eşleştiriliyor...")

def clean_id(x):
    return x.replace('_SYN', '').replace('_validation', '')

df_merged['original_item_id'] = df_merged['original_item_id'] = df_merged['item_id'].apply(clean_id)
df_merged['store_id'] = df_merged['original_item_id'].str[-4:]
df_merged['base_item_id'] = df_merged['original_item_id'].str[:-5]

df_final = pd.merge(
    df_merged,
    df_prices,
    left_on=['base_item_id', 'store_id', 'wm_yr_wk'],
    right_on=['item_id', 'store_id', 'wm_yr_wk'],
    how='left'
)

df_final = df_final[['item_id_x', 'd', 'sales', 'wm_yr_wk', 'sell_price', 'event_name_1']]
df_final.columns = ['item_id', 'day', 'sales', 'wm_yr_wk', 'sell_price', 'event_name']
df_final['sell_price'] = df_final['sell_price'].fillna(0)

# ---------------------------------------------------------
#  YEDİRİLEN KISIM: İKİ FARKLI KAYDETME STRATEJİSİ
# ---------------------------------------------------------

# A) LSTM İÇİN TEKNİK DOSYA (Noktalı - Python'ın anladığı dil)
print(f"\n1. Adım: LSTM için teknik dosya (noktalı) kaydediliyor...")
df_final.to_csv('Sentetik_Hobbies_Final_Dataset.csv', index=False, sep=';')

# B) EXCEL İÇİN GÖRÜNÜM DOSYASI (Virgüllü - İnsanların anladığı dil)
print(f"2. Adım: Excel için görünüm dosyası (virgüllü) hazırlanıyor...")

# Sayısal değerleri stringe çevirip noktaları virgüle dönüştürüyoruz
df_final['sell_price'] = df_final['sell_price'].astype(str).str.replace('.', ',', regex=False)
df_final['sales'] = df_final['sales'].astype(str).str.replace('.', ',', regex=False)

df_final.to_csv('Sentetik_Hobbies_Excel_Gorunumu.csv', index=False, sep=';', encoding='utf-8-sig')

print("\n İki dosya da başarıyla oluşturuldu.")
print("- 'Sentetik_Hobbies_Final_Dataset.csv' -> LSTM eğitimi icin.")
print("- 'Sentetik_Hobbies_Excel_Gorunumu.csv' -> Excel ile kontrol etmek için.")