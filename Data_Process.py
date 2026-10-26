import pandas as pd

print("Devasa veri RAM'e çekiliyor, birkaç saniye bekle kanka...")
df = pd.read_csv('sales_train_validation.csv')
print("Veri başarıyla yüklendi! ✅")

# 1. Adım: 30 bin satırın tamamını eritmek yerine, SADECE bizim ürünümüzü seçelim
urun_id = 'HOBBIES_1_001'
# Bu ürünün Kaliforniya'daki (CA_1) mağazadaki verisini alalım (Tam noktasal atış)
df_bizim_urun = df[(df['item_id'] == urun_id) & (df['store_id'] == 'CA_1')]

# 2. Adım: Sadece bu tek satırlık veriyi SQL tablosu gibi aşağıya doğru eritelim (Melt)
sabit_ozellikler = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
df_melted = pd.melt(df_bizim_urun,
                    id_vars=sabit_ozellikler,
                    var_name='Gun',
                    value_name='Satis_Adedi')

# 3. Adım: Satışların sıfır (0) OLMADIĞI, yani ilk satışın başladığı günleri bulalım
satilan_gunler = df_melted[df_melted['Satis_Adedi'] > 0]

print(f"\n--- {urun_id} Kodlu Ürünün Piyasaya Çıktığı O İlk Gün! ---")
print(satilan_gunler.head(1))

# İlk satış gününün tablodaki satır numarasını (index) alıyoruz
ilk_satis_index = satilan_gunler.index[0]

# O günden başlayarak tam 30 günlük (1 aylık) dilimi koparıp alıyoruz
ilk_1_ay_gercek_veri = df_melted.iloc[ilk_satis_index : ilk_satis_index + 30]

print("\n--- İşte LLM ile Üretmeye Çalışacağımız GERÇEK İlk 1 Aylık Hedef Verimiz ---")
# Sadece günleri ve satış adetlerini yanyana şık bir şekilde basalım
print(ilk_1_ay_gercek_veri[['Gun', 'Satis_Adedi']].to_string(index=False))

# 10 kardeş ürünümüzün listesi
kardes_urunler = ['HOBBIES_1_002', 'HOBBIES_1_003', 'HOBBIES_1_004', 'HOBBIES_1_005',
                  'HOBBIES_1_006', 'HOBBIES_1_007', 'HOBBIES_1_008', 'HOBBIES_1_009',
                  'HOBBIES_1_010', 'HOBBIES_1_011']

# 1. ADIM: Ana tablodan (df) sadece bu 10 kardeşi ve CA_1 mağazasını topluca alıyoruz
df_10_kardes = df[(df['item_id'].isin(kardes_urunler)) & (df['store_id'] == 'CA_1')]

# 2. ADIM: Şimdi sadece bu 10 kardeşlik ufak grubu eritiyoruz (Bilgisayarın hiç kasmayacak)
sabit_ozellikler = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
df_kardesler_melted = pd.melt(df_10_kardes,
                              id_vars=sabit_ozellikler,
                              var_name='Gun',
                              value_name='Satis_Adedi')

import pandas as pd

print("Takvim ve 20 Ürünlük Portföy Analizi Başlıyor (Hedef Ürün HARİÇ!)...\n")

# 1. Takvim dosyasını içeri alıyoruz
df_calendar = pd.read_csv('calendar.csv')
df_calendar['date'] = pd.to_datetime(df_calendar['date'])
df_calendar['Ay'] = df_calendar['date'].dt.month
df_calendar['Yil'] = df_calendar['date'].dt.year

# 2. HEDEF ÜRÜNÜ DIŞLAMA (Veri Sızıntısını Önlüyoruz!)
tum_hobbies = df[df['item_id'].str.contains('HOBBIES')]['item_id'].unique()

# 'HOBBIES_1_001' ürününü listeden acımadan siliyoruz
kardes_urunler = [urun for urun in tum_hobbies if urun != 'HOBBIES_1_001']

# Kalan temiz listeden ilk 20 tanesini alıyoruz
urunler_20 = kardes_urunler[:20]

# Sadece bu 20 kardeşi ve CA_1 mağazasını filtreliyoruz
df_20_kardes = df[(df['item_id'].isin(urunler_20)) & (df['store_id'] == 'CA_1')]

# 3. Veriyi Eritiyoruz
sabit_ozellikler = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
df_melted_20 = pd.melt(df_20_kardes, id_vars=sabit_ozellikler, var_name='Gun', value_name='Satis_Adedi')

# 4. Takvimle Birleştirme
df_analiz = pd.merge(df_melted_20, df_calendar, left_on='Gun', right_on='d', how='left')

# --- SONUÇLARI BASTIRMA ---
print("--- 20 ÜRÜNLÜK GRUBUN AYLARA GÖRE ORTALAMA SATIŞI (Mevsimsellik) ---")
aylik_ortalama = df_analiz.groupby('Ay')['Satis_Adedi'].mean().round(2)
print(aylik_ortalama.to_string())

print("\n--- 20 ÜRÜNLÜK GRUBUN YILLARA GÖRE ORTALAMA SATIŞI (Trend) ---")
yillik_ortalama = df_analiz.groupby('Yil')['Satis_Adedi'].mean().round(2)
print(yillik_ortalama.to_string())

print("\n--- HER BİR ÜRÜN İÇİN AYLIK ORTALAMA SATIŞ MATRİSİ ---")

# Pivot table ile tabloyu harika bir ızgaraya çeviriyoruz
# Satırlar: Ürün İsimleri (item_id)
# Sütunlar: Aylar (Ay)
# Değerler: Satış Adedinin Ortalaması (mean)
urun_aylik_matris = df_analiz.pivot_table(index='item_id',
                                          columns='Ay',
                                          values='Satis_Adedi',
                                          aggfunc='mean').round(2)

print(urun_aylik_matris.to_string())