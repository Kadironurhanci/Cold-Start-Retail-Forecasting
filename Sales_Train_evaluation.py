import pandas as pd

# 1. Ana veriyi ve bizim listeleri yükle
df_ana = pd.read_csv('sales_train_evaluation.csv')
train_list = pd.read_csv('Food_Llama_Uretim_Listesi_400.csv', sep=';')
test_list = pd.read_csv('Food_Test_Profilleri_100.csv', sep=';')

# 2. Ürün ID'lerini ayrıştır (urun_id formatımız: item_id + "_" + store_id)
# Ana tabloda bunlar ayrı sütunlarda olduğu için birleştirip filtreleme yapmalıyız
df_ana['urun_id_full'] = df_ana['item_id'] + "_" + df_ana['store_id']

# 3. Sadece bizim 500 ürünü çek
secilen_id_listesi = list(train_list['urun_id']) + list(test_list['urun_id'])
food_gercek_veriler = df_ana[df_ana['urun_id_full'].isin(secilen_id_listesi)]

# 4. Kaydet
food_gercek_veriler.to_csv('Food_Gercek_Satis_Verileri_500.csv', index=False, sep=';')
print(f"✅ 500 ürünün gerçek günlük satışları 'Food_Gercek_Satis_Verileri_500.csv' olarak kaydedildi!")

# 1. Ana veriyi ve listelerini yükle
df_ana = pd.read_csv('sales_train_evaluation.csv')
train_list = pd.read_csv('Food_Llama_Uretim_Listesi_400.csv', sep=';')
test_list = pd.read_csv('Food_Test_Profilleri_100.csv', sep=';')

# 2. Ürün ID'lerini eşleştir
df_ana['urun_id_full'] = df_ana['item_id'] + "_" + df_ana['store_id']

# 3. 400 Eğitim ve 100 Test verisini ayır (Tüm günleriyle birlikte)
food_train_gercek = df_ana[df_ana['urun_id_full'].isin(train_list['urun_id'])]
food_test_gercek = df_ana[df_ana['urun_id_full'].isin(test_list['urun_id'])]

# 4. Kaydet
food_train_gercek.to_csv('Food_Gercek_Satislar_Train_400.csv', index=False, sep=';')
food_test_gercek.to_csv('Food_Gercek_Satislar_Test_100.csv', index=False, sep=';')

print("✅ Günlük veriler ayrıştırıldı!")