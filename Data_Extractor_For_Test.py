import pandas as pd

# 1. Elit 500'lük listeyi yükle
# Not: Bir önceki kodda 'urun_id' olarak kaydettiğimiz için sütun adı 'urun_id' olmalı
df = pd.read_csv('Food_Urun_Profilleri.csv', sep=';')

print(f"📊 Dosya yüklendi. Mevcut sütunlar: {df.columns.tolist()}")

# 2. Departman bazlı 100 tanesini "Sınav" için ayır (33-33-34 kuralı)
# Sütun adını 'item_id' yerine 'urun_id' olarak güncelledim
test_f1 = df[df['dept_id'] == 'FOODS_1'].sample(n=33, random_state=42)
test_f2 = df[df['dept_id'] == 'FOODS_2'].sample(n=33, random_state=42)
test_f3 = df[df['dept_id'] == 'FOODS_3'].sample(n=34, random_state=42)

food_test_profilleri = pd.concat([test_f1, test_f2, test_f3])

# 3. Kalan 400 tanesini "Eğitim" için ayır
# Burada da 'urun_id' kullanıyoruz
food_train_profilleri = df[~df['urun_id'].isin(food_test_profilleri['urun_id'])]

# 4. Kaydet
food_test_profilleri.to_csv('Food_Test_Profilleri_100.csv', index=False, sep=';')
food_train_profilleri.to_csv('Food_Llama_Uretim_Listesi_400.csv', index=False, sep=';')

print(f"\n✅ Hata giderildi!")
print(f"🎯 Llama'nın üreteceği (Eğitim): {len(food_train_profilleri)} ürün")
print(f"📝 Sınavda soracağımız (Test): {len(food_test_profilleri)} ürün")