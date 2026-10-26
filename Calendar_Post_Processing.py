import pandas as pd

print("--- ✂️ TAKVİM BUDAMASI (POST-PROCESSING) BAŞLIYOR ---")

# 1. Dosyaları Okuma
print("Veriler yükleniyor (Bu işlem birkaç saniye sürebilir)...")
# Gemini'nin sabaha kadar ürettiği ham verimiz
df_syn = pd.read_csv('Akilli_Sentetik_Hobbies.csv', sep=';')
# Orijinal M5 takvim dosyamız
df_cal = pd.read_csv('calendar.csv')

# 2. Kapalı Günleri (Christmas) Tespit Etme
print("Takvim analiz ediliyor...")
# 'event_name_1' kolonunda Christmas yazan satırları buluyoruz
kapali_gunler_df = df_cal[df_cal['event_name_1'] == 'Christmas']

# Bu günlerin 'd_...' formatındaki ID'lerini bir listeye alıyoruz
kapali_d_listesi = kapali_gunler_df['d'].tolist()

print(f"Takvimden tespit edilen kapalı gün (Noel) sayısı: {len(kapali_d_listesi)}")
print(f"Kapalı Gün Kodları: {kapali_d_listesi}")

# 3. Budama İşlemi (Cerrahi Müdahale)
print("Sentetik verideki Noel günleri tespit edilip satışlar '0' yapılıyor...")
for d_gun in kapali_d_listesi:
    # Eğer o gün bizim 1913 günlük sütunlarımız arasında varsa sıfırla
    if d_gun in df_syn.columns:
        df_syn[d_gun] = 0

# 4. Yeni Dosyayı Kaydetme
yeni_dosya_adi = 'Sentetik_Hobbies_Processed.csv'
print("Dosya kaydediliyor...")
df_syn.to_csv(yeni_dosya_adi, index=False, sep=';')

print(f"\n✅ İşlem Kusursuz Tamamlandı!")
print(f"Temizlenmiş yeni verin '{yeni_dosya_adi}' adıyla klasörüne eklendi.")