import pandas as pd
import numpy as np

print("--- 🔍 HOBBIES Veri Profilleme Motoru Başlatıldı ---")

# 1. Orijinal veriyi yüklüyoruz
print("Orijinal M5 verisi okunuyor (sales_train_validation.csv)...")
df = pd.read_csv('sales_train_validation.csv')

# 2. Sadece HOBBIES kategorisini ve CA_1 mağazasını filtreliyoruz
# İstersen tüm mağazaları da yapabilirsin ama şimdilik hızlı test için CA_1
hobbies_df = df[(df['cat_id'] == 'HOBBIES') & (df['store_id'] == 'CA_1')].copy()

# Gün sütunlarını buluyoruz (d_1, d_2 ... d_1913)
gun_sutunlari = [col for col in df.columns if col.startswith('d_')]

urun_profilleri = []

print(f"\n📦 Toplam {len(hobbies_df)} HOBBIES ürünü analiz ediliyor...")

# 3. Her bir ürünün genetiğini çıkarıyoruz
for index, row in hobbies_df.iterrows():
    urun_id = row['id']
    satislar = row[gun_sutunlari].values.astype(int)

    # Ürünün ilk satıldığı günü (doğum gününü) buluyoruz
    # satislar > 0 olan ilk index'i bulur
    ilk_satis_index = np.argmax(satislar > 0)

    # Eğer ürün HİÇ satılmamışsa (hepsi 0 ise) atla
    if ilk_satis_index == 0 and satislar[0] == 0:
        continue

    dogum_gunu = ilk_satis_index + 1  # (index 0 = d_1 olduğu için +1)

    # Aktif yaşam süresini (doğum gününden 1913'e kadar) alıyoruz
    aktif_satis_donemi = satislar[ilk_satis_index:]
    aktif_gun_sayisi = len(aktif_satis_donemi)

    # AKTİF dönemdeki sıfır (satışsızlık) oranını hesaplıyoruz
    sifir_sayisi = np.sum(aktif_satis_donemi == 0)
    sifir_orani = round((sifir_sayisi / aktif_gun_sayisi) * 100, 2)

    # Maksimum satış ve ortalama satış (sıfırlar hariç)
    satilan_gunler = aktif_satis_donemi[aktif_satis_donemi > 0]
    max_satis = satilan_gunler.max() if len(satilan_gunler) > 0 else 0
    ortalama_satis = round(satilan_gunler.mean(), 2) if len(satilan_gunler) > 0 else 0

    urun_profilleri.append({
        'urun_id': urun_id,
        'dogum_gunu': dogum_gunu,
        'aktif_gun_sayisi': aktif_gun_sayisi,
        'sifir_orani_%': sifir_orani,
        'max_gunluk_satis': max_satis,
        'ort_satis_adeti': ortalama_satis
    })

# 4. Sonuçları bir DataFrame'e çevirip kaydedelim
profil_df = pd.DataFrame(urun_profilleri)
# Sayıları virgüllü formata çevir ki Türkçe Excel kafayı yemesin
profil_df['ort_satis_adeti'] = profil_df['ort_satis_adeti'].astype(str).str.replace('.', ',')

profil_df.to_csv('Hobbies_Urun_Profilleri.csv', index=False, sep=';')
profil_df.to_csv('Hobbies_Urun_Profilleri.csv', index=False, sep=';')

print("\n✅ Analiz Tamamlandı!")
print("-" * 50)
print("Örnek 5 Ürün Profili:")
print(profil_df.head(5).to_string(index=False))
print("-" * 50)
print("Tüm detaylar 'Hobbies_Urun_Profilleri.csv' dosyasına kaydedildi.")