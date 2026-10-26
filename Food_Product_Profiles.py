import pandas as pd
import numpy as np
from tqdm import tqdm

# 1. Veriyi Yükle
data_path = 'sales_train_evaluation.csv'
print("📂 Ana veri seti yükleniyor...")
df = pd.read_csv(data_path)

gun_sutunlari = [col for col in df.columns if col.startswith('d_')]


def profil_hesapla(row):
    satislar = row[gun_sutunlari].values.astype(int)
    # Ürün ne zaman rafa girdi?
    first_sale_idx = np.where(satislar > 0)[0][0] if np.any(satislar > 0) else 0
    aktif_satislar = satislar[first_sale_idx:]

    if len(aktif_satislar) == 0: return None

    sifir_orani = np.sum(aktif_satislar == 0) / len(aktif_satislar)
    return sifir_orani, first_sale_idx + 1, len(aktif_satislar), np.max(aktif_satislar), np.mean(
        aktif_satislar[aktif_satislar > 0])


print("🔍 Departman bazlı eleme ve akıllı seçim başlıyor...")


def departman_filtrele(dept_name, adet, alt_sinir, ust_sinir):
    temp_df = df[df['dept_id'] == dept_name].copy()
    valid_profiller = []

    for idx, row in temp_df.iterrows():
        res = profil_hesapla(row)
        if res:
            sifir_orani = res[0]
            # Belirlediğin sıfır oran aralığına uyuyor mu?
            if alt_sinir <= sifir_orani <= ust_sinir:
                valid_profiller.append({
                    'urun_id': row['item_id'] + "_" + row['store_id'],
                    'dept_id': dept_name,
                    'dogum_gunu': res[1],
                    'aktif_gun_sayisi': res[2],
                    'sifir_orani_%': round(sifir_orani * 100, 2),
                    'max_gunluk_satis': res[3],
                    'ort_satis_adeti': round(res[4], 2)
                })

    # Elde edilen adaylar arasından en iyi (belirlenen adet kadar) örnek seçilir
    res_df = pd.DataFrame(valid_profiller)
    if len(res_df) > adet:
        return res_df.sample(n=adet, random_state=42)
    return res_df


# --- STRATEJİK SEÇİM ARALIKLARI ---
# FOODS_3: %0 ile %20 arası sıfır (Sürekli satanlar)
f3_profiller = departman_filtrele('FOODS_3', 275, 0.0, 0.20)

# FOODS_2: %20 ile %40 arası sıfır (Orta halliler)
f2_profiller = departman_filtrele('FOODS_2', 150, 0.20, 0.40)

# FOODS_1: %40 ve üzeri sıfır (Daha seyrek ama gıda karakterinde)
f1_profiller = departman_filtrele('FOODS_1', 75, 0.40, 0.70)

# Birleştirme
food_profil_final = pd.concat([f1_profiller, f2_profiller, f3_profiller])

# Kaydet
food_profil_final.to_csv('Food_Urun_Profilleri.csv', index=False, sep=';')

print(f"\n✅ Akıllı Gıda Sepeti Hazır!")
print(f"📊 FOODS_3 (Temel): {len(f3_profiller)} ürün (Sıfır Oranı < %20)")
print(f"📊 FOODS_2 (Orta): {len(f2_profiller)} ürün (Sıfır Oranı %20-%40)")
print(f"📊 FOODS_1 (Niş): {len(f1_profiller)} ürün (Sıfır Oranı %40-%70)")
print(f"\n🚀 'Food_Urun_Profilleri.csv' oluşturuldu. Llama'yı besleyebilirsin!")