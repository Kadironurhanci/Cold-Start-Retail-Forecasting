import pandas as pd
import numpy as np
import os, re, time, random
from groq import Groq

# 🛑 1. REPRODUCIBILITY & API SETUP
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

client = Groq(api_key="")
PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"

# 2. Üretim Emrini Yükle
emri_df = pd.read_csv('Llama_Gida_Uretim_Emri_400.csv', sep=';')
dosya_adi = 'Sentetik_Gida_Verileri_Llama.csv'

# 🛑 3. PERFORMANS FIX: Mevcut ID'leri Loop Dışında Bir Kez Oku
existing_ids = set()
if os.path.exists(dosya_adi) and os.path.getsize(dosya_adi) > 0:
    check_df = pd.read_csv(dosya_adi, sep=';', usecols=['item_id'])
    existing_ids = set(check_df['item_id'].astype(str).unique())

print(f"🚀 Gıda Fabrikası Başladı! Hedef: {len(emri_df)} ürün. (Atlanacak: {len(existing_ids)})")

for index, row in emri_df.iterrows():
    hedef_id = str(row['urun_id'])

    if hedef_id in existing_ids:
        continue

    dogum_gunu = int(row['dogum_gunu'])
    hedef_sifir = row['hedef_sifir_orani']
    ort_satis = row['ort_satis_adeti']
    max_satis = row['max_gunluk_satis']

    # 🛑 4. GÜVENLİ GÜN HESABI (Off-by-one bug fix)
    aktif_gun_sayisi = max(0, min(1913, 1913 - dogum_gunu + 1))

    print(f"\n📦 {hedef_id} üretiliyor... (Doğum: d_{dogum_gunu}, Aktif: {aktif_gun_sayisi} gün)")

    llama_verisi = []
    aktif_model = PRIMARY_MODEL

    while len(llama_verisi) < aktif_gun_sayisi:
        eksik = aktif_gun_sayisi - len(llama_verisi)
        # 🛑 5. STABİLİTE: Batch size 200'e düşürüldü (Token kesilme riski sıfırlandı)
        istenecek = min(eksik, 200)

        wait_time = 5  # Exponential backoff başlangıcı
        for attempt in range(3):  # 3 deneme hakkı
            try:
                prompt = f"Sen Walmart Gıda Müdürüsün. Bana tam olarak {istenecek} adet günlük satış sayısı üret. Kurallar: Ort {ort_satis}, Max {max_satis}, Sıfır %{hedef_sifir}. Sadece virgülle ayır."

                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=aktif_model,
                    temperature=0.85,
                    max_tokens=1500  # 200 sayı için yeterli ve güvenli
                )

                content = response.choices[0].message.content
                # 🛑 6. REGEX FIX: Sadece sayıları tek tek ayıkla (Hatalı birleşmeleri önler)
                parca = re.findall(r'\d+', content)

                if len(parca) >= (istenecek * 0.8):  # Verinin çoğu geldiyse kabul et
                    llama_verisi.extend(parca)
                    print(f"   [+] {len(parca)} gün eklendi. ({len(llama_verisi)}/{aktif_gun_sayisi})")
                    time.sleep(1)  # Normal hız
                    break
                else:
                    raise ValueError("Eksik veri geldi")

            except Exception as e:
                print(f"   ⚠️ Deneme {attempt + 1} başarısız: {aktif_model} hata verdi. {wait_time}sn bekleniyor...")
                time.sleep(wait_time)
                wait_time *= 2  # Hata aldıkça daha uzun bekle
                aktif_model = FALLBACK_MODEL  # Hatada küçük modele geç (Hızlı kurtarma)

    # 🛑 7. VERİ TİPİ VE PADDING FIX (Hepsi integer)
    aktif_seri = [int(x) for x in llama_verisi[:aktif_gun_sayisi]]
    tam_seri = ([0] * (dogum_gunu - 1)) + aktif_seri

    # 1913'e tam eşitleme (Padding veya Cutting)
    tam_seri = tam_seri[:1913]
    while len(tam_seri) < 1913:
        tam_seri.append(0)

    # 🛑 8. GÜVENLİ KAYIT (Header & Size check)
    row_dict = {'item_id': hedef_id}
    for i in range(1, 1914):
        row_dict[f'd_{i}'] = tam_seri[i - 1]

    file_exists = os.path.exists(dosya_adi) and os.path.getsize(dosya_adi) > 0
    pd.DataFrame([row_dict]).to_csv(dosya_adi, mode='a', header=not file_exists, index=False, sep=';')
    print(f"   💾 {hedef_id} kaydedildi. Motor 70B'ye geri dönüyor.")
    aktif_model = PRIMARY_MODEL  # Yeni ürün için tekrar ana modele dön

print("\n🎉 AFİYET OLSUN! GIDA FABRİKASI KUSURSUZ ÇALIŞTI.")