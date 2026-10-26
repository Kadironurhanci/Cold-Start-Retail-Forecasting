import pandas as pd
import time
import re
import os
import random
from groq import Groq

# 1. AYARLAR VE MOTOR SEÇENEKLERİ
# LÜTFEN KENDİ GROQ API KEY'İNİ BURAYA YAPIŞTIR:
GROQ_API_KEY = ""
client = Groq(api_key=GROQ_API_KEY)

# Birinci motor (Çok Zeki), İkinci motor (Çok Hızlı)
PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"

# Başlangıç motorumuz ana model
aktif_model = PRIMARY_MODEL

dosya_adi = 'Akilli_Sentetik_Hobbies_Llama.csv'

print("--- 🦙 LLAMA SENTETİK VERİ FABRİKASI BAŞLATILDI ---")
profil_df = pd.read_csv('Hobbies_Urun_Profilleri.csv', sep=';')

# 2. KALINAN YERİ BULMA (RESUME ÖZELLİĞİ)
tamamlananlar = []
if os.path.exists(dosya_adi):
    mevcut_df = pd.read_csv(dosya_adi, sep=';')
    tamamlananlar = mevcut_df['item_id'].tolist()
    print(f"📂 Dosya bulundu! {len(tamamlananlar)} ürün zaten üretilmiş. Kaldığımız yerden devam ediyoruz...")
else:
    print("📂 Yeni dosya oluşturulacak. Sıfırdan başlanıyor...")

# 3. ÜRETİM DÖNGÜSÜ
for index, row in profil_df.iterrows():
    urun_id = row['urun_id']

    # KANKANIN İSTEĞİ: LLAMA İMZASI EKLENDİ!
    hedef_id = urun_id + "_SYN_LLAMA"

    if hedef_id in tamamlananlar:
        continue

    dogum_gunu = int(row['dogum_gunu'])
    aktif_gun = int(row['aktif_gun_sayisi'])
    sifir_oran = float(str(row['sifir_orani_%']).replace(',', '.'))
    max_satis = int(row['max_gunluk_satis'])
    ort_satis = float(str(row['ort_satis_adeti']).replace(',', '.'))

    if aktif_gun < 10:
        continue

    yeni_dogum_gunu = random.randint(1, 1500)
    yeni_aktif_gun = 1913 - yeni_dogum_gunu + 1

    print(f"\n📦 {hedef_id} üretiliyor... (Model: {aktif_model})")

    print(f"\n📦 {hedef_id} üretiliyor... (Hedef: {yeni_aktif_gun} gün - Parça parça toplanacak)")

    llama_verisi = []

    # 🛑 KANKANIN İSTEDİĞİ ASLA SIFIRLA DOLDURMAYAN DÖNGÜ
    while len(llama_verisi) < yeni_aktif_gun:
        eksik_adet = yeni_aktif_gun - len(llama_verisi)
        # Modele tek seferde en fazla 400 adet soruyoruz. Ne o yorulur, ne de Groq kotası dolar!
        istenecek_adet = min(eksik_adet, 400)

        try:
            system_prompt = "Sen sadece virgülle ayrılmış sayılar üreten bir makinesin. ASLA harf veya boşluk kullanma. Sadece sayı ver."

            user_prompt = f"""
                Sen Amerika'da bir Walmart mağazasında 'Hobbies' reyonu müdürüsün. 
                Bana tam olarak {istenecek_adet} adet günlük satış verisi üret.

                KURALLAR:
                1. Sayıların yaklaşık %{sifir_oran} kadarı "0" olmalıdır.
                2. Üst üste 5 defadan fazla "0" YAZAMAZSIN! Aralara satış koy.
                3. Normal günlerde genelde 1, 2 veya 3 adet satılır (Ortalama {ort_satis}).
                4. Rastgele patlamalar yap! Bu pikler {max_satis} seviyelerine çıkabilir.
                5. SADECE virgülle ayrılmış tam sayılar yaz, ASLA HARF KULLANMA.
                6. Toplam sayı adedi TAM OLARAK {istenecek_adet} olmalıdır.
                """

            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=aktif_model,
                temperature=0.85,
                max_tokens=2000  # Sadece 400 sayı istediğimiz için 2000 token rahat yeter!
            )

            response_text = response.choices[0].message.content
            clean_text = re.sub(r'[^0-9,]', '', response_text.strip())
            parca_verisi = [s for s in clean_text.split(',') if s]

            if len(parca_verisi) > 0:
                llama_verisi.extend(parca_verisi)
                print(f"   [+] {len(parca_verisi)} gün üretildi. Toplam ulaşılan: {len(llama_verisi)}/{yeni_aktif_gun}")
                time.sleep(3)  # Groq'u kızdırmamak için 3 saniye nefeslenme
            else:
                print("   ⚠️ Boş cevap geldi, tekrar deneniyor...")
                time.sleep(3)

        except Exception as e:
            hata_mesaji = str(e).lower()
            print(f"   🚨 GROQ HATASI: {hata_mesaji}")

            if "429" in hata_mesaji or "rate limit" in hata_mesaji or "413" in hata_mesaji:
                if aktif_model == PRIMARY_MODEL:
                    print("   🔄 70B yoruldu! Motor 8B'ye geçiriliyor ve 10 sn bekleniyor...")
                    aktif_model = FALLBACK_MODEL
                    time.sleep(10)
                else:
                    print("   ⏳ 8B'de de limite çarptık. 40 saniye soğuma molası...")
                    time.sleep(40)
                    aktif_model = PRIMARY_MODEL
            else:
                print(f"   ❌ API Çöktü! 10 saniye bekleniyor...")
                time.sleep(10)

    # DÖNGÜ BİTTİ (İstediğimiz rakama %100 saf yapay zeka ile ulaştık!)
    llama_verisi = llama_verisi[:yeni_aktif_gun]

    # Başlangıçtaki o gerçekçi boş dönemi 0 ile dolduruyoruz.
    baslangic_sifirlari = ['0'] * (yeni_dogum_gunu - 1)
    tam_seri = baslangic_sifirlari + llama_verisi

    row_data = {'item_id': hedef_id}
    for i in range(1, 1914):
        if i - 1 < len(tam_seri):
            row_data[f'd_{i}'] = tam_seri[i - 1]
        else:
            row_data[f'd_{i}'] = '0'

    df_tek = pd.DataFrame([row_data])

    if not os.path.exists(dosya_adi):
        df_tek.to_csv(dosya_adi, index=False, sep=';')
    else:
        df_tek.to_csv(dosya_adi, mode='a', header=False, index=False, sep=';')

    print(f"   💾 EFSANE VERİ KAYDEDİLDİ! Sıradaki ürüne geçiliyor...\n")

print("\n🎉 BÜTÜN ÜRÜNLER İÇİN LLAMA VERİ ÜRETİMİ TAMAMLANDI!")