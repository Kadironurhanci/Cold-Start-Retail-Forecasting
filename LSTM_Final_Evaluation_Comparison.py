import pandas as pd
import numpy as np
import os
import pickle
import tensorflow as tf
from tqdm import tqdm

print("Şampiyonlar ligini başlattık bismillah ")

#dosya yolları
proje_klasoru = os.path.dirname(os.path.abspath(__file__))

veri_klasoru = os.path.join(proje_klasoru, "Tez_Veri_Setleri")
model_klasoru = os.path.join(proje_klasoru, "Tez_Modelleri")
rapor_yolu = os.path.join(proje_klasoru, "TEZ_FINAL_KARSILASTIRMA_RAPORU.csv")  # Tek bir ana dosya

test_dosyasi = os.path.join(veri_klasoru, "1_Test_Set_100_Gercek.csv")
sentetik_dosyasi = os.path.join(proje_klasoru, "Sentetik_Hobbies_Final_Dataset.csv")

#verileri getir
print("Test ve Sentetik Veriler Yükleniyor...")
df_gercek = pd.read_csv(test_dosyasi, sep=';', low_memory=False)
df_syn = pd.read_csv(sentetik_dosyasi, sep=';', low_memory=False)

df_gercek['d_num'] = df_gercek['day'].str.replace('d_', '').astype(int)
df_syn['d_num'] = df_syn['day'].str.replace('d_', '').astype(int)

#modelleri getir
model_isimleri = ['Baseline_0', 'Mix_25', 'Mix_50', 'Mix_100']
modeller = {}
scalerlar = {}

print("\nmodeller sahaya çıkıyor kral")
for m in model_isimleri:
    model_yolu = os.path.join(model_klasoru, f"LSTM_Model_{m}.keras")
    scaler_yolu = os.path.join(model_klasoru, f"Scaler_{m}.pkl")

    if os.path.exists(model_yolu) and os.path.exists(scaler_yolu):
        modeller[m] = tf.keras.models.load_model(model_yolu)
        with open(scaler_yolu, 'rb') as f:
            scalerlar[m] = pickle.load(f)
        print(f"✅ {m} hazır!")
    else:
        print(f"❌ HATA: {m} dosyaları bulunamadı!")

#kaldığı yerden devam etme işi
tamamlanan_urunler = []

# Eğer daha önceden başlanmış bir Excel raporu varsa, oku ve bitenleri tespit et
if os.path.exists(rapor_yolu):
    try:
        gecmis_df = pd.read_csv(rapor_yolu, sep=';')
        tamamlanan_urunler = gecmis_df['Urun_ID'].tolist()
        print(f"\n📦 Bulunan kayıt: {len(tamamlanan_urunler)} adet ürün zaten test edilmiş. Bunlar atlanıyor...")
    except Exception as e:
        print(f"\n⚠️ Kayıt okuma hatası, sıfırdan başlanıyor. Detay: {e}")

# Hedef ürünleri al ve daha önce bitmiş olanları listeden çıkar
tum_urunler = df_gercek['item_id'].unique()
kalan_urunler = [u for u in tum_urunler if u not in tamamlanan_urunler]

if len(kalan_urunler) == 0:
    print("\n🎉 TEBRİKLER! Tüm ürünlerin testi zaten tamamlanmış. Gidip raporu inceleyebilirsin.")
    exit()  # Kod burada durur, boşuna çalışmaz

print(
    f"\nToplam {len(kalan_urunler)} kalan ürün için tahmin savaşı başlıyor. İstediğin zaman durdurabilirsin, kayıt güvende!")

# =====================================================================
# 5. SINAV DÖNGÜSÜ (ANINDA KAYIT VE WALK-FORWARD İLE)
# =====================================================================
for urun in tqdm(kalan_urunler, desc="Ürünler Sınavda"):
    urun_gercek_df = df_gercek[df_gercek['item_id'] == urun].sort_values('d_num')
    aktif_gercek = urun_gercek_df[urun_gercek_df['sell_price'] > 0].copy()

    # 🛑 KANKANIN EFSANE DOKUNUŞU: ÖNCÜ SIFIRLARI BUDAMA (SONRASINA ODAKLANMA)
    satis_olan_gunler = aktif_gercek[aktif_gercek['sales'] > 0]

    if satis_olan_gunler.empty:
        continue

    # Ürünün SATIŞA BAŞLADIĞI o ilk tarihi (d_num) buluyoruz.
    ilk_satis_gunu = satis_olan_gunler.iloc[0]['d_num']

    # Milat direkt ilk satışın olduğu gün! (Geriye gitmek yok, sıfırları eledik)
    milad_gunu = ilk_satis_gunu

    # 1. GERÇEK VERİYİ HİZALA: Veriyi çöplerden arındırıyoruz!
    aktif_gercek = aktif_gercek[aktif_gercek['d_num'] >= milad_gunu].copy()

    if len(aktif_gercek) <= 30:
        continue

    #31. günden sonraki verileri al
    gercek_gelecek_satis = aktif_gercek.iloc[30:]['sales'].values

    #SENTETİK VERİYİ HİZALA: Zaman çizelgesi şaşmasın diye onu da milattan kesiyoruz
    urun_syn_isim = urun + "_SYN"
    urun_syn_df = df_syn[df_syn['item_id'] == urun_syn_isim].sort_values('d_num')
    urun_syn_df = urun_syn_df[urun_syn_df['d_num'] >= milad_gunu].copy()

    if len(urun_syn_df) < 30:
        continue

    # Sentetik tohum: Milattan sonraki ilk 30 gün
    sentetik_tohum = urun_syn_df.iloc[:30]['sales'].values

    urun_sonuclari = {
        'Urun_ID': urun,
        'Toplam_Gelecek_Gun': len(gercek_gelecek_satis),
        'Gercek_Satis_Toplami': np.sum(gercek_gelecek_satis)
    }

    for m in model_isimleri:
        scaler = scalerlar[m]
        model = modeller[m]

        current_window = list(scaler.transform(sentetik_tohum.reshape(-1, 1)).flatten())
        tahminler = []

        # Her bir gün için ileriye doğru tahmin
        for d_idx in range(len(gercek_gelecek_satis)):
            input_seq = np.array(current_window).reshape(1, 30, 1)
            input_tensor = tf.convert_to_tensor(input_seq, dtype=tf.float32)

            pred_scaled = model(input_tensor, training=False).numpy()[0][0]
            pred_val = scaler.inverse_transform([[pred_scaled]])[0][0]

            #Excel'e gidecek listeye eksi olmayan ham tahmini atıyoruz
            ham_tahmin = max(0, pred_val)
            tahminler.append(ham_tahmin)

            #KRİTİK GÜNCELLEME: WALK-FORWARD (ÖĞRETMEN TAKTİĞİ)
            #kendi tahmin ettiği hatadan öğrenmemesi lazım
            gercek_bugunun_satisi = gercek_gelecek_satis[d_idx]
            yeni_scaled = scaler.transform([[gercek_bugunun_satisi]])[0][0]

            current_window.append(yeni_scaled)
            current_window.pop(0)

        #accuracy hesaplamaları
        tahmin_dizisi = np.array(tahminler)
        gercek_dizisi = np.array(gercek_gelecek_satis)

        toplam_gercek = np.sum(gercek_dizisi)
        toplam_tahmin = round(np.sum(tahmin_dizisi))

        mutlak_hata = np.sum(np.abs(gercek_dizisi - tahmin_dizisi))
        wape = mutlak_hata / max(1, toplam_gercek)
        dogruluk_yuzdesi = max(0, (1 - wape)) * 100

        urun_sonuclari[f'{m}_Tahmin_Toplami'] = toplam_tahmin
        urun_sonuclari[f'{m}_Dogruluk(%)'] = round(dogruluk_yuzdesi, 2)

    #anında dosyaya yazması lazım pc kapanıyo bazen restart atıyo
    df_tek_urun = pd.DataFrame([urun_sonuclari])

    if not os.path.exists(rapor_yolu):
        #dosya yoksa oluştur
        df_tek_urun.to_csv(rapor_yolu, index=False, sep=';', mode='w')
    else:
        #varsa en alta ekle
        df_tek_urun.to_csv(rapor_yolu, index=False, sep=';', mode='a', header=False)

#hepsi bittiğinde uyarı mesajı
print("\ntestler tamam")
print(f"Rapor Yolu: {rapor_yolu}")

#performansı yazdır
try:
    df_final = pd.read_csv(rapor_yolu, sep=';')
    print("\nGENEL MODEL PERFORMANSLARI")
    for m in model_isimleri:
        ort_dogruluk = df_final[f'{m}_Dogruluk(%)'].mean()
        print(f">> {m}: %{round(ort_dogruluk, 2)}")
except:
    pass