import pandas as pd
import numpy as np
import os
import pickle
import tensorflow as tf
from tqdm import tqdm

print("\n" + "=" * 60)
print("--- 🏁 GRU ŞAMPİYONLAR LİGİ: ÖLÜMSÜZ TEST MOTORU ---")
print("=" * 60)

# 1. Dosya ve Klasör Yolları
proje_klasoru = os.path.dirname(os.path.abspath(__file__))
veri_klasoru = os.path.join(proje_klasoru, "Tez_Veri_Setleri")
model_klasoru = os.path.join(proje_klasoru, "Tez_Modelleri")
# GRU için ayrı rapor dosyası (Karışmasın diye)
rapor_yolu = os.path.join(proje_klasoru, "GRU_FINAL_KARSILASTIRMA_RAPORU.csv")

test_dosyasi = os.path.join(veri_klasoru, "1_Test_Set_100_Gercek.csv")
sentetik_dosyasi = os.path.join(proje_klasoru, "Sentetik_Hobbies_Final_Dataset.csv")

# 2. Verileri Yükle
print("📦 Veriler yükleniyor...")
df_gercek = pd.read_csv(test_dosyasi, sep=';', low_memory=False)
df_syn = pd.read_csv(sentetik_dosyasi, sep=';', low_memory=False)

df_gercek['d_num'] = df_gercek['day'].str.replace('d_', '').astype(int)
df_syn['d_num'] = df_syn['day'].str.replace('d_', '').astype(int)

# 3. Modelleri ve Scaler'ları Yükle (GRU Spesifik)
model_isimleri = ['Baseline_0', 'Mix_25', 'Mix_50', 'Mix_100']
modeller = {}
scalerlar = {}

print("\n🧠 GRU Modelleri sahaya sürülüyor...")
for m in model_isimleri:
    model_yolu = os.path.join(model_klasoru, f"GRU_Model_{m}.keras")
    scaler_yolu = os.path.join(model_klasoru, f"Scaler_{m}.pkl")

    if os.path.exists(model_yolu) and os.path.exists(scaler_yolu):
        modeller[m] = tf.keras.models.load_model(model_yolu)
        with open(scaler_yolu, 'rb') as f:
            scalerlar[m] = pickle.load(f)
        print(f"✅ GRU {m} hazır!")
    else:
        print(f"❌ HATA: {m} dosyaları eksik!")

# 4. RESUME (KALDIĞI YERDEN DEVAM) MANTIĞI
tamamlanan_urunler = []
if os.path.exists(rapor_yolu):
    try:
        gecmis_df = pd.read_csv(rapor_yolu, sep=';')
        tamamlanan_urunler = gecmis_df['Urun_ID'].tolist()
        print(f"📦 {len(tamamlanan_urunler)} ürün zaten işlenmiş, atlanıyor...")
    except:
        pass

tum_urunler = df_gercek['item_id'].unique()
kalan_urunler = [u for u in tum_urunler if u not in tamamlanan_urunler]

if len(kalan_urunler) == 0:
    print("🎉 Tüm GRU testleri tamamlanmış!")
    exit()

# 5. SINAV DÖNGÜSÜ (WALK-FORWARD & MILAD BUDAMA)
for urun in tqdm(kalan_urunler, desc="GRU Sınavda"):
    urun_gercek_df = df_gercek[df_gercek['item_id'] == urun].sort_values('d_num')
    aktif_gercek = urun_gercek_df[urun_gercek_df['sell_price'] > 0].copy()

    # Milat Bulma (İlk gerçek satışın olduğu gün)
    satis_olan_gunler = aktif_gercek[aktif_gercek['sales'] > 0]
    if satis_olan_gunler.empty: continue

    milad_gunu = satis_olan_gunler.iloc[0]['d_num']
    aktif_gercek = aktif_gercek[aktif_gercek['d_num'] >= milad_gunu].copy()

    if len(aktif_gercek) <= 30: continue

    gercek_gelecek_satis = aktif_gercek.iloc[30:]['sales'].values

    # Sentetik Tohum Hizalama
    urun_syn_isim = urun + "_SYN"
    urun_syn_df = df_syn[df_syn['item_id'] == urun_syn_isim].sort_values('d_num')
    urun_syn_df = urun_syn_df[urun_syn_df['d_num'] >= milad_gunu].copy()

    if len(urun_syn_df) < 30: continue
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

        for d_idx in range(len(gercek_gelecek_satis)):
            input_seq = np.array(current_window).reshape(1, 30, 1)
            # GRU tahmini
            pred_scaled = model(tf.convert_to_tensor(input_seq, dtype=tf.float32), training=False).numpy()[0][0]
            pred_val = scaler.inverse_transform([[pred_scaled]])[0][0]

            ham_tahmin = max(0, pred_val)
            tahminler.append(ham_tahmin)

            # 🛑 WALK-FORWARD: Gerçek Walmart verisini pencereye bas!
            gercek_bugun = gercek_gelecek_satis[d_idx]
            yeni_scaled = scaler.transform([[gercek_bugun]])[0][0]
            current_window.append(yeni_scaled)
            current_window.pop(0)

        # Metrikler (WAPE Tabanlı Doğruluk)
        tahmin_dizisi = np.array(tahminler)
        gercek_dizisi = np.array(gercek_gelecek_satis)

        mutlak_hata = np.sum(np.abs(gercek_dizisi - tahmin_dizisi))
        toplam_gercek = np.sum(gercek_dizisi)
        wape = mutlak_hata / max(1, toplam_gercek)
        dogruluk = max(0, (1 - wape)) * 100

        urun_sonuclari[f'{m}_Tahmin_Toplami'] = round(np.sum(tahmin_dizisi))
        urun_sonuclari[f'{m}_Dogruluk(%)'] = round(dogruluk, 2)

    # ANINDA KAYIT
    df_tek_urun = pd.DataFrame([urun_sonuclari])
    header_mi = not os.path.exists(rapor_yolu)
    df_tek_urun.to_csv(rapor_yolu, index=False, sep=';', mode='a', header=header_mi)

print("\n🏆 GRU TESTLERİ BİTTİ! GENEL TABLO:")
df_final = pd.read_csv(rapor_yolu, sep=';')
for m in model_isimleri:
    print(f">> {m}: %{round(df_final[f'{m}_Dogruluk(%)'].mean(), 2)}")