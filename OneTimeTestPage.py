import os
import numpy as np
import matplotlib.pyplot as plt

print("--- TEZ SUNUMU ICIN KARSILASTIRMALI GRAFIK MOTORU BASLIYOR ---")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
graphs_save_dir = os.path.join(PROJECT_DIR, "Tez_Grafikleri_Llama")

if not os.path.exists(graphs_save_dir):
    os.makedirs(graphs_save_dir)

# Senin test sonuclarindan alinan gercek metrikler
kategoriler = ['Baseline (0% Sentetik)', 'Mix %25 Sentetik', 'Mix %50 Sentetik', 'Mix %100 Sentetik']

# RMSE Degerleri (Daha dusuk daha iyi)
rmse_gru = [2.6609, 2.6515, 2.6552, 2.6838]
rmse_lstm = [2.8339, 2.7209, 2.6232, 2.6676]

# R-Kare Degerleri (Daha yuksek daha iyi)
r2_gru = [0.2932, 0.2982, 0.2962, 0.2810]
r2_lstm = [0.1983, 0.2610, 0.3131, 0.2897]

# sMAPE Degerleri (Daha dusuk daha iyi)
smape_gru = [159.82, 161.10, 160.12, 158.68]
smape_lstm = [159.16, 158.21, 112.65, 162.19]

x = np.arange(len(kategoriler))
width = 0.35


def ciz_ve_kaydet(veri_gru, veri_lstm, baslik, y_ekseni_adi, dosya_adi, en_iyi_dusuk_mu):
    fig, ax = plt.subplots(figsize=(10, 6))

    rects1 = ax.bar(x - width / 2, veri_gru, width, label='GRU Modeli', color='#ff7f0e')
    rects2 = ax.bar(x + width / 2, veri_lstm, width, label='LSTM Modeli', color='#1f77b4')

    ax.set_ylabel(y_ekseni_adi, fontsize=12)
    ax.set_title(baslik, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(kategoriler, fontsize=11)
    ax.legend(fontsize=12)

    # Barlarin uzerine degerleri yazdirma
    ax.bar_label(rects1, padding=3, fmt='%.4f' if 'R-Kare' in baslik else '%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.4f' if 'R-Kare' in baslik else '%.2f')

    # Arka plan cizgileri
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # En iyi modelin vurgulanmasi (Juri icin efsane detay)
    if en_iyi_dusuk_mu:
        min_val = min(min(veri_gru), min(veri_lstm))
        if min_val in veri_gru:
            ax.patches[veri_gru.index(min_val)].set_facecolor('#d62728')
        else:
            ax.patches[len(veri_gru) + veri_lstm.index(min_val)].set_facecolor('#d62728')
    else:
        max_val = max(max(veri_gru), max(veri_lstm))
        if max_val in veri_gru:
            ax.patches[veri_gru.index(max_val)].set_facecolor('#2ca02c')
        else:
            ax.patches[len(veri_gru) + veri_lstm.index(max_val)].set_facecolor('#2ca02c')

    fig.tight_layout()
    plt.savefig(os.path.join(graphs_save_dir, dosya_adi), dpi=300)
    plt.close()


# 1. Grafik: RMSE Karsilastirmasi (Dusuk olan kirmizi ile isaretlenir)
ciz_ve_kaydet(rmse_gru, rmse_lstm, 'Modellerin Ortalama Hata (RMSE) Karsilastirmasi', 'RMSE Skoru',
              '1_RMSE_Karsilastirma.png', en_iyi_dusuk_mu=True)

# 2. Grafik: R-Kare Karsilastirmasi (Yuksek olan yesil ile isaretlenir)
ciz_ve_kaydet(r2_gru, r2_lstm, 'Modellerin Aciklanabilirlik (R-Kare) Karsilastirmasi', 'R-Kare Skoru',
              '2_R2_Karsilastirma.png', en_iyi_dusuk_mu=False)

# 3. Grafik: sMAPE Karsilastirmasi (Dusuk olan kirmizi ile isaretlenir)
ciz_ve_kaydet(smape_gru, smape_lstm, 'Modellerin Yuzdesel Hata (sMAPE) Karsilastirmasi', 'sMAPE (%)',
              '3_sMAPE_Karsilastirma.png', en_iyi_dusuk_mu=True)

print(f"ISLEM TAMAM! 3 adet karsilastirmali bar grafigi '{graphs_save_dir}' klasorune HD olarak kaydedildi.")