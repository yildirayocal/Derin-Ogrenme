import os
import sys
import numpy as np

_PROJE = os.path.dirname(os.path.abspath(__file__))
if _PROJE not in sys.path:
    sys.path.insert(0, _PROJE)

from data_loader    import CIFARLoader
from knn_classifier import KNNSiniflandirici
from utils import (
    Timer,
    karisiklik_matrisi,
    sinif_metrikler,
    sinif_raporu_yazdir,
    grafik_k_dogruluk,
    grafik_karisiklik,
    grafik_sinif_dogruluk,
    grafik_sure,
    grafik_ornek_tahminler,
    grafik_ogrenme_egrisi,
    grafik_dataset_karsilastirma,
    sonuclari_kaydet,
)


# Çalıştırma modu: "cifar10"  |  "cifar100"  |  "ikisi"
MOD = "ikisi"

# Eğitim / test örnek sayısı
#   None → tam veri (50.000 / 10.000)  ←  yavaş!
#   int  → hızlı deney
NUM_EGITIM  = 5_000
NUM_TEST    = 1_000

# CIFAR-100 etiket tipi: "ince" (100 sınıf) veya "kaba" (20 üst-sınıf)
CIFAR100_ETIKET = "ince"

# Test edilecek k değerleri
K_DEGERLERI = [1, 3, 5, 7, 10, 15, 20]

# Test edilecek mesafe metrikleri
METRIKLER   = ["l2", "l1"]

# Detaylı analiz için en iyi k
EN_IYI_K    = 7

# Sonuçların kaydedileceği klasör
SONUC_YOLU  = os.path.join(_PROJE, "results")

# Tekrarlanabilirlik
TOHUM       = 42

# =============================================================


def baslik(metin, seviye=1):
    if seviye == 1:
        print("\n" + "=" * 65)
        print(f"  {metin}")
        print("=" * 65)
    else:
        print(f"\n  {'─' * 60}")
        print(f"  ▶  {metin}")
        print(f"  {'─' * 60}")


# =============================================================
#  DENEY 1:  k ve Metrik Karşılaştırması
# =============================================================

def deney1_k_metrik(X_egit, y_egit, X_test, y_test,
                     sinif_isimleri, dataset_adi, alt_klasor):
    """
    Tüm k değerleri × metrik kombinasyonlarını test eder.
    Hem CIFAR-10 hem CIFAR-100 için çalışır.
    """
    baslik(f"DENEY 1  —  k & Metrik Karşılaştırması  [{dataset_adi}]")

    sonuclar = {}
    sureler  = {}
    klas     = os.path.join(SONUC_YOLU, alt_klasor)
    os.makedirs(klas, exist_ok=True)

    for metrik in METRIKLER:
        print(f"\n  Metrik: {metrik.upper()}")
        sonuclar[metrik] = []
        sureler[metrik]  = []

        for k in K_DEGERLERI:
            clf = KNNSiniflandirici(k=k, metrik=metrik)
            clf.fit(X_egit, y_egit)

            with Timer() as t:
                acc = clf.score(X_test, y_test)

            sonuclar[metrik].append(acc)
            sureler[metrik].append(t.sure)

            cubuk = "▓" * int(acc * 50)
            print(f"    k={k:3d}  {cubuk:<25}  {acc*100:6.2f}%  ({t.sure:.1f}s)")

    # Özet tablo
    baslik("Özet", seviye=2)
    sas = f"  {'k':>4} | " + " | ".join(f"{m.upper():>8}" for m in METRIKLER)
    print(sas)
    print("  " + "-" * len(sas))
    for i, k in enumerate(K_DEGERLERI):
        satir = (f"  {k:>4} | "
                 + " | ".join(f"{sonuclar[m][i]*100:7.2f}%" for m in METRIKLER))
        en_iyi = sonuclar["l2"][i] == max(sonuclar["l2"])
        print(satir + ("  ★" if en_iyi else ""))

    # Grafikler
    grafik_k_dogruluk(
        K_DEGERLERI, sonuclar,
        kaydet_yolu=os.path.join(klas, "accuracy_vs_k.png"),
        dataset_adi=dataset_adi
    )
    grafik_sure(
        K_DEGERLERI, sureler,
        kaydet_yolu=os.path.join(klas, "prediction_time.png"),
        dataset_adi=dataset_adi
    )
    sonuclari_kaydet(
        K_DEGERLERI, sonuclar,
        kaydet_yolu=os.path.join(klas, "sonuclar.txt"),
        ek_bilgi={
            "Dataset"       : dataset_adi,
            "Eğitim örneği" : NUM_EGITIM,
            "Test örneği"   : NUM_TEST,
        }
    )

    return sonuclar, sureler


# =============================================================
#  DENEY 2:  Detaylı Sınıf Analizi
# =============================================================

def deney2_detayli(X_egit, y_egit, X_test, y_test,
                   sinif_isimleri, n_sinif, dataset_adi, alt_klasor):
    """
    En iyi k ile karışıklık matrisi, sınıf raporu, örnek görseller.
    """
    baslik(f"DENEY 2  —  Detaylı Analiz  [{dataset_adi}, k={EN_IYI_K}]")

    klas           = os.path.join(SONUC_YOLU, alt_klasor)
    sinif_recall   = {}
    en_iyi_metrik  = METRIKLER[0]   # L2

    for metrik in METRIKLER:
        clf = KNNSiniflandirici(k=EN_IYI_K, metrik=metrik)
        clf.fit(X_egit, y_egit)

        with Timer() as t:
            y_tahmin = clf.predict(X_test)
        print(f"\n  [{metrik.upper()}]  Tahmin tamamlandı ({t.sure:.1f}s)")

        sinif_raporu_yazdir(y_test, y_tahmin, EN_IYI_K, metrik, sinif_isimleri)

        cm  = karisiklik_matrisi(y_test, y_tahmin, n_sinif)
        met = sinif_metrikler(cm)
        sinif_recall[f"{metrik.upper()} k={EN_IYI_K}"] = met["recall"]

        if metrik == en_iyi_metrik:
            grafik_karisiklik(
                cm, sinif_isimleri,
                baslik=f"Karışıklık Matrisi  [{dataset_adi}, k={EN_IYI_K}, {metrik.upper()}]",
                kaydet_yolu=os.path.join(klas, "confusion_matrix.png")
            )
            grafik_ornek_tahminler(
                X_test, y_test, y_tahmin, sinif_isimleri,
                kaydet_yolu=os.path.join(klas, "sample_predictions.png"),
                n_ornek=16
            )

    grafik_sinif_dogruluk(
        sinif_recall, sinif_isimleri,
        kaydet_yolu=os.path.join(klas, "class_accuracy.png")
    )


# =============================================================
#  DENEY 3:  Öğrenme Eğrisi
# =============================================================

def deney3_ogrenme_egrisi(X_egit_tam, y_egit_tam, X_test, y_test,
                           dataset_adi, alt_klasor):
    """Eğitim boyutu arttıkça doğruluk nasıl değişir?"""
    baslik(f"DENEY 3  —  Öğrenme Eğrisi  [{dataset_adi}]")

    boyutlar = [b for b in [500, 1_000, 2_000, 3_000, 5_000]
                if b <= len(X_egit_tam)]
    klas     = os.path.join(SONUC_YOLU, alt_klasor)
    ogr_s    = {m: [] for m in METRIKLER}

    for boyut in boyutlar:
        print(f"\n  Eğitim boyutu: {boyut:,}")
        for metrik in METRIKLER:
            clf = KNNSiniflandirici(k=EN_IYI_K, metrik=metrik)
            clf.fit(X_egit_tam[:boyut], y_egit_tam[:boyut])
            with Timer() as t:
                acc = clf.score(X_test, y_test)
            ogr_s[metrik].append(acc)
            print(f"    {metrik.upper()}: {acc*100:.2f}%  ({t.sure:.1f}s)")

    grafik_ogrenme_egrisi(
        boyutlar, ogr_s,
        kaydet_yolu=os.path.join(klas, "learning_curve.png"),
        dataset_adi=dataset_adi
    )

    return ogr_s


# =============================================================
#  TEK DATASET İŞLE
# =============================================================

def dataset_isle(dataset, cifar100_etiket="ince"):
    """Bir dataset için tüm deneyleri çalıştırır ve sonuçları döndürür."""

    loader = CIFARLoader(dataset=dataset, cifar100_etiket=cifar100_etiket)
    X_egit, y_egit, X_test, y_test = loader.yukle(
        normalize  = True,
        num_egitim = NUM_EGITIM,
        num_test   = NUM_TEST,
        karistir   = True,
        tohum      = TOHUM
    )

    adi       = loader.dataset.upper()
    if dataset == "cifar100":
        adi  += f"-{cifar100_etiket.upper()}"
    alt_klas  = dataset + (f"_{cifar100_etiket}" if dataset == "cifar100" else "")

    sonuclar, _ = deney1_k_metrik(X_egit, y_egit, X_test, y_test,
                                   loader.sinif_isimleri, adi, alt_klas)
    deney2_detayli(X_egit, y_egit, X_test, y_test,
                   loader.sinif_isimleri, loader.n_sinif, adi, alt_klas)
    deney3_ogrenme_egrisi(X_egit, y_egit, X_test, y_test, adi, alt_klas)

    return sonuclar


# =============================================================
#  ANA AKIŞ
# =============================================================

def main():
    os.makedirs(SONUC_YOLU, exist_ok=True)

    print("\n" + "█" * 65)
    print(f"  CIFAR k-NN Deneyleri  —  Mod: {MOD.upper()}")
    print("█" * 65)

    sonuclar10  = None
    sonuclar100 = None

    # -----------------------------------------------------------
    if MOD in ("cifar10", "ikisi"):
        baslik("CIFAR-10  (10 sınıf)")
        sonuclar10 = dataset_isle("cifar10")

    # -----------------------------------------------------------
    if MOD in ("cifar100", "ikisi"):
        baslik(f"CIFAR-100  ({CIFAR100_ETIKET} etiket)")
        sonuclar100 = dataset_isle("cifar100", CIFAR100_ETIKET)

    # -----------------------------------------------------------
    if MOD == "ikisi" and sonuclar10 and sonuclar100:
        baslik("KARŞILAŞTIRMA  —  CIFAR-10 vs CIFAR-100")
        grafik_dataset_karsilastirma(
            K_DEGERLERI, sonuclar10, sonuclar100,
            kaydet_yolu=os.path.join(SONUC_YOLU, "cifar10_vs_cifar100.png")
        )

    # -----------------------------------------------------------
    baslik("TAMAMLANDI")

    if sonuclar10:
        bi = int(np.argmax(sonuclar10["l2"]))
        print(f"  CIFAR-10  L2: En iyi k={K_DEGERLERI[bi]}"
              f"  →  {sonuclar10['l2'][bi]*100:.2f}%")

    if sonuclar100:
        bi = int(np.argmax(sonuclar100["l2"]))
        print(f"  CIFAR-100 L2: En iyi k={K_DEGERLERI[bi]}"
              f"  →  {sonuclar100['l2'][bi]*100:.2f}%")

    print(f"\n  Tüm grafikler: {SONUC_YOLU}")
    print()


# =============================================================
if __name__ == "__main__":
    main()
