import os
import time
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Timer:
    """with Timer() as t: ... → t.sure saniye cinsinden süre"""
    def __enter__(self):
        self._bas = time.perf_counter()
        return self
    def __exit__(self, *_):
        self.sure = time.perf_counter() - self._bas
    def __str__(self):
        return f"{self.sure:.2f}s"

def karisiklik_matrisi(y_gercek, y_tahmin, n_sinif):
    """
    n_sinif × n_sinif karışıklık matrisi.
    cm[gercek][tahmin] = örnek sayısı.
    """
    cm = np.zeros((n_sinif, n_sinif), dtype=int)
    for g, t in zip(
        np.asarray(y_gercek, dtype=int),
        np.asarray(y_tahmin, dtype=int)
    ):
        if 0 <= g < n_sinif and 0 <= t < n_sinif:
            cm[g, t] += 1
    return cm


def sinif_metrikler(cm):
    """
    Precision / Recall / F1 / Support.

    Dönüş: dict
        'precision', 'recall', 'f1' → (n_sinif,) float
        'support'                   → (n_sinif,) int
    """
    tp      = cm.diagonal().astype(float)
    fp      = cm.sum(axis=0) - tp
    fn      = cm.sum(axis=1) - tp
    support = cm.sum(axis=1).astype(float)

    prec = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
    rec  = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
    f1   = np.where((prec + rec) > 0,
                    2.0 * prec * rec / (prec + rec), 0.0)
    return {"precision": prec, "recall": rec, "f1": f1, "support": support}


def sinif_raporu_yazdir(y_gercek, y_tahmin, k, metrik, sinif_isimleri):
    """
    Terminale sınıf bazında precision / recall / F1 tablosu basar.
    CIFAR-10 (10 sınıf) veya CIFAR-100 (100 sınıf) için çalışır.
    100 sınıflı için sadece ilk 20 ve son 5 sınıfı gösterir.
    """
    y_g   = np.asarray(y_gercek, dtype=int)
    y_t   = np.asarray(y_tahmin, dtype=int)
    n     = len(sinif_isimleri)
    cm    = karisiklik_matrisi(y_g, y_t, n)
    met   = sinif_metrikler(cm)
    genel = float(np.mean(y_g == y_t))
    agirl = float(np.average(met["f1"], weights=met["support"]))

    print("\n" + "=" * 70)
    print(f"  k={k}  |  {metrik.upper()}  |  Genel Doğruluk: {genel*100:.2f}%  |  {n} sınıf")
    print("=" * 70)
    print(f"  {'ID':>4}  {'Sınıf':<24} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Destek':>8}")
    print("  " + "-" * 66)

    # 100 sınıf için sadece seçili satırları göster
    if n <= 20:
        gosterilecek = list(range(n))
    else:
        gosterilecek = list(range(20)) + ["..."] + list(range(n - 5, n))

    for idx in gosterilecek:
        if idx == "...":
            print(f"  {'...':>4}  {'(ara sınıflar atlandı)':<24}")
            continue
        isim = sinif_isimleri[idx] if idx < len(sinif_isimleri) else str(idx)
        print(
            f"  {idx:>4}  {isim:<24}"
            f"  {met['precision'][idx]*100:>8.1f}%"
            f"  {met['recall'][idx]*100:>6.1f}%"
            f"  {met['f1'][idx]*100:>6.1f}%"
            f"  {int(met['support'][idx]):>7}"
        )
    print("  " + "-" * 66)
    print(f"  {'':>4}  {'Ağırlıklı Ortalama':<24}  {'':>9}  {'':>7}  {agirl*100:>6.1f}%")
    print("=" * 70 + "\n")


def _kaydet(fig, yol):
    os.makedirs(os.path.dirname(yol) or ".", exist_ok=True)
    fig.savefig(yol, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✔] Kaydedildi: {yol}")



def grafik_k_dogruluk(k_degerleri, sonuclar, kaydet_yolu, dataset_adi="CIFAR"):
    """
    sonuclar = {'l2': [0.33, 0.36, ...], 'l1': [...]}
    """
    RENKLER  = {"l2": "#2196F3", "l1": "#FF5722", "cos": "#4CAF50"}
    ISARETCI = {"l2": "o",       "l1": "s",        "cos": "^"}

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for metrik, dogruluklar in sonuclar.items():
        pct  = [d * 100 for d in dogruluklar]
        renk = RENKLER.get(metrik, "#888")

        ax.plot(
            k_degerleri, pct,
            color=renk, marker=ISARETCI.get(metrik, "o"),
            linewidth=2.4, markersize=9,
            markerfacecolor="white", markeredgewidth=2.2,
            label=f"Metrik: {metrik.upper()}"
        )

        # Değer etiketleri
        for kv, pv in zip(k_degerleri, pct):
            ax.annotate(f"{pv:.1f}", xy=(kv, pv), xytext=(0, 8),
                        textcoords="offset points",
                        ha="center", fontsize=8, color=renk, alpha=0.85)

        # En iyi nokta
        bi = int(np.argmax(pct))
        ax.annotate(
            f"En iyi\nk={k_degerleri[bi]}, {pct[bi]:.1f}%",
            xy=(k_degerleri[bi], pct[bi]),
            xytext=(k_degerleri[bi] + 0.8, pct[bi] + 2.5),
            fontsize=8.5, color=renk, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=renk, lw=1.2)
        )

    tum = [d * 100 for v in sonuclar.values() for d in v]
    ax.set_ylim(max(0, min(tum) - 5), min(100, max(tum) + 10))
    ax.set_xticks(k_degerleri)
    ax.set_xlabel("k  (Komşu Sayısı)", fontsize=12)
    ax.set_ylabel("Test Doğruluğu (%)", fontsize=12)
    ax.set_title(f"{dataset_adi}  k-NN  —  k Değeri  ↔  Test Doğruluğu",
                 fontsize=14, fontweight="bold", pad=14)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle="--")

    plt.tight_layout()
    _kaydet(fig, kaydet_yolu)


def grafik_karisiklik(cm, sinif_isimleri, baslik, kaydet_yolu):
    """
    CIFAR-10 (10×10) veya CIFAR-100 (100×100 / 20×20) destekler.
    100×100 matris için etiket fontunu küçültür.
    """
    n     = len(sinif_isimleri)
    cm_n  = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    boyut = max(8, n * 0.9)
    fig, eksenler = plt.subplots(1, 2, figsize=(min(boyut * 2, 22), min(boyut, 11)))

    font_etiket = max(4, 9 - n // 20)
    font_hucre  = max(3, 7 - n // 20)

    for ax, veri, yan_baslik, fmt in [
        (eksenler[0], cm,   "Ham Sayılar",   "d"),
        (eksenler[1], cm_n, "Normalize (%)", ".0%"),
    ]:
        im   = ax.imshow(veri, interpolation="nearest",
                         cmap="Blues", vmin=0, vmax=veri.max())
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(sinif_isimleri, rotation=90,
                           ha="right", fontsize=font_etiket)
        ax.set_yticklabels(sinif_isimleri, fontsize=font_etiket)

        # Hücre değerleri — sadece n≤30 için yaz (çok kalabalık olmasın)
        if n <= 30:
            esik = veri.max() / 2.0
            for i in range(n):
                for j in range(n):
                    dstr = (f"{cm[i,j]}"
                            if fmt == "d"
                            else f"{cm_n[i,j]*100:.0f}%")
                    ax.text(j, i, dstr,
                            ha="center", va="center",
                            fontsize=font_hucre,
                            color="white" if veri[i, j] > esik else "black")

        ax.set_xlabel("Tahmin Edilen Sınıf", fontsize=10)
        ax.set_ylabel("Gerçek Sınıf", fontsize=10)
        ax.set_title(f"{baslik}\n({yan_baslik})", fontsize=11, fontweight="bold")

    plt.tight_layout()
    _kaydet(fig, kaydet_yolu)



def grafik_sinif_dogruluk(sinif_recall_dict, sinif_isimleri,
                           kaydet_yolu, top_n=20):
    """
    sinif_recall_dict = {'L2 k=7': [0.37, 0.45, ...], ...}
    CIFAR-100'de 100 sınıfın tamamını çizmek yerine
    en iyi top_n / en kötü top_n sınıfları gösterir.
    """
    n = len(sinif_isimleri)
    RENKLER = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    # İlk serideki recall'a göre sırala
    ilk_seri = list(sinif_recall_dict.values())[0]
    if n > top_n * 2:
        # En iyi top_n + en kötü top_n
        sirali_idx  = np.argsort(ilk_seri)
        gosterilen  = list(sirali_idx[:top_n]) + list(sirali_idx[-top_n:])
        alt_baslik  = f"En Kötü {top_n} ve En İyi {top_n} Sınıf"
    else:
        gosterilen = list(range(n))
        alt_baslik  = "Tüm Sınıflar"

    etiketler = [sinif_isimleri[i] for i in gosterilen]
    x  = np.arange(len(gosterilen))
    ng = len(sinif_recall_dict)
    w  = 0.65 / ng

    fig, ax = plt.subplots(figsize=(max(12, len(gosterilen) * 0.4), 5.5))

    for i, (seri_adi, degerler) in enumerate(sinif_recall_dict.items()):
        secilmis = [degerler[j] for j in gosterilen]
        ofs      = (i - ng / 2 + 0.5) * w
        bars     = ax.bar(x + ofs, [d * 100 for d in secilmis],
                          w, label=seri_adi,
                          color=RENKLER[i % len(RENKLER)], alpha=0.85)

        if len(gosterilen) <= 30:
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width() / 2, h + 0.2,
                        f"{h:.0f}", ha="center", va="bottom",
                        fontsize=6, rotation=90)

    tum = [d for v in sinif_recall_dict.values() for d in v]
    ax.axhline(np.mean(tum) * 100, color="#607D8B",
               linestyle="--", lw=1.5, alpha=0.7,
               label=f"Genel Ort.: {np.mean(tum)*100:.1f}%")

    ax.set_xticks(x)
    ax.set_xticklabels(etiketler, rotation=40, ha="right",
                       fontsize=max(6, 9 - len(gosterilen) // 10))
    ax.set_xlabel("Sınıf", fontsize=11)
    ax.set_ylabel("Recall (%)", fontsize=11)
    ax.set_title(f"Sınıf Bazında Doğruluk  —  {alt_baslik}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.set_ylim(0, min(100, max(d * 100 for v in sinif_recall_dict.values() for d in v) + 12))

    plt.tight_layout()
    _kaydet(fig, kaydet_yolu)



def grafik_sure(k_degerleri, sureler, kaydet_yolu, dataset_adi="CIFAR"):
    """sureler = {'l2': [5.2, 5.3, ...], 'l1': [12.1, ...]}"""
    RENKLER = ["#2196F3", "#FF5722", "#4CAF50"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x  = np.arange(len(k_degerleri))
    ng = len(sureler)
    w  = 0.6 / ng

    for i, (metrik, sl) in enumerate(sureler.items()):
        ofs = (i - ng / 2 + 0.5) * w
        ax1.bar(x + ofs, sl, w, label=metrik.upper(),
                color=RENKLER[i % len(RENKLER)], alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels([str(k) for k in k_degerleri])
    ax1.set_xlabel("k Değeri", fontsize=11)
    ax1.set_ylabel("Tahmin Süresi (sn)", fontsize=11)
    ax1.set_title(f"{dataset_adi}  —  k'nin Süreye Etkisi",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, axis="y", alpha=0.25)

    maks = max(sl[0] for sl in sureler.values())
    ax1.text(
        len(k_degerleri) / 2 - 0.5, maks * 0.5,
        "k artışı tahmin\nsüresini neredeyse\nETKİLEMEZ",
        fontsize=9, ha="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="#FFFDE7", ec="#FFC107")
    )

    metrikler  = list(sureler.keys())
    ort_sureler = [float(np.mean(sureler[m])) for m in metrikler]
    cbars = ax2.bar([m.upper() for m in metrikler], ort_sureler,
                    color=[RENKLER[i % len(RENKLER)] for i in range(len(metrikler))],
                    alpha=0.85, width=0.5)
    for c, s in zip(cbars, ort_sureler):
        ax2.text(c.get_x() + c.get_width() / 2, s + 0.05,
                 f"{s:.1f}s", ha="center", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Mesafe Metriği", fontsize=11)
    ax2.set_ylabel("Ortalama Süre (sn)", fontsize=11)
    ax2.set_title("Metriğin Süreye Etkisi", fontsize=12, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    _kaydet(fig, kaydet_yolu)



def grafik_ornek_tahminler(X_test, y_gercek, y_tahmin,
                            sinif_isimleri, kaydet_yolu, n_ornek=16):
    """
    Örnek görüntüleri gerçek ve tahmin etiketiyle gösterir.
    CIFAR-10 ve CIFAR-100 için çalışır.
    """
    y_g = np.asarray(y_gercek, dtype=int)
    y_t = np.asarray(y_tahmin, dtype=int)

    rng  = np.random.default_rng(42)
    idxs = rng.choice(len(X_test), size=min(n_ornek, len(X_test)), replace=False)

    sutun = 8
    satir = (len(idxs) + sutun - 1) // sutun
    fig, eksenler = plt.subplots(satir, sutun,
                                  figsize=(sutun * 1.9, satir * 2.5))

    if isinstance(eksenler, np.ndarray):
        eksenler = eksenler.flatten()
    else:
        eksenler = [eksenler]

    for ax, idx in zip(eksenler, idxs):
        goruntu = X_test[idx].reshape(3, 32, 32).transpose(1, 2, 0)
        goruntu = np.clip(goruntu, 0.0, 1.0)

        dogru   = (y_t[idx] == y_g[idx])
        cerceve = "#27AE60" if dogru else "#E74C3C"

        g_isim = sinif_isimleri[y_g[idx]] if y_g[idx] < len(sinif_isimleri) else str(y_g[idx])
        t_isim = sinif_isimleri[y_t[idx]] if y_t[idx] < len(sinif_isimleri) else str(y_t[idx])

        ax.imshow(goruntu, interpolation="bilinear")
        ax.set_title(f"G:{g_isim}\nT:{t_isim}",
                     fontsize=6.5, color=cerceve, fontweight="bold")

        for kenar in ax.spines.values():
            kenar.set_visible(True)
            kenar.set_edgecolor(cerceve)
            kenar.set_linewidth(3)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in eksenler[len(idxs):]:
        ax.axis("off")

    n_d = sum(1 for i in idxs if y_t[i] == y_g[i])
    fig.suptitle(
        f"Örnek Tahminler  (Yeşil=Doğru / Kırmızı=Yanlış)  "
        f"[{n_d}/{len(idxs)}  ≈  {n_d/len(idxs)*100:.0f}%]",
        fontsize=10, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    _kaydet(fig, kaydet_yolu)

def grafik_ogrenme_egrisi(boyutlar, sonuclar, kaydet_yolu, dataset_adi="CIFAR"):
    """sonuclar = {'l2': [0.30, 0.34, ...], 'l1': [...]}"""
    RENKLER = {"l2": "#2196F3", "l1": "#FF5722", "cos": "#4CAF50"}
    fig, ax = plt.subplots(figsize=(9, 5))

    for metrik, dogr in sonuclar.items():
        renk = RENKLER.get(metrik, "#888")
        ax.plot(boyutlar, [d * 100 for d in dogr],
                marker="o", color=renk, linewidth=2.2,
                markersize=8, label=metrik.upper())
        for bx, dy in zip(boyutlar, dogr):
            ax.annotate(f"{dy*100:.1f}%", xy=(bx, dy * 100),
                        xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=8, color=renk)

    ax.set_xlabel("Eğitim Seti Boyutu", fontsize=12)
    ax.set_ylabel("Test Doğruluğu (%)", fontsize=12)
    ax.set_title(f"{dataset_adi}  k-NN  —  Öğrenme Eğrisi",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v):,}")
    )
    plt.tight_layout()
    _kaydet(fig, kaydet_yolu)


def grafik_dataset_karsilastirma(k_degerleri, sonuclar_10, sonuclar_100,
                                  kaydet_yolu):
    """
    Aynı grafik üzerinde CIFAR-10 ve CIFAR-100 L2 doğruluklarını gösterir.

    sonuclar_10  = {'l2': [0.33, 0.36, ...]}
    sonuclar_100 = {'l2': [0.14, 0.16, ...]}
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))

    konfigler = [
        (sonuclar_10.get("l2",  []),  "#2196F3", "o", "CIFAR-10  L2"),
        (sonuclar_10.get("l1",  []),  "#64B5F6", "s", "CIFAR-10  L1"),
        (sonuclar_100.get("l2", []),  "#FF5722", "^", "CIFAR-100 L2 (ince)"),
        (sonuclar_100.get("l1", []),  "#FF8A65", "D", "CIFAR-100 L1 (ince)"),
    ]

    for dogr, renk, isaretci, etiket in konfigler:
        if not dogr:
            continue
        pct = [d * 100 for d in dogr]
        ax.plot(k_degerleri[:len(pct)], pct,
                color=renk, marker=isaretci,
                linewidth=2.2, markersize=8,
                markerfacecolor="white", markeredgewidth=2,
                label=etiket)

    ax.set_xticks(k_degerleri)
    ax.set_xlabel("k  (Komşu Sayısı)", fontsize=12)
    ax.set_ylabel("Test Doğruluğu (%)", fontsize=12)
    ax.set_title("CIFAR-10  vs  CIFAR-100  —  k-NN Karşılaştırması",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle="--")

    plt.tight_layout()
    _kaydet(fig, kaydet_yolu)


def sonuclari_kaydet(k_degerleri, sonuclar, kaydet_yolu, ek_bilgi=None):
    os.makedirs(os.path.dirname(kaydet_yolu) or ".", exist_ok=True)
    metrikler = list(sonuclar.keys())

    with open(kaydet_yolu, "w", encoding="utf-8") as f:
        f.write("CIFAR  k-NN  Deney Sonuçları\n")
        f.write("=" * 45 + "\n\n")

        if ek_bilgi:
            for anahtar, deger in ek_bilgi.items():
                f.write(f"  {anahtar:<24}: {deger}\n")
            f.write("\n")

        f.write(f"  {'k':>4}")
        for m in metrikler:
            f.write(f"  {m.upper():>10}")
        f.write("\n  " + "-" * (8 + 12 * len(metrikler)) + "\n")

        for i, k in enumerate(k_degerleri):
            f.write(f"  {k:>4}")
            for m in metrikler:
                f.write(f"  {sonuclar[m][i]*100:>9.2f}%")
            f.write("\n")

        f.write("\n")
        for m in metrikler:
            bi = int(np.argmax(sonuclar[m]))
            f.write(
                f"  En iyi {m.upper():<4}: "
                f"k={k_degerleri[bi]}  →  {sonuclar[m][bi]*100:.2f}%\n"
            )

    print(f"  [✔] Sonuçlar kaydedildi: {kaydet_yolu}")
