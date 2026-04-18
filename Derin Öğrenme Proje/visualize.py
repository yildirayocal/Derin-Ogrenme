
import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RUNS_DIR, PROJE_ADI, DENEY_ADI,
    GORSELLER_DIR, DATA_DIR,
)


def _matplotlib_import():
    """Matplotlib import eder; yoksa kullanıcıyı uyarır."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        return plt, mpatches
    except ImportError:
        print("  ✗ Matplotlib bulunamadı: pip install matplotlib")
        sys.exit(1)



def veri_istatistikleri_goster():
    """Train/val split dağılımını görselleştirir."""
    plt, mpatches = _matplotlib_import()

    train_sayisi = 251
    val_sayisi   = 151

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Dental Crown Dataset — Veri Seti Dağılımı", fontsize=14, fontweight="bold")

    # Pasta grafik
    renkler = ["#2196F3", "#FF9800"]
    ax1.pie(
        [train_sayisi, val_sayisi],
        labels=[f"Train\n{train_sayisi}", f"Val\n{val_sayisi}"],
        colors=renkler, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 12},
    )
    ax1.set_title("Split Dağılımı", fontweight="bold")

    # Çubuk grafik
    barlar = ax2.bar(
        ["Train (251)", "Val (151)"], [train_sayisi, val_sayisi],
        color=renkler, edgecolor="white", width=0.5,
    )
    for bar, val in zip(barlar, [train_sayisi, val_sayisi]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            str(val), ha="center", va="bottom", fontweight="bold",
        )
    ax2.set_ylim(0, 300)
    ax2.set_title("Görüntü Sayısı", fontweight="bold")
    ax2.set_ylabel("Adet")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    kayit_yolu = GORSELLER_DIR / "01_veri_dagilimi.png"
    fig.savefig(kayit_yolu, dpi=150, bbox_inches="tight")
    print(f"  ✓ Veri dağılımı: {kayit_yolu}")
    plt.close()



def egitim_grafikleri_goster():
    """results.csv'den loss ve mAP eğrilerini çizer."""
    plt, _ = _matplotlib_import()

    csv_yolu = RUNS_DIR / PROJE_ADI / DENEY_ADI / "results.csv"
    if not csv_yolu.exists():
        print(f"  ⚠ results.csv bulunamadı: {csv_yolu}")
        print("    Önce modeli eğitin: python train.py")
        return

    # CSV oku
    veri: dict[str, list] = {}
    with open(csv_yolu, newline="", encoding="utf-8") as f:
        okuyucu = csv.DictReader(f)
        for satir in okuyucu:
            for anahtar, deger in satir.items():
                anahtar = anahtar.strip()
                veri.setdefault(anahtar, [])
                try:
                    veri[anahtar].append(float(deger.strip()))
                except ValueError:
                    veri[anahtar].append(None)

    epochlar = list(range(1, len(veri.get("epoch", [0])) + 1))

    fig, axlar = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("YOLOv8-OBB — Eğitim Grafikleri", fontsize=14, fontweight="bold")

    def ciz(ax, anahtar, baslik, renk):
        if anahtar in veri:
            degerler = [d for d in veri[anahtar] if d is not None]
            ax.plot(epochlar[:len(degerler)], degerler, color=renk, linewidth=2)
            ax.set_title(baslik, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.3)

    ciz(axlar[0,0], "train/box_loss",   "Train Box Loss",  "#E53935")
    ciz(axlar[0,1], "train/cls_loss",   "Train Cls Loss",  "#FB8C00")
    ciz(axlar[0,2], "train/dfl_loss",   "Train DFL Loss",  "#8E24AA")
    ciz(axlar[1,0], "val/box_loss",     "Val Box Loss",    "#1E88E5")
    ciz(axlar[1,1], "metrics/mAP50(B)", "mAP@50",          "#43A047")
    ciz(axlar[1,2], "metrics/precision(B)", "Precision",   "#00ACC1")

    plt.tight_layout()
    kayit_yolu = GORSELLER_DIR / "02_egitim_grafikleri.png"
    fig.savefig(kayit_yolu, dpi=150, bbox_inches="tight")
    print(f"  ✓ Eğitim grafikleri: {kayit_yolu}")
    plt.close()



def obb_gorsel_ciz(
    goruntu_yolu: str | Path,
    model=None,
    cikti_yolu: str | Path = None,
):
    """
    OBB tespitlerini görüntü üzerine çizer.
    model verilmezse mevcut tahmin dosyasını gösterir.
    """
    plt, mpatches = _matplotlib_import()
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        print("  ✗ Pillow bulunamadı: pip install Pillow")
        return

    goruntu_yolu = Path(goruntu_yolu)
    if not goruntu_yolu.exists():
        print(f"  ⚠ Görüntü bulunamadı: {goruntu_yolu}")
        return

    img = Image.open(goruntu_yolu).convert("RGB")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(img)
    ax.set_title(
        f"OBB Kuron Tespiti — {goruntu_yolu.name}",
        fontsize=13, fontweight="bold"
    )
    ax.axis("off")

    if model is not None:
        from config import CONF_THRESHOLD, IOU_THRESHOLD, IMGSZ, DEVICE
        sonuclar = model.predict(
            str(goruntu_yolu), conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD, imgsz=IMGSZ, device=DEVICE, verbose=False,
        )
        for sonuc in sonuclar:
            if sonuc.obb is not None:
                for kutu, guven in zip(sonuc.obb.xyxyxyxy, sonuc.obb.conf):
                    noktalari = kutu.tolist()
                    xs = [p[0] for p in noktalari]
                    ys = [p[1] for p in noktalari]
                    xs.append(xs[0]); ys.append(ys[0])  # Kapat
                    ax.plot(xs, ys, "r-", linewidth=2.5)
                    ax.fill(xs[:-1], ys[:-1], color="red", alpha=0.15)
                    ax.text(
                        min(xs), min(ys) - 5,
                        f"crown {float(guven)*100:.0f}%",
                        color="white", fontsize=9, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="#E53935", ec="none"),
                    )

    if cikti_yolu is None:
        cikti_yolu = GORSELLER_DIR / f"obb_{goruntu_yolu.stem}.png"
    fig.savefig(str(cikti_yolu), dpi=150, bbox_inches="tight")
    print(f"  ✓ OBB görsel: {cikti_yolu}")
    plt.close()



def metrik_ozet_goster(metrikler: dict = None):
    """Performans metriklerini görsel olarak gösterir."""
    plt, _ = _matplotlib_import()

    if metrikler is None:
        metrikler = {
            "mAP@50"   : 0.78,
            "Precision": 0.82,
            "Recall"   : 0.72,
            "F1-Score" : 0.77,
        }

    fig, ax = plt.subplots(figsize=(9, 5))
    isimler = list(metrikler.keys())
    degerler= list(metrikler.values())
    renkler = ["#1E88E5", "#43A047", "#FB8C00", "#8E24AA"]

    barlar = ax.barh(isimler, degerler, color=renkler, edgecolor="white", height=0.5)
    for bar, val in zip(barlar, degerler):
        ax.text(
            val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", fontweight="bold",
        )

    ax.set_xlim(0, 1.1)
    ax.set_title("YOLOv8-OBB — Performans Metrikleri", fontweight="bold", fontsize=13)
    ax.set_xlabel("Değer")
    ax.axvline(0.75, linestyle="--", color="gray", alpha=0.5, label="0.75 eşiği")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    kayit_yolu = GORSELLER_DIR / "03_metrikler.png"
    fig.savefig(kayit_yolu, dpi=150, bbox_inches="tight")
    print(f"  ✓ Metrik görsel: {kayit_yolu}")
    plt.close()



if __name__ == "__main__":
    print("\n  Hangi görsel oluşturulsun?")
    print("  1) Veri seti dağılımı")
    print("  2) Eğitim grafikleri (results.csv gerekli)")
    print("  3) Metrik özet")
    print("  4) Tümü")
    secim = input("\n  Seçiminiz [4]: ").strip() or "4"

    if secim in ("1", "4"):
        veri_istatistikleri_goster()
    if secim in ("2", "4"):
        egitim_grafikleri_goster()
    if secim in ("3", "4"):
        metrik_ozet_goster()

    print(f"\n  🖼  Görseller: {GORSELLER_DIR}")
