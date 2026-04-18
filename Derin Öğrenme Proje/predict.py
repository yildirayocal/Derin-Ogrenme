
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RUNS_DIR, PROJE_ADI, DENEY_ADI, TAHMINLER_DIR,
    CONF_THRESHOLD, IOU_THRESHOLD, MAX_DET,
    IMGSZ, DEVICE,
)


def model_yukle(best_pt: str | Path = None):
    """
    En iyi model ağırlıklarını yükler.
    best_pt belirtilmezse otomatik olarak runs/ klasöründen bulur.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ✗ Ultralytics kütüphanesi bulunamadı!")
        print("    pip install ultralytics")
        sys.exit(1)

    if best_pt is None:
        # Otomatik bul
        aradigin = list(Path(RUNS_DIR / PROJE_ADI).rglob("best.pt"))
        if not aradigin:
            print(f"  ✗ best.pt bulunamadı: {RUNS_DIR / PROJE_ADI}")
            print("    Önce modeli eğitin: python train.py")
            sys.exit(1)
        # En son kaydedileni seç
        best_pt = max(aradigin, key=lambda p: p.stat().st_mtime)

    print(f"  ✓ Model yükleniyor: {best_pt}")
    model = YOLO(str(best_pt))
    print(f"  ✓ Model hazır — Görev: {model.task}")
    return model


def goruntu_tahmin(model, goruntu_yolu: str | Path) -> dict:
    """
    Tek bir görüntü üzerinde OBB kuron tespiti yapar.

    Returns:
        dict: {
            dosya, kuron_sayisi, sure_ms,
            tespitler: [{sinif, guven, obb_koordinatlar}]
        }
    """
    yol = Path(goruntu_yolu)
    if not yol.exists():
        return {"hata": f"Dosya bulunamadı: {yol}"}

    t0 = time.time()
    sonuclar = model.predict(
        source   = str(yol),
        conf     = CONF_THRESHOLD,
        iou      = IOU_THRESHOLD,
        imgsz    = IMGSZ,
        max_det  = MAX_DET,
        device   = DEVICE,
        save     = True,
        save_dir = str(TAHMINLER_DIR / "tahmin"),
        verbose  = False,
    )
    sure_ms = (time.time() - t0) * 1000

    tespitler = []
    for sonuc in sonuclar:
        if sonuc.obb is not None:
            for i, (kutu, guven, sinif_id) in enumerate(
                zip(sonuc.obb.xyxyxyxy, sonuc.obb.conf, sonuc.obb.cls)
            ):
                tespitler.append({
                    "id"             : i + 1,
                    "sinif"          : sonuc.names[int(sinif_id)],
                    "guven"          : round(float(guven) * 100, 1),
                    "obb_koordinatlar": kutu.tolist(),
                })

    return {
        "dosya"       : yol.name,
        "kuron_sayisi": len(tespitler),
        "sure_ms"     : round(sure_ms, 1),
        "tespitler"   : tespitler,
        "kayit_yeri"  : str(TAHMINLER_DIR / "tahmin"),
    }


def toplu_tahmin(model, klasor: str | Path) -> dict:
    """
    Bir klasördeki tüm görüntüleri toplu olarak işler.
    """
    klasor = Path(klasor)
    goruntu_dosyalari = list(klasor.glob("*.jpg")) + list(klasor.glob("*.png"))

    if not goruntu_dosyalari:
        print(f"  ⚠ Klasörde görüntü bulunamadı: {klasor}")
        return {}

    print(f"\n  📂 Toplu tahmin: {len(goruntu_dosyalari)} görüntü")
    t0 = time.time()

    sonuclar = model.predict(
        source   = str(klasor),
        conf     = CONF_THRESHOLD,
        iou      = IOU_THRESHOLD,
        imgsz    = IMGSZ,
        device   = DEVICE,
        save     = True,
        save_dir = str(TAHMINLER_DIR / "toplu"),
        verbose  = False,
    )

    sure = time.time() - t0
    tespitler_listesi = []
    toplam_kuron = 0

    for sonuc in sonuclar:
        sayi = len(sonuc.obb) if sonuc.obb is not None else 0
        toplam_kuron += sayi
        tespitler_listesi.append({
            "dosya"       : Path(sonuc.path).name,
            "kuron_sayisi": sayi,
        })

    ozet = {
        "toplam_goruntu" : len(goruntu_dosyalari),
        "toplam_kuron"   : toplam_kuron,
        "ortalama_kuron" : round(toplam_kuron / max(len(goruntu_dosyalari), 1), 1),
        "maks_kuron"     : max((t["kuron_sayisi"] for t in tespitler_listesi), default=0),
        "sure_saniye"    : round(sure, 1),
        "goruntu_basi_ms": round(sure * 1000 / max(len(goruntu_dosyalari), 1), 1),
        "dagilim": {
            "0 kuron"    : sum(1 for t in tespitler_listesi if t["kuron_sayisi"] == 0),
            "1-3 kuron"  : sum(1 for t in tespitler_listesi if 1 <= t["kuron_sayisi"] <= 3),
            "4+ kuron"   : sum(1 for t in tespitler_listesi if t["kuron_sayisi"] >= 4),
        },
        "tespitler"      : tespitler_listesi,
    }
    return ozet


def tahmin_yazdir(sonuc: dict):
    """Tekil tahmin sonucunu terminale güzel formatta yazdırır."""
    if "hata" in sonuc:
        print(f"  ✗ {sonuc['hata']}")
        return

    print("\n" + "═" * 45)
    print(f"  Dosya      : {sonuc['dosya']}")
    print(f"  Kuron sayısı: {sonuc['kuron_sayisi']} adet  ★")
    print(f"  Süre       : {sonuc['sure_ms']}ms")
    print()
    for t in sonuc["tespitler"]:
        bar = "▓" * int(t["guven"] / 10) + "░" * (10 - int(t["guven"] / 10))
        print(f"  [{t['id']}] {t['sinif']:8s}  güven: {t['guven']:5.1f}%  {bar}")
    print("═" * 45)
    print(f"  Kaydedildi: {sonuc['kayit_yeri']}")


def rapor_kaydet(sonuc: dict, dosya_adi: str = "analiz_raporu.json"):
    """Tahmin sonuçlarını JSON dosyasına kaydeder."""
    hedef = TAHMINLER_DIR / dosya_adi
    with open(hedef, "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 Rapor kaydedildi: {hedef}")



if __name__ == "__main__":
    model = model_yukle()

    print("\n  Tahmin modu:")
    print("  1) Tekil görüntü tahmini")
    print("  2) Klasör tahmini (toplu)")
    secim = input("\n  Seçiminiz [1]: ").strip() or "1"

    if secim == "2":
        klasor = input("  Klasör yolu: ").strip()
        ozet = toplu_tahmin(model, klasor)
        if ozet:
            print(f"\n  📊 Özet:")
            print(f"    Toplam görüntü  : {ozet['toplam_goruntu']}")
            print(f"    Toplam kuron    : {ozet['toplam_kuron']}")
            print(f"    Ort/görüntü    : {ozet['ortalama_kuron']}")
            print(f"    Süre           : {ozet['sure_saniye']}s")
            rapor_kaydet(ozet)
    else:
        goruntu_yolu = input("  Görüntü yolu (.jpg/.png): ").strip()
        sonuc = goruntu_tahmin(model, goruntu_yolu)
        tahmin_yazdir(sonuc)
        rapor_kaydet(sonuc, "tekil_rapor.json")
