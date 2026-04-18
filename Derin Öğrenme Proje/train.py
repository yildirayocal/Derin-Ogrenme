
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_YAML, MODEL_ADI, PROJE_ADI, DENEY_ADI, RUNS_DIR,
    EPOCH, BATCH, IMGSZ, LR0, LRF, MOMENTUM, WEIGHT_DECAY,
    WARMUP_EPOCH, PATIENCE, MOSAIC, FLIPLR, FLIPUD, DEGREES, SCALE,
    DEVICE, DEVICE_TR,
)


def egit(
    model_adi: str = MODEL_ADI,
    epoch: int     = EPOCH,
    batch: int     = BATCH,
) -> Path:
    """
    YOLOv8-OBB modelini eğitir.

    Args:
        model_adi : Başlangıç ağırlıkları (yolov8n-obb.pt / yolov8s-obb.pt vb.)
        epoch     : Eğitim turu sayısı
        batch     : Batch boyutu

    Returns:
        best_pt   : En iyi model ağırlıklarının yolu (best.pt)
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ✗ Ultralytics kütüphanesi bulunamadı!")
        print("    Kurmak için: pip install ultralytics")
        sys.exit(1)

    if not DATA_YAML.exists():
        print(f"  ✗ data.yaml bulunamadı: {DATA_YAML}")
        print("    Önce veri setini hazırlayın: python dataset_setup.py")
        sys.exit(1)

    print("\n" + "=" * 55)
    print("  🚀 YOLOv8-OBB EĞİTİMİ BAŞLIYOR")
    print("=" * 55)
    print(f"  Model    : {model_adi}")
    print(f"  Veri     : {DATA_YAML}")
    print(f"  Epoch    : {epoch}")
    print(f"  Batch    : {batch}")
    print(f"  Görüntü  : {IMGSZ}×{IMGSZ}")
    print(f"  Cihaz    : {DEVICE_TR} ({DEVICE})")
    print(f"  Proje    : {PROJE_ADI}/{DENEY_ADI}")
    print("=" * 55)


    model = YOLO(model_adi)
    print(f"\n  ✓ Transfer öğrenme başlatıldı: {model_adi}")
    print(f"    Backbone: CSP-Darknet53 + C2f")
    print(f"    Head    : OBB (Oriented Bounding Box)")


    t0 = time.time()
    sonuclar = model.train(
        data        = str(DATA_YAML),
        epochs      = epoch,
        batch       = batch,
        imgsz       = IMGSZ,
        device      = DEVICE,
        project     = str(RUNS_DIR / PROJE_ADI),
        name        = DENEY_ADI,
        exist_ok    = True,

        # Optimizer & LR
        lr0         = LR0,
        lrf         = LRF,
        momentum    = MOMENTUM,
        weight_decay= WEIGHT_DECAY,
        warmup_epochs= WARMUP_EPOCH,

        # Augmentation
        mosaic      = MOSAIC,
        fliplr      = FLIPLR,
        flipud      = FLIPUD,
        degrees     = DEGREES,
        scale       = SCALE,

        # Early stopping & loglama
        patience    = PATIENCE,
        save        = True,
        save_period = 10,      # Her 10 epoch checkpoint
        plots       = True,
        verbose     = True,
    )

    sure = (time.time() - t0) / 60
    best_pt = Path(sonuclar.save_dir) / "weights" / "best.pt"

    print("\n" + "=" * 55)
    print("  🎉 EĞİTİM TAMAMLANDI")
    print("=" * 55)
    print(f"  Süre       : {sure:.1f} dakika")
    print(f"  En iyi pt  : {best_pt}")

    # Metrikleri yazdır
    if hasattr(sonuclar, "results_dict"):
        r = sonuclar.results_dict
        print(f"  mAP@50     : {r.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  mAP@50-95  : {r.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"  Precision  : {r.get('metrics/precision(B)', 0):.4f}")
        print(f"  Recall     : {r.get('metrics/recall(B)', 0):.4f}")

    print("=" * 55)
    return best_pt


def model_dogrula(best_pt: Path) -> dict:
    """
    Eğitilmiş modeli val seti üzerinde doğrular.
    """
    from ultralytics import YOLO
    print(f"\n  🔍 Model doğrulanıyor: {best_pt}")
    model = YOLO(str(best_pt))
    metrics = model.val(
        data    = str(DATA_YAML),
        imgsz   = IMGSZ,
        device  = DEVICE,
        plots   = True,
        verbose = True,
    )
    return {
        "mAP50"     : metrics.box.map50,
        "mAP50_95"  : metrics.box.map,
        "precision" : metrics.box.p.mean(),
        "recall"    : metrics.box.r.mean(),
    }



if __name__ == "__main__":
    print("\n  Model seçimi:")
    print("  1) yolov8n-obb.pt — Nano  (hızlı, CPU uyumlu)")
    print("  2) yolov8s-obb.pt — Small (önerilen)")
    print("  3) yolov8m-obb.pt — Medium (yüksek doğruluk)")
    secim = input("\n  Seçiminiz [1]: ").strip() or "1"

    model_map = {
        "1": "yolov8n-obb.pt",
        "2": "yolov8s-obb.pt",
        "3": "yolov8m-obb.pt",
    }
    secilen_model = model_map.get(secim, "yolov8n-obb.pt")

    best_pt = egit(model_adi=secilen_model)

    dogrula = input("\n  Val seti doğrulaması yapılsın mı? (e/h): ").strip().lower()
    if dogrula == "e":
        metrikler = model_dogrula(best_pt)
        print("\n  📊 Doğrulama Metrikleri:")
        for k, v in metrikler.items():
            print(f"    {k:12s}: {v:.4f}")
