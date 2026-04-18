

import os
import sys
import time
import shutil
import requests
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue


sys.path.insert(0, str(Path(__file__).parent))
from config import (
    NDJSON_DOSYA, DATA_DIR, DATA_YAML,
    IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL,
    DOWNLOAD_WORKERS, DOWNLOAD_TIMEOUT,
    ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE,
    ROBOFLOW_PROJECT, ROBOFLOW_VERSION, ROBOFLOW_FORMAT,
)
from ndjson_parser import NDJSONParser




def goruntu_indir(item: dict, hedef_klasor: Path) -> tuple[bool, str]:
    """Tek bir görüntüyü URL'den indirir."""
    hedef = hedef_klasor / item["dosya"]
    if hedef.exists():
        return True, f"✓ Mevcut: {item['dosya']}"

    try:
        r = requests.get(item["url"], timeout=DOWNLOAD_TIMEOUT, stream=True)
        r.raise_for_status()
        with open(hedef, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return True, f"↓ İndirildi: {item['dosya']}"
    except Exception as e:
        return False, f"✗ Hata ({item['dosya']}): {e}"


def label_olustur(item: dict, label_klasor: Path, parser: NDJSONParser):
    """Görüntü için YOLO OBB .txt label dosyası oluşturur."""
    dosya_adi = Path(item["dosya"]).stem
    label_yol = label_klasor / f"{dosya_adi}.txt"

    if not label_yol.exists():
        satirlar = parser.annotasyonlari_yolo_formatina_cevir(item["annotations"])
        label_yol.write_text("\n".join(satirlar), encoding="utf-8")


def ndjson_ile_kur(ndjson_yolu: Path = NDJSON_DOSYA) -> bool:
    """
    NDJSON dosyasından veri setini indirir ve hazırlar.
    Returns: True başarılı, False başarısız
    """
    print("\n" + "=" * 55)
    print("  📥 NDJSON İLE VERİ SETİ HAZIRLAMA")
    print("=" * 55)

    if not ndjson_yolu.exists():
        print(f"\n  ✗ NDJSON dosyası bulunamadı: {ndjson_yolu}")
        print("    Lütfen dosyayı proje klasörüne kopyalayın.")
        return False

    parser = NDJSONParser(ndjson_yolu)
    parser.ozet_yazdir()

    # data.yaml oluştur
    DATA_YAML.write_text(parser.yaml_icerik_olustur(DATA_DIR), encoding="utf-8")
    print(f"\n  ✓ data.yaml oluşturuldu: {DATA_YAML}")

    # Split bazında indirme + label oluşturma
    for split in ("train", "val"):
        klasor     = IMAGES_TRAIN if split == "train" else IMAGES_VAL
        lbl_klasor = LABELS_TRAIN if split == "train" else LABELS_VAL
        items      = parser.url_listesi(split)

        print(f"\n  [{split.upper()}] {len(items)} görüntü indiriliyor "
              f"({DOWNLOAD_WORKERS} paralel iş parçacığı)...")

        basarili = 0
        hatali   = 0
        t0       = time.time()

        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
            futures = {
                ex.submit(goruntu_indir, item, klasor): item
                for item in items
            }
            for i, fut in enumerate(as_completed(futures), 1):
                ok, mesaj = fut.result()
                item = futures[fut]
                if ok:
                    basarili += 1
                    label_olustur(item, lbl_klasor, parser)
                else:
                    hatali += 1
                    print(f"    {mesaj}")

                # İlerleme çubuğu
                yuzde = i / len(items)
                bar   = "█" * int(yuzde * 30) + "░" * (30 - int(yuzde * 30))
                print(f"\r    [{bar}] {i}/{len(items)} ({yuzde*100:.0f}%)", end="")

        sure = time.time() - t0
        print(f"\n    ✓ Tamamlandı: {basarili} başarılı, {hatali} hatalı — {sure:.1f}s")

    _boyut_yazdir()
    return True




def roboflow_ile_kur(
    api_key: str = ROBOFLOW_API_KEY,
    workspace: str = ROBOFLOW_WORKSPACE,
    project: str   = ROBOFLOW_PROJECT,
    version: int   = ROBOFLOW_VERSION,
    format_: str   = ROBOFLOW_FORMAT,
) -> bool:

    print("\n" + "=" * 55)
    print("  🔵 ROBOFLOW İLE VERİ SETİ HAZIRLAMA")
    print("=" * 55)

    try:
        from roboflow import Roboflow
    except ImportError:
        print("\n  ✗ Roboflow kütüphanesi bulunamadı.")
        print("    Kurmak için: pip install roboflow")
        return False

    if api_key == "YOUR_API_KEY_HERE":
        print("\n  ✗ Roboflow API anahtarı girilmemiş!")
        print("    config.py → ROBOFLOW_API_KEY değerini doldurun.")
        print("    ya da: export ROBOFLOW_API_KEY='xxxx'")
        return False

    try:
        print(f"\n  Bağlanıyor: {workspace}/{project} (v{version})...")
        rf        = Roboflow(api_key=api_key)
        proje_obj = rf.workspace(workspace).project(project)
        dataset   = proje_obj.version(version).download(
            format_, location=str(DATA_DIR)
        )
        print(f"\n  ✓ Roboflow veri seti indirildi: {DATA_DIR}")
        print(f"    Konum: {dataset.location}")

        # data.yaml'ı kontrol et
        yaml_dosya = Path(dataset.location) / "data.yaml"
        if yaml_dosya.exists():
            shutil.copy(yaml_dosya, DATA_YAML)
            print(f"  ✓ data.yaml kopyalandı: {DATA_YAML}")
        else:
            print("  ⚠ data.yaml bulunamadı, NDJSON parser ile oluşturuluyor...")
            parser = NDJSONParser(NDJSON_DOSYA)
            DATA_YAML.write_text(parser.yaml_icerik_olustur(DATA_DIR), encoding="utf-8")

        _boyut_yazdir()
        return True

    except Exception as e:
        print(f"\n  ✗ Roboflow hatası: {e}")
        return False




def _boyut_yazdir():
    """İndirilen klasörlerin boyutunu ve dosya sayısını yazdırır."""
    print("\n  📁 İndirilen Dosyalar:")
    for klasor, ad in [(IMAGES_TRAIN, "train görseller"),
                       (IMAGES_VAL,   "val görseller"),
                       (LABELS_TRAIN, "train labellar"),
                       (LABELS_VAL,   "val labellar")]:
        dosyalar = list(klasor.iterdir()) if klasor.exists() else []
        print(f"    {ad:20s}: {len(dosyalar):4d} dosya — {klasor}")


def veri_seti_hazir_mi() -> bool:
    """Veri setinin indirilmiş olup olmadığını kontrol eder."""
    train_sayisi = len(list(IMAGES_TRAIN.glob("*.jpg"))) if IMAGES_TRAIN.exists() else 0
    val_sayisi   = len(list(IMAGES_VAL.glob("*.jpg")))   if IMAGES_VAL.exists()   else 0
    return train_sayisi > 0 and val_sayisi > 0 and DATA_YAML.exists()



if __name__ == "__main__":
    print("\n  Hangi yöntemi kullanmak istersiniz?")
    print("  1) NDJSON (Ultralytics) yolu — API anahtarı gerekmez")
    print("  2) Roboflow API — config.py'de API anahtarı gerekir")
    secim = input("\n  Seçiminiz (1/2): ").strip()

    if secim == "2":
        basarili = roboflow_ile_kur()
    else:
        basarili = ndjson_ile_kur()

    if basarili:
        print("\n  🎉 Veri seti hazır! Eğitim için: python train.py")
    else:
        print("\n  ✗ Veri hazırlama başarısız. Hataları kontrol edin.")
