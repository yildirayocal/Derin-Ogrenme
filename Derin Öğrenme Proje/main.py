
import sys
import os
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))


def baslik_yazdir():
    print("\n" + "╔" + "═" * 53 + "╗")
    print("║    Diş Protez Kuron Tespit Sistemi              ║")
    print("║      YOLOv8-OBB  ·  Roboflow  ·  Ultralytics     ║")
    print("╚" + "═" * 53 + "╝")
    from config import DEVICE_TR, TOTAL_IMAGES, MODEL_ADI
    print(f"  Cihaz : {DEVICE_TR}  |  Veri: {TOTAL_IMAGES} X-ray  |  Model: {MODEL_ADI}")
    print()


def menu_yazdir():
    print("  ┌─── MENÜ ────────────────────────────────────┐")
    print("  │  1 │ NDJSON veri setini analiz et           │")
    print("  │  2 │ Veri setini indir (NDJSON / Roboflow)  │")
    print("  │  3 │ YOLOv8-OBB modelini eğit               │")
    print("  │  4 │ Tekil görüntü tahmini                  │")
    print("  │  5 │ Toplu klasör tahmini                   │")
    print("  │  6 │ Görselleştirme                         │")
    print("  │  0 │ Çıkış                                  │")
    print("  └────────────────────────────────────────────┘")


def adim1_ndjson_analiz():
    """NDJSON dosyasını analiz eder."""
    from config import NDJSON_DOSYA
    from ndjson_parser import NDJSONParser

    if not NDJSON_DOSYA.exists():
        print(f"\n  ✗ NDJSON dosyası bulunamadı: {NDJSON_DOSYA}")
        print("    Dosyayı proje klasörüne kopyalayın.")
        return

    parser = NDJSONParser(NDJSON_DOSYA)
    parser.ozet_yazdir()


    train_items = parser.url_listesi("train")
    annotasyonlu = [it for it in train_items if it["annotations"]]
    if annotasyonlu:
        print(f"\n   Örnek Annotasyon (train, 1. görüntü):")
        satirlar = parser.annotasyonlari_yolo_formatina_cevir(
            annotasyonlu[0]["annotations"]
        )
        for satir in satirlar[:3]:
            print(f"    {satir}")
        if len(satirlar) > 3:
            print(f"    ... ({len(satirlar)} toplam annotasyon)")


    from config import DATA_DIR, DATA_YAML
    secim = input("\n  data.yaml oluşturulsun mu? (e/h): ").strip().lower()
    if secim == "e":
        DATA_YAML.write_text(parser.yaml_icerik_olustur(DATA_DIR), encoding="utf-8")
        print(f"  ✓ data.yaml oluşturuldu: {DATA_YAML}")


def adim2_veri_indir():
    """Veri setini indirir."""
    print("\n  Hangi yöntemle indireyim?")
    print("  1) NDJSON (Ultralytics) — API anahtarı gerekmez")
    print("  2) Roboflow API — config.py'de API anahtarı gerekir")
    secim = input("\n  Seçiminiz [1]: ").strip() or "1"

    from dataset_setup import ndjson_ile_kur, roboflow_ile_kur, veri_seti_hazir_mi

    if veri_seti_hazir_mi():
        devam = input("  ⚠ Veri seti zaten mevcut. Tekrar indirilsin mi? (e/h): ").strip().lower()
        if devam != "e":
            return

    if secim == "2":
        roboflow_ile_kur()
    else:
        ndjson_ile_kur()


def adim3_egit():
    """Modeli eğitir."""
    from dataset_setup import veri_seti_hazir_mi
    if not veri_seti_hazir_mi():
        print("\n  ⚠ Veri seti bulunamadı. Önce indirin (Menü → 2)")
        return

    print("\n  Model boyutu seçin:")
    print("  1) Nano  — yolov8n-obb.pt  (hızlı, CPU uyumlu)")
    print("  2) Small — yolov8s-obb.pt  (önerilen)")
    print("  3) Medium — yolov8m-obb.pt (yüksek doğruluk)")
    secim = input("\n  Seçiminiz [1]: ").strip() or "1"
    model_map = {"1": "yolov8n-obb.pt", "2": "yolov8s-obb.pt", "3": "yolov8m-obb.pt"}

    from train import egit, model_dogrula
    best_pt = egit(model_adi=model_map.get(secim, "yolov8n-obb.pt"))

    dogrula = input("\n  Doğrulama yapılsın mı? (e/h): ").strip().lower()
    if dogrula == "e":
        metrikler = model_dogrula(best_pt)
        print("\n   Doğrulama Metrikleri:")
        for k, v in metrikler.items():
            print(f"    {k:12s}: {v:.4f}")


def adim4_tekil_tahmin():
    """Tekil görüntü tahmini yapar."""
    from predict import model_yukle, goruntu_tahmin, tahmin_yazdir, rapor_kaydet
    model = model_yukle()
    goruntu_yolu = input("\n  Görüntü yolu (.jpg/.png): ").strip()
    sonuc = goruntu_tahmin(model, goruntu_yolu)
    tahmin_yazdir(sonuc)

    kayit = input("  JSON raporu kaydedilsin mi? (e/h): ").strip().lower()
    if kayit == "e":
        rapor_kaydet(sonuc, "tekil_rapor.json")


def adim5_toplu_tahmin():
    """Toplu klasör tahmini yapar."""
    from predict import model_yukle, toplu_tahmin, rapor_kaydet
    model = model_yukle()
    klasor = input("\n  Görüntü klasörü yolu: ").strip()
    ozet = toplu_tahmin(model, klasor)
    if ozet:
        print(f"\n   Toplu Tahmin Özeti:")
        print(f"    Toplam görüntü  : {ozet['toplam_goruntu']}")
        print(f"    Toplam kuron    : {ozet['toplam_kuron']}")
        print(f"    Ort/görüntü    : {ozet['ortalama_kuron']}")
        print(f"    Maks kuron     : {ozet['maks_kuron']}")
        print(f"    Toplam süre    : {ozet['sure_saniye']}s")
        print(f"    Görüntü başı   : {ozet['goruntu_basi_ms']}ms")
        rapor_kaydet(ozet, "toplu_rapor.json")


def adim6_gorsellestirilme():
    """Görselleştirme menüsü."""
    print("\n  Hangi görsel oluşturulsun?")
    print("  1) Veri seti dağılımı")
    print("  2) Eğitim grafikleri (results.csv gerekli)")
    print("  3) Metrik özet")
    print("  4) Tümü")
    secim = input("\n  Seçiminiz [4]: ").strip() or "4"

    from visualize import (
        veri_istatistikleri_goster,
        egitim_grafikleri_goster,
        metrik_ozet_goster,
    )
    from config import GORSELLER_DIR

    if secim in ("1", "4"): veri_istatistikleri_goster()
    if secim in ("2", "4"): egitim_grafikleri_goster()
    if secim in ("3", "4"): metrik_ozet_goster()
    print(f"\n  🖼  Görseller kaydedildi: {GORSELLER_DIR}")


# ─── Ana Döngü ────────────────────────────────────────────────────────────────
ADIMLAR = {
    "1": ("NDJSON Analizi",          adim1_ndjson_analiz),
    "2": ("Veri İndir",              adim2_veri_indir),
    "3": ("Modeli Eğit",             adim3_egit),
    "4": ("Tekil Tahmin",            adim4_tekil_tahmin),
    "5": ("Toplu Tahmin",            adim5_toplu_tahmin),
    "6": ("Görselleştirme",          adim6_gorsellestirilme),
}


if __name__ == "__main__":
    baslik_yazdir()
    while True:
        menu_yazdir()
        secim = input("\n  Seçiminiz: ").strip()

        if secim == "0":
            print("\n  👋 Çıkılıyor...\n")
            sys.exit(0)

        if secim not in ADIMLAR:
            print("  ⚠ Geçersiz seçim. Lütfen 0-6 arasında bir sayı girin.")
            continue

        ad, fonksiyon = ADIMLAR[secim]
        print(f"\n  {'─' * 50}")
        print(f"  ▶ {ad}")
        print(f"  {'─' * 50}")
        try:
            fonksiyon()
        except KeyboardInterrupt:
            print("\n  ⚠ İşlem iptal edildi.")
        except Exception as e:
            print(f"\n  ✗ Hata: {e}")
            import traceback
            traceback.print_exc()

        input("\n  [Enter] ile devam edin...")
