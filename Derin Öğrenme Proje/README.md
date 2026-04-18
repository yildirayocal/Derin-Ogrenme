# 🦷 Diş Protez Kuron Tespit Sistemi
**YOLOv8-OBB · Roboflow · Ultralytics**

Panoramik diş röntgenlerinde protez kronları OBB (Oriented Bounding Box) yöntemiyle tespit eden derin öğrenme sistemi.

---

## 🚀 Hızlı Başlangıç

```bash
# 1. Bağımlılıkları kur
pip install -r requirements.txt

# 2. NDJSON dosyasını proje klasörüne kopyala
cp dental-prosthetic-crown-detection-x-rays.ndjson ./

# 3. Çalıştır
python main.py
```

---

## 📁 Proje Yapısı

```
kron_tespit_projesi/
├── main.py                ← ▶ GİRİŞ NOKTASI (menü)
├── config.py              ← Merkezi konfigürasyon
├── ndjson_parser.py       ← NDJSON veri okuyucu
├── dataset_setup.py       ← İndirme (NDJSON / Roboflow)
├── train.py               ← YOLOv8-OBB eğitim
├── predict.py             ← Tahmin motoru
├── visualize.py           ← Görselleştirme
├── requirements.txt
├── dental-...-.ndjson     ← VERİ MANİFESTO
├── data/
│   ├── images/train/  (251 X-ray)
│   ├── images/val/   (151 X-ray)
│   ├── labels/train/ (OBB .txt)
│   ├── labels/val/
│   └── data.yaml
├── runs/dental_crown_obb/
│   └── yolov8n_run1/weights/best.pt
├── tahminler/
└── gorseller/
```

---

## 🔵 Roboflow Entegrasyonu

`config.py` dosyasında aşağıdaki değerleri doldurun:

```python
ROBOFLOW_API_KEY   = "YOUR_API_KEY"      # app.roboflow.com → Settings → API
ROBOFLOW_WORKSPACE = "your-workspace"
ROBOFLOW_PROJECT   = "dental-crown-obb"
ROBOFLOW_VERSION   = 1
```

Ya da ortam değişkeni olarak:
```bash
export ROBOFLOW_API_KEY="xxxx"
```

Ardından `main.py → Menü 2 → Seçim 2` ile Roboflow'dan otomatik indir.

---

## 🏋️ Model Eğitimi

| Model | Parametre | mAP@50 (DOTA) | Hız | Öneri |
|-------|-----------|---------------|-----|-------|
| yolov8n-obb | 3.1M | ~78% | ⚡ Hızlı | CPU / Başlangıç |
| yolov8s-obb | 11.4M | ~79% | ✓ İyi | **Genel kullanım** |
| yolov8m-obb | 26.4M | ~80% | Orta | Yüksek doğruluk |

```bash
python train.py   # 100 epoch, otomatik GPU/CPU tespiti
```

---

## 🔍 Tahmin

```bash
python predict.py     # Tekil veya toplu tahmin
```

JSON raporu otomatik olarak `tahminler/` klasörüne kaydedilir.

---

## 📊 Beklenen Metrikler (251 train görüntüsü)

| Metrik | Beklenen |
|--------|----------|
| mAP@50 | ~%70-78 |
| Precision | ~%80-85 |
| Recall | ~%68-75 |
| Çıkarım Süresi | <20ms |

---

## 📝 Lisans
Akademik proje — Derin Öğrenme Dersi Vize Ödevi
