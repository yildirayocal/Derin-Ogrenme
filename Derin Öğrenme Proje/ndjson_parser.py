

import json
from pathlib import Path
from typing import Optional


class NDJSONParser:
    """
    Ultralytics NDJSON formatındaki diş röntgeni veri setini okur.

    NDJSON Yapısı:
        Satır 1 : {"type": "dataset", "task": "obb", "class_names": {...}, ...}
        Satır 2+: {"type": "image", "file": "xxx.jpg", "url": "...", "split": "train",
                   "annotations": {"obb": [[cls, x1,y1,x2,y2,x3,y3,x4,y4], ...]}}
    """

    def __init__(self, ndjson_yolu: str | Path):
        self.yol = Path(ndjson_yolu)
        self.meta: dict = {}
        self.goruntular: list[dict] = []
        self._oku()

    # ─── Okuma ───────────────────────────────────────────────────────────────

    def _oku(self):
        """NDJSON dosyasını satır satır okur."""
        if not self.yol.exists():
            raise FileNotFoundError(f"NDJSON dosyası bulunamadı: {self.yol}")

        with open(self.yol, "r", encoding="utf-8") as f:
            for i, satir in enumerate(f):
                satir = satir.strip()
                if not satir:
                    continue
                try:
                    kayit = json.loads(satir)
                except json.JSONDecodeError as e:
                    print(f"  ⚠  Satır {i+1} atlandı (JSON hatası): {e}")
                    continue

                if kayit.get("type") == "dataset":
                    self.meta = kayit
                elif kayit.get("type") == "image":
                    self.goruntular.append(kayit)


    def ozet(self) -> dict:
        """Veri seti özetini döndürür."""
        train = [g for g in self.goruntular if g.get("split") == "train"]
        val   = [g for g in self.goruntular if g.get("split") == "val"]
        return {
            "toplam"     : len(self.goruntular),
            "train"      : len(train),
            "val"        : len(val),
            "siniflar"   : self.meta.get("class_names", {"0": "crown"}),
            "gorev"      : self.meta.get("task", "obb"),
            "veri_seti"  : self.meta.get("name", "dental-crown"),
        }

    def goruntuleri_listele(self, split: Optional[str] = None) -> list[dict]:
        """Split'e göre görüntü kayıtlarını döndürür."""
        if split is None:
            return self.goruntular
        return [g for g in self.goruntular if g.get("split") == split]

    def url_listesi(self, split: Optional[str] = None) -> list[dict]:
        """
        İndirme için hazır URL listesi döndürür.
        Her eleman: {"url": str, "dosya": str, "split": str, "annotations": list}
        """
        sonuc = []
        for g in self.goruntuleri_listele(split):
            if "url" not in g:
                continue
            sonuc.append({
                "url"        : g["url"],
                "dosya"      : g.get("file", Path(g["url"]).name),
                "split"      : g.get("split", "train"),
                "genislik"   : g.get("width", 640),
                "yukseklik"  : g.get("height", 640),
                "annotations": g.get("annotations", {}).get("obb", []),
            })
        return sonuc

    def annotasyonlari_yolo_formatina_cevir(self, annotations: list) -> list[str]:
        """
        NDJSON OBB annotasyonlarını YOLO OBB .txt formatına çevirir.
        NDJSON: [class_id, x1, y1, x2, y2, x3, y3, x4, y4]
        YOLO  : class_id x1 y1 x2 y2 x3 y3 x4 y4
        """
        satirlar = []
        for ann in annotations:
            if len(ann) >= 9:
                cls_id = int(ann[0])
                coords = " ".join(f"{v:.6f}" for v in ann[1:9])
                satirlar.append(f"{cls_id} {coords}")
        return satirlar

    def yaml_icerik_olustur(self, data_dizin: str | Path) -> str:
        """data.yaml içeriği oluşturur."""
        s = self.ozet()
        sinif_isimleri = list(s["siniflar"].values())
        return (
            f"path: {Path(data_dizin).resolve()}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"nc: {len(sinif_isimleri)}\n"
            f"names: {sinif_isimleri}\n"
        )


    def ozet_yazdir(self):
        """Veri seti özetini terminale yazdırır."""
        s = self.ozet()
        print("=" * 50)
        print("   NDJSON VERİ SETİ ÖZETİ")
        print("=" * 50)
        print(f"  Veri Seti  : {s['veri_seti']}")
        print(f"  Görev      : {s['gorev'].upper()}")
        print(f"  Toplam     : {s['toplam']} görüntü")
        print(f"  Train      : {s['train']} (%{s['train']/max(s['toplam'],1)*100:.1f})")
        print(f"  Val        : {s['val']} (%{s['val']/max(s['toplam'],1)*100:.1f})")
        print(f"  Sınıflar   : {s['siniflar']}")
        print("=" * 50)

        # Annotasyon istatistikleri
        toplam_ann = sum(
            len(g.get("annotations", {}).get("obb", []))
            for g in self.goruntular
        )
        print(f"  Toplam OBB : {toplam_ann} annotasyon")
        if self.goruntular:
            ort = toplam_ann / len(self.goruntular)
            print(f"  Ort/Görüntü: {ort:.1f} kuron")
        print("=" * 50)

    def __repr__(self) -> str:
        s = self.ozet()
        return (f"NDJSONParser(toplam={s['toplam']}, "
                f"train={s['train']}, val={s['val']}, "
                f"görev={s['gorev']})")



if __name__ == "__main__":
    from config import NDJSON_DOSYA
    parser = NDJSONParser(NDJSON_DOSYA)
    parser.ozet_yazdir()
    print(f"\nİlk 3 URL:")
    for item in parser.url_listesi("train")[:3]:
        print(f"  {item['dosya']} → {item['url'][:60]}...")
