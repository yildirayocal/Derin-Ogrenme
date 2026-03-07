import os
import pickle
import warnings
import numpy as np


CIFAR10_YOLU  = r"C:\Yıldıray\cifar-10-batches-py"   # CIFAR-10 klasörü
CIFAR100_YOLU = r"C:\Yıldıray\cifar-100-python"       # CIFAR-100 klasörü


CIFAR10_SINIFLARI = [
    "airplane",    # 0
    "automobile",  # 1
    "bird",        # 2
    "cat",         # 3
    "deer",        # 4
    "dog",         # 5
    "frog",        # 6
    "horse",       # 7
    "ship",        # 8
    "truck"        # 9
]


CIFAR100_INCE = [
    "apple", "aquarium_fish", "baby", "bear", "beaver",
    "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly",
    "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach",
    "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox",
    "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard",
    "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid",
    "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew",
    "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe",
    "whale", "willow_tree", "wolf", "woman", "worm"
]



CIFAR100_KABA = [
    "aquatic_mammals",       # 0
    "fish",                  # 1
    "flowers",               # 2
    "food_containers",       # 3
    "fruit_and_vegetables",  # 4
    "household_electrical",  # 5
    "household_furniture",   # 6
    "insects",               # 7
    "large_carnivores",      # 8
    "large_man-made_outdoor",# 9
    "large_natural_outdoor", # 10
    "large_omnivores",       # 11
    "medium_mammals",        # 12
    "non-insect_invertebrates",# 13
    "people",                # 14
    "reptiles",              # 15
    "small_mammals",         # 16
    "trees",                 # 17
    "vehicles_1",            # 18
    "vehicles_2"             # 19
]



def _pickle_oku(tam_yol):
    """Tek bir CIFAR pickle dosyasını açar ve sözlük döndürür."""
    with open(tam_yol, "rb") as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pickle.load(f, encoding="bytes")



class CIFARLoader:
    """
    CIFAR-10 ve CIFAR-100 için tek sınıf loader.

    Parametreler
    ------------
    dataset         : "cifar10" veya "cifar100"
    cifar100_etiket : "ince" (100 sınıf) veya "kaba" (20 üst-sınıf)
                      Sadece dataset="cifar100" ise geçerli.

    Özellikler
    ----------
    .sinif_isimleri : Yüklenen dataset'in sınıf isim listesi
    .n_sinif        : Sınıf sayısı (10 / 20 / 100)
    .dataset        : "cifar10" veya "cifar100"
    """

    def __init__(self, dataset="cifar10", cifar100_etiket="ince"):
        dataset = dataset.lower().strip()
        if dataset not in ("cifar10", "cifar100"):
            raise ValueError(
                f"Geçersiz dataset: '{dataset}'. "
                f"Seçenekler: 'cifar10', 'cifar100'"
            )
        if cifar100_etiket not in ("ince", "kaba"):
            raise ValueError(
                f"Geçersiz cifar100_etiket: '{cifar100_etiket}'. "
                f"Seçenekler: 'ince' (100 sınıf), 'kaba' (20 sınıf)"
            )

        self.dataset          = dataset
        self.cifar100_etiket  = cifar100_etiket

        if dataset == "cifar10":
            self.veri_yolu     = CIFAR10_YOLU
            self.sinif_isimleri = CIFAR10_SINIFLARI
            self.n_sinif        = 10
        else:
            self.veri_yolu     = CIFAR100_YOLU
            if cifar100_etiket == "ince":
                self.sinif_isimleri = CIFAR100_INCE
                self.n_sinif        = 100
            else:
                self.sinif_isimleri = CIFAR100_KABA
                self.n_sinif        = 20

        self._klasor_kontrol()



    def _klasor_kontrol(self):
        """Veri klasörünün ve gerekli dosyaların var olduğunu kontrol eder."""
        if not os.path.isdir(self.veri_yolu):
            raise FileNotFoundError(
                f"\n[HATA] Veri klasörü bulunamadı:\n"
                f"  {self.veri_yolu}\n\n"
                f"Lütfen veri setinin doğru konumda olduğunu doğrulayın.\n"
                f"  CIFAR-10  indirme : https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz\n"
                f"  CIFAR-100 indirme : https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
            )

        if self.dataset == "cifar10":
            beklenen = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]
        else:
            beklenen = ["train", "test"]

        eksik = [d for d in beklenen
                 if not os.path.isfile(os.path.join(self.veri_yolu, d))]
        if eksik:
            raise FileNotFoundError(
                f"\n[HATA] Eksik dosyalar: {eksik}\n"
                f"Klasör: {self.veri_yolu}"
            )



    def _cifar10_yukle(self):
        """CIFAR-10 için 5 batch'i birleştirip eğitim ve test verisini döndürür."""
        X_parcalar, y_parcalar = [], []
        for i in range(1, 6):
            yol  = os.path.join(self.veri_yolu, f"data_batch_{i}")
            veri = _pickle_oku(yol)
            X_parcalar.append(veri[b"data"].astype(np.float32))
            y_parcalar.append(np.array(veri[b"labels"], dtype=np.int64))

        X_egit = np.concatenate(X_parcalar, axis=0)   # (50000, 3072)
        y_egit = np.concatenate(y_parcalar, axis=0)   # (50000,)

        test_veri = _pickle_oku(os.path.join(self.veri_yolu, "test_batch"))
        X_test    = test_veri[b"data"].astype(np.float32)
        y_test    = np.array(test_veri[b"labels"], dtype=np.int64)

        return X_egit, y_egit, X_test, y_test


    def _cifar100_yukle(self):
        """
        CIFAR-100 için train ve test dosyasını okur.

        CIFAR-100 pickle anahtarları:
          b'data'          → (N, 3072) uint8
          b'fine_labels'   → 100 ince sınıf etiketi
          b'coarse_labels' → 20 kaba sınıf etiketi
        """
        etiket_anahtari = (b"fine_labels"
                           if self.cifar100_etiket == "ince"
                           else b"coarse_labels")

        egit_veri = _pickle_oku(os.path.join(self.veri_yolu, "train"))
        X_egit    = egit_veri[b"data"].astype(np.float32)
        y_egit    = np.array(egit_veri[etiket_anahtari], dtype=np.int64)

        test_veri = _pickle_oku(os.path.join(self.veri_yolu, "test"))
        X_test    = test_veri[b"data"].astype(np.float32)
        y_test    = np.array(test_veri[etiket_anahtari], dtype=np.int64)

        return X_egit, y_egit, X_test, y_test


    def yukle(
        self,
        normalize  = True,
        num_egitim = None,
        num_test   = None,
        karistir   = True,
        tohum      = 42
    ):
        """
        Veri setini diskten okur ve hazır numpy dizileri döndürür.

        Parametreler
        ------------
        normalize  : bool
            True → piksel [0,255] → [0.0, 1.0]
        num_egitim : int veya None
            None → tüm 50.000 eğitim örneği
            int  → ilk N örnek (hızlı deney için)
        num_test   : int veya None
            None → tüm 10.000 test örneği
            int  → ilk N örnek
        karistir   : bool
            True → eğitim verisini rastgele karıştır
        tohum      : int
            Tekrarlanabilirlik için rastgele tohum

        Dönüş
        -----
        X_egit : (N_egit, 3072) float32/float64
        y_egit : (N_egit,)      int64
        X_test : (N_test, 3072) float32/float64
        y_test : (N_test,)      int64
        """

        # Uygun yükleme fonksiyonunu seç
        if self.dataset == "cifar10":
            X_egit, y_egit, X_test, y_test = self._cifar10_yukle()
        else:
            X_egit, y_egit, X_test, y_test = self._cifar100_yukle()

        # Karıştır
        if karistir:
            rng  = np.random.default_rng(tohum)
            sira = rng.permutation(len(X_egit))
            X_egit = X_egit[sira]
            y_egit = y_egit[sira]

        # Alt-küme
        if num_egitim is not None:
            X_egit = X_egit[:num_egitim]
            y_egit = y_egit[:num_egitim]
        if num_test is not None:
            X_test = X_test[:num_test]
            y_test = y_test[:num_test]

        # Normalizasyon
        if normalize:
            X_egit = X_egit / 255.0
            X_test = X_test / 255.0

        # Özet
        self._ozet_yazdir(X_egit, y_egit, X_test, y_test)

        return X_egit, y_egit, X_test, y_test


    def _ozet_yazdir(self, X_egit, y_egit, X_test, y_test):
        etiket_aciklama = (
            ""
            if self.dataset == "cifar10"
            else f"  ({self.cifar100_etiket} etiket)"
        )
        print("\n" + "=" * 60)
        print(f"  {self.dataset.upper()}{etiket_aciklama}  —  Veri Seti Yüklendi")
        print("=" * 60)
        print(f"  Eğitim : {X_egit.shape}  dtype={X_egit.dtype}")
        print(f"  Test   : {X_test.shape}")
        print(f"  Sınıf  : {self.n_sinif}  adet")
        print(f"  Piksel : [{X_egit.min():.3f} – {X_egit.max():.3f}]")

        # Sınıf dağılımı (ilk 10 sınıfı göster)
        gosterilecek = min(self.n_sinif, 10)
        print(f"\n  Sınıf dağılımı (eğitim, ilk {gosterilecek} sınıf):")
        for idx in range(gosterilecek):
            adet   = int(np.sum(y_egit == idx))
            yuzde  = adet / len(y_egit) * 100
            cubuk  = "▓" * int(yuzde / 2)
            isim   = self.sinif_isimleri[idx] if idx < len(self.sinif_isimleri) else str(idx)
            print(f"    {idx:>3}  {isim:<22}  {adet:6d}  {cubuk}")
        if self.n_sinif > 10:
            print(f"    ... ve {self.n_sinif - 10} sınıf daha")
        print("=" * 60 + "\n")



    def sinif_ismi(self, sinif_id):
        """Sayısal etiketi insan-okunabilir isme çevirir."""
        idx = int(sinif_id)
        if 0 <= idx < len(self.sinif_isimleri):
            return self.sinif_isimleri[idx]
        return f"sinif_{idx}"

    def __repr__(self):
        return (
            f"CIFARLoader(dataset='{self.dataset}', "
            f"n_sinif={self.n_sinif}, "
            f"yol='{self.veri_yolu}')"
        )
