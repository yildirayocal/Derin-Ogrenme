import numpy as np
from collections import Counter


class KNNSiniflandirici:
    """
    k-En Yakın Komşu Sınıflandırıcı

    k-NN "tembel öğrenme" algoritmasıdır:
      fit()     →  Eğitim verisini bellekte sakla  (O(1) zaman)
      predict() →  Tüm hesaplama burada           (O(N·D) zaman)
    """

    METRIKLER = ("l2", "l1", "cos")

    # -----------------------------------------------------------
    def __init__(self, k=1, metrik="l2"):
        """
        Parametreler
        ------------
        k      : int  komşu sayısı  (k ≥ 1)
        metrik : str  'l2', 'l1' veya 'cos'
        """
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"k pozitif tam sayı olmalı. Verilen: {k!r}")
        if metrik not in self.METRIKLER:
            raise ValueError(
                f"Geçersiz metrik: '{metrik}'. Seçenekler: {self.METRIKLER}"
            )
        self.k       = k
        self.metrik  = metrik
        self._X_egit = None
        self._y_egit = None

    # ===========================================================
    #  FIT
    # ===========================================================
    def fit(self, X_egitim, y_egitim):
        """
        Eğitim verisini bellekte saklar.
        CIFAR-10 veya CIFAR-100 fark etmez, her ikisi de çalışır.
        """
        self._X_egit = np.asarray(X_egitim, dtype=np.float64)
        self._y_egit = np.asarray(y_egitim, dtype=np.int64)

        if self.k > len(self._X_egit):
            raise ValueError(
                f"k={self.k} eğitim örnek sayısından "
                f"({len(self._X_egit)}) büyük olamaz."
            )
        return self

    # ===========================================================
    #  MESAFE HESAPLAMA
    # ===========================================================
    def _mesafe_hesapla(self, X_test):
        """
        Dönüş: numpy dizisi  shape=(n_test, n_egitim)
               dist[i,j] = test[i] ile egitim[j] arasındaki mesafe
        """
        X_test = np.asarray(X_test, dtype=np.float64)
        if   self.metrik == "l2":  return self._l2(X_test)
        elif self.metrik == "l1":  return self._l1(X_test)
        else:                      return self._cos(X_test)

    # -----------------------------------------------------------
    def _l2(self, X_test):
        """
        Öklid mesafesi — vektörize matris çarpımı sürümü.
        ||a-b||² = ||a||² + ||b||² - 2·(a·b)
        Naif döngüye kıyasla ~10-20× hız kazanımı sağlar.
        """
        k_test  = np.sum(X_test       ** 2, axis=1, keepdims=True)  # (n_test,1)
        k_egit  = np.sum(self._X_egit ** 2, axis=1)                 # (n_egit,)
        carpim  = X_test @ self._X_egit.T                           # (n_test,n_egit)
        d2      = k_test + k_egit - 2.0 * carpim
        return np.sqrt(np.clip(d2, 0.0, None))

    # -----------------------------------------------------------
    def _l1(self, X_test):
        """
        Manhattan mesafesi — bellek tasarruflu sürüm.
        BATCH=5 ve float32 ile bellek kullanimi ~300 MB'a iner.
        (200 batch float64 = 22.9 GB → bellek tasması!)
        """
        BATCH = 5
        X_t   = X_test.astype(np.float32)
        X_e   = self._X_egit.astype(np.float32)
        n_t   = len(X_t)
        n_e   = len(X_e)
        dist  = np.zeros((n_t, n_e), dtype=np.float32)
        for bas in range(0, n_t, BATCH):
            son  = min(bas + BATCH, n_t)
            fark = (X_t[bas:son, np.newaxis, :]
                    - X_e[np.newaxis, :, :])
            dist[bas:son] = np.sum(np.abs(fark), axis=2)
        return dist.astype(np.float64)

    # -----------------------------------------------------------
    def _cos(self, X_test):
        """
        Kosinüs uzaklığı = 1 - kosinüs_benzerliği
        0 → özdeş yön   |   2 → zıt yön
        """
        eps   = 1e-10
        n_t   = X_test        / (np.linalg.norm(X_test,        axis=1, keepdims=True) + eps)
        n_e   = self._X_egit  / (np.linalg.norm(self._X_egit,  axis=1, keepdims=True) + eps)
        return 1.0 - (n_t @ n_e.T)

    # ===========================================================
    #  PREDICT
    # ===========================================================
    def predict(self, X_test):
        """
        Her test örneği için sınıf etiketi tahmin eder.

        Adımlar:
          1. Mesafe matrisi hesapla          (n_test × n_egit)
          2. Her satırda k en yakın indeksi al
          3. Bu komşuların etiketlerini al
          4. Çoğunluk oyu → tahmin
        """
        self._kontrol()
        X_test  = np.asarray(X_test, dtype=np.float64)
        dist    = self._mesafe_hesapla(X_test)                  # (n_test, n_egit)
        k_idx   = np.argsort(dist, axis=1)[:, :self.k]         # (n_test, k)
        k_label = self._y_egit[k_idx]                          # (n_test, k)
        return np.array(
            [Counter(r.tolist()).most_common(1)[0][0] for r in k_label],
            dtype=np.int64
        )

    # ===========================================================
    #  PREDICT_PROBA
    # ===========================================================
    def predict_proba(self, X_test, n_sinif=10):
        """
        Her sınıf için oy oranını döndürür.
        CIFAR-100 için n_sinif=100 (ince) veya 20 (kaba) geçirin.

        Dönüş: numpy dizisi  shape=(n_test, n_sinif)
        """
        self._kontrol()
        dist    = self._mesafe_hesapla(X_test)
        k_idx   = np.argsort(dist, axis=1)[:, :self.k]
        k_label = self._y_egit[k_idx]
        prob    = np.zeros((len(X_test), n_sinif), dtype=np.float64)
        for i, r in enumerate(k_label):
            for cls, cnt in Counter(r.tolist()).items():
                if cls < n_sinif:
                    prob[i, cls] = cnt / self.k
        return prob

    # ===========================================================
    #  SCORE
    # ===========================================================
    def score(self, X_test, y_test):
        """Test seti doğruluk oranı  →  0.0 ile 1.0 arasında float."""
        preds = self.predict(X_test)
        return float(np.mean(preds == np.asarray(y_test, dtype=np.int64)))

    # ===========================================================
    #  Yardımcı
    # ===========================================================
    def _kontrol(self):
        if self._X_egit is None:
            raise RuntimeError("Önce clf.fit(X, y) çağrılmalı!")

    def __repr__(self):
        n = len(self._X_egit) if self._X_egit is not None else "—"
        return (
            f"KNNSiniflandirici("
            f"k={self.k}, metrik='{self.metrik}', n_egitim={n})"
        )
