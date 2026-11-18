# -*- coding: utf-8 -*-
"""
Derin Öğrenme Ödevi
Derin Öğrenme ve Evrişimsel Sinir Ağlarının (CNN) Temel Yapısının İncelenmesi

Programlama dili : Python
Kütüphane        : TensorFlow / Keras
Veri seti        : Fashion-MNIST

Çalıştırma: python derin_ogrenme_odevi.py
Grafikler "ciktilar" klasörüne PNG olarak kaydedilir.
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

CIKTI = "ciktilar"
os.makedirs(CIKTI, exist_ok=True)


def baslik(metin):
    print("\n" + "=" * 70)
    print(metin)
    print("=" * 70)


# ---------------------------------------------------------------------
# 3. BÖLÜM: Konvolüsyon işleminin kodlanması
# ---------------------------------------------------------------------
baslik("3. BÖLÜM: Konvolüsyon katmanı")

# Rastgelelik kontrolü
tf.keras.utils.set_random_seed(42)

# 3x3 boyutunda 16 filtreli konvolüsyon katmanı
conv = tf.keras.layers.Conv2D(
    filters=16,
    kernel_size=(3, 3),
    activation='relu',
    kernel_initializer='he_normal'
)

# Örnek giriş görüntüsü
dummy_input = tf.random.normal((1, 128, 128, 1))

# Konvolüsyon işlemi
output = conv(dummy_input)

# Ağırlıkları al
weights, biases = conv.get_weights()

print("Giriş boyutu:", dummy_input.shape)
print("Çıkış boyutu:", output.shape)
print("Filtre ağırlıklarının boyutu:", weights.shape)
print("Bias boyutu:", biases.shape)

# İlk filtrenin ağırlıkları
print("İlk filtre:")
print(weights[:, :, 0, 0])

# Çalışma 7: Bütün filtrelerin başlangıç ağırlıklarını görselleştirme
fig, axes = plt.subplots(2, 8, figsize=(14, 4))
vmax = np.abs(weights).max()
for i, ax in enumerate(axes.flat):
    filtre = weights[:, :, 0, i]
    im = ax.imshow(filtre, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    for (r, c), val in np.ndenumerate(filtre):
        ax.text(c, r, f"{val:.2f}", ha='center', va='center', fontsize=6)
    ax.set_title(f"Filtre {i + 1}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("16 adet 3x3 filtrenin başlangıç ağırlıkları (he_normal)")
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
plt.savefig(os.path.join(CIKTI, "filtreler.png"), dpi=150, bbox_inches='tight')
plt.show()

# Ek: İlk 4 filtrenin ürettiği özellik haritaları
fig, axes = plt.subplots(1, 5, figsize=(14, 3))
axes[0].imshow(dummy_input[0, :, :, 0], cmap='gray')
axes[0].set_title("Giriş (128x128)")
for i in range(4):
    axes[i + 1].imshow(output[0, :, :, i], cmap='viridis')
    axes[i + 1].set_title(f"Filtre {i + 1} çıkışı (126x126)")
for ax in axes:
    ax.axis('off')
plt.savefig(os.path.join(CIKTI, "ozellik_haritalari.png"), dpi=150, bbox_inches='tight')
plt.show()


# ---------------------------------------------------------------------
# 4. BÖLÜM: Pooling işleminin incelenmesi
# ---------------------------------------------------------------------
baslik("4. BÖLÜM: Max Pooling")

# Örnek giriş matrisi
input_data = tf.constant([
    [[1., 3., 2., 4.],
     [5., 6., 1., 2.],
     [7., 2., 8., 3.],
     [4., 1., 5., 9.]]
])

# Boyutları düzenle: Batch, Height, Width, Channels
input_data = tf.reshape(input_data, (1, 4, 4, 1))

pool = tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2),
    strides=2
)

output = pool(input_data)

print("Giriş:")
print(input_data.numpy()[0, :, :, 0])
print("Pooling sonucu:")
print(output.numpy()[0, :, :, 0])
print("Giriş boyutu:", input_data.shape, "-> Çıkış boyutu:", output.shape)

# Elle hesaplanan sonuç ile karşılaştırma
elle = np.array([[6., 4.], [7., 9.]])
print("Elle hesaplanan:")
print(elle)
print("Program sonucu elle hesaplananla aynı mı?",
      np.array_equal(elle, output.numpy()[0, :, :, 0]))


# ---------------------------------------------------------------------
# 5. BÖLÜM: ReLU ve Softmax
# ---------------------------------------------------------------------
baslik("5. BÖLÜM-A: ReLU")

x = tf.constant([-5., -2., -1., 0., 2., 4., 6.])

relu = tf.keras.layers.ReLU()

y = relu(x)

print("Giriş:", x.numpy())
print("ReLU çıkışı:", y.numpy())

baslik("5. BÖLÜM-B: Softmax")

logits = tf.constant([[2.0, 1.0, 0.1]])

softmax = tf.keras.layers.Softmax()

probabilities = softmax(logits)

print("Softmax sonuçları:")
print(probabilities.numpy())
print("Olasılıkların toplamı:")
print(tf.reduce_sum(probabilities).numpy())
print("En yüksek olasılıklı sınıf (argmax):", int(tf.argmax(probabilities, axis=1)[0]))


# ---------------------------------------------------------------------
# 6. BÖLÜM: Basit bir CNN modeli
# ---------------------------------------------------------------------
baslik("6. BÖLÜM-A: Veri setinin yüklenmesi")

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

# Normalizasyon
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Kanal boyutu ekleme
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

print("Eğitim verisi:", x_train.shape)
print("Test verisi:", x_test.shape)

SINIFLAR = ["T-shirt/Üst", "Pantolon", "Kazak", "Elbise", "Ceket",
            "Sandalet", "Gömlek", "Spor ayakkabı", "Çanta", "Bot"]

# Örnek görüntüler
fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
for i, ax in enumerate(axes.flat):
    idx = np.where(y_train == i)[0][0]
    ax.imshow(x_train[idx, :, :, 0], cmap='gray')
    ax.set_title(f"{i}: {SINIFLAR[i]}", fontsize=9)
    ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(CIKTI, "ornek_goruntuler.png"), dpi=150, bbox_inches='tight')
plt.show()


def model_olustur(ilk_filtre=8):
    """Ödevde verilen mimari. ilk_filtre: ilk Conv2D katmanındaki filtre sayısı."""
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        # Birinci konvolüsyon
        tf.keras.layers.Conv2D(ilk_filtre, (3, 3), activation='relu'),
        # Birinci pooling
        tf.keras.layers.MaxPooling2D((2, 2)),
        # İkinci konvolüsyon
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        # İkinci pooling
        tf.keras.layers.MaxPooling2D((2, 2)),
        # Çok boyutlu özellikleri tek boyuta dönüştür
        tf.keras.layers.Flatten(),
        # Tam bağlantılı katman
        tf.keras.layers.Dense(64, activation='relu'),
        # 10 sınıf için çıktı
        tf.keras.layers.Dense(10, activation='softmax')
    ])


def egit_ve_degerlendir(ilk_filtre, ozet=False):
    tf.keras.utils.set_random_seed(42)  # iki deney aynı başlangıç koşullarında
    model = model_olustur(ilk_filtre)
    if ozet:
        model.summary()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    t0 = time.time()
    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.1
    )
    sure = time.time() - t0
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    return model, history, test_loss, test_accuracy, sure


baslik("6. BÖLÜM-B/C: Model mimarisi, derleme ve eğitim (8 filtre)")
model, history, test_loss, test_accuracy, sure8 = egit_ve_degerlendir(8, ozet=True)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# D. Eğitim sonuçlarının görselleştirilmesi
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(CIKTI, "egitim_grafikleri.png"), dpi=150, bbox_inches='tight')
plt.show()

# Ek: Test görüntülerinden örnek tahminler
tahmin = model.predict(x_test[:10], verbose=0)
fig, axes = plt.subplots(2, 5, figsize=(11, 6.5))
for i, ax in enumerate(axes.flat):
    p = int(np.argmax(tahmin[i]))
    dogru = p == y_test[i]
    ax.imshow(x_test[i, :, :, 0], cmap='gray')
    ax.set_title(f"Tahmin: {SINIFLAR[p]}\n(%{tahmin[i][p] * 100:.1f})\nGerçek: {SINIFLAR[y_test[i]]}",
                 fontsize=8, color='green' if dogru else 'red')
    ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(CIKTI, "ornek_tahminler.png"), dpi=150, bbox_inches='tight')
plt.show()


# ---------------------------------------------------------------------
# 7. BÖLÜM: Model mimarisinin anlaşılması (katman boyutları)
# ---------------------------------------------------------------------
baslik("7. BÖLÜM: Katman çıkış boyutları")
for katman in model.layers:
    print(f"{katman.name:20s} {str(katman.output.shape):22s} parametre: {katman.count_params()}")
print("Toplam parametre:", model.count_params())


# ---------------------------------------------------------------------
# 8. BÖLÜM: Deney (8 filtre vs 16 filtre)
# ---------------------------------------------------------------------
baslik("8. BÖLÜM: Deney 2 (ilk katmanda 16 filtre)")
model16, history16, test_loss16, test_accuracy16, sure16 = egit_ve_degerlendir(16)

baslik("DENEY SONUÇLARI")
satirlar = [
    ("Filtre sayısı", "8", "16"),
    ("Toplam parametre", f"{model.count_params():,}", f"{model16.count_params():,}"),
    ("Eğitim doğruluğu (son epoch)", f"{history.history['accuracy'][-1]:.4f}",
     f"{history16.history['accuracy'][-1]:.4f}"),
    ("Doğrulama doğruluğu (son epoch)", f"{history.history['val_accuracy'][-1]:.4f}",
     f"{history16.history['val_accuracy'][-1]:.4f}"),
    ("Test doğruluğu", f"{test_accuracy:.4f}", f"{test_accuracy16:.4f}"),
    ("Test kaybı", f"{test_loss:.4f}", f"{test_loss16:.4f}"),
    ("Eğitim süresi (sn)", f"{sure8:.1f}", f"{sure16:.1f}"),
]
print(f"{'Ölçüt':34s}{'Deney 1':>12s}{'Deney 2':>12s}")
for ad, a, b in satirlar:
    print(f"{ad:34s}{a:>12s}{b:>12s}")

# Deney karşılaştırma grafiği
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], 'o-', label='8 filtre - Training')
plt.plot(history.history['val_accuracy'], 'o--', label='8 filtre - Validation')
plt.plot(history16.history['accuracy'], 's-', label='16 filtre - Training')
plt.plot(history16.history['val_accuracy'], 's--', label='16 filtre - Validation')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.legend(fontsize=8)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], 'o-', label='8 filtre - Training')
plt.plot(history.history['val_loss'], 'o--', label='8 filtre - Validation')
plt.plot(history16.history['loss'], 's-', label='16 filtre - Training')
plt.plot(history16.history['val_loss'], 's--', label='16 filtre - Validation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(CIKTI, "deney_karsilastirma.png"), dpi=150, bbox_inches='tight')
plt.show()
