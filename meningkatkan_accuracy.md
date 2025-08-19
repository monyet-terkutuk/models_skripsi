Nah bagus 👍, kamu sudah punya baseline.
Kalau mau **meningkatkan model** biar val\_accuracy lebih tinggi dan overfitting lebih kecil, ada beberapa langkah praktis yang bisa langsung kamu coba:

---

### 1. **Data Augmentation**

Supaya model lebih general dan tidak hafalan ke data train:

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

Validation datagen tetap pakai `rescale=1./255`.

---

### 2. **Fine-tuning MobileNet**

Sekarang `base_model.trainable = False`, jadi hanya classifier (Dense layer) yang belajar.
Setelah training awal, kamu bisa buka beberapa layer terakhir MobileNet:

```python
base_model.trainable = True
for layer in base_model.layers[:-20]:  # freeze semua kecuali 20 layer terakhir
    layer.trainable = False
```

Lalu compile ulang:

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

Training ulang dengan epoch lebih panjang.

---

### 3. **Callbacks (EarlyStopping & ModelCheckpoint)**

Supaya training lebih stabil dan otomatis ambil model terbaik:

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("outputs/best_model.keras", save_best_only=True)
]

history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=callbacks
)
```

---

### 4. **Tambah Layer Regularisasi**

Untuk mengurangi overfitting, tambahkan **Dropout** sebelum output:

```python
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation='softmax')
])
```

---

### 5. **Learning Rate Tuning**

Default Adam mungkin terlalu cepat. Coba lebih kecil:

```python
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=1e-4), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
```

---

📌 Jadi alur peningkatan biasanya:

1. Latih dulu dengan **base\_model frozen** (seperti kode kamu).
2. Setelah stabil, **unfreeze layer terakhir MobileNet** dan fine-tune.
3. Gunakan **augmentation + dropout**.
4. Jalankan training **lebih lama** dengan callbacks.

---

Mau saya bikinkan **versi kode lengkap yang sudah digabung semua perbaikan ini**, biar tinggal kamu jalankan untuk dapat model yang lebih kuat?
