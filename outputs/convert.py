import tensorflow as tf

# Load model dari file .keras
model = tf.keras.models.load_model("my_model.keras")

# Konversi ke TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Simpan sebagai file .tflite
with open("my_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model berhasil dikonversi ke my_model.tflite ✅")

