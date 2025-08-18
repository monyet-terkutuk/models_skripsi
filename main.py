from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

# Path dataset
train_dir = 'data/train'
val_dir = 'data/validation'

# Image preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Load MobileNet tanpa fully connected layer (include_top=False)
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False  # freeze

# Tambah layer klasifikasi
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Simpan model
if not os.path.exists('outputs'):
    os.makedirs('outputs')

# model.save('outputs/model_mobilenet.h5')
# model.save('outputs/modelv3.keras')  # Format Keras baru
# model.save('outputs/modelv2.h5')

from tensorflow.keras.models import save_model
save_model(model, 'outputs/modelV2.h5')
