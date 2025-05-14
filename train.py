import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
import numpy as np

# GPU Ayarları
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # GPU bellek büyümesini dinamik olarak ayarla
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU bulundu: {len(physical_devices)} adet")
        print("GPU bellek büyümesi aktif edildi")
    except RuntimeError as e:
        print(f"GPU ayarları yapılandırılamadı: {e}")
else:
    print("GPU bulunamadı, CPU kullanılacak")

# Mixed Precision Training
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print("Mixed Precision Training aktif edildi")

# === 1. VERİ YOLLARI ===
base_dir = r"C:\Users\eruru\OneDrive\Belgeler\bitki_guncel_proje\data"
train = os.path.join(base_dir, "train")
valid = os.path.join(base_dir, "valid")
test = os.path.join(base_dir, "test")

# === 2. GELİŞMİŞ VERİ ARTIRMA ===
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2],
    validation_split=0.2,
    rescale=1./255
)

valid_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# === 3. VERİ SETLERİNİ YÜKLE ===
train_generator = train_datagen.flow_from_directory(
    train,
    target_size=(224, 224),  # ResNet50 için standart boyut
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = valid_datagen.flow_from_directory(
    train,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# === 4. RESNET50 TABANLI MODEL MİMARİSİ ===
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Transfer learning için base model'i dondur
base_model.trainable = False

# Model mimarisi
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# === 5. MODEL DERLEME ===
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === 6. CALLBACKS ===
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    ),
    ModelCheckpoint(
        'models/bitki_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
]

# === 7. İLK EĞİTİM AŞAMASI ===
print("\n🚀 İlk eğitim aşaması başlıyor...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=callbacks
)

# === 8. FINE-TUNING AŞAMASI ===
# Son birkaç katmanı çöz
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Modeli düşük learning rate ile tekrar derle
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n🚀 Fine-tuning aşaması başlıyor...")
history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=callbacks
)

# === 9. MODEL KAYDETME ===
model.save('models/bitki_model.h5')

# Sınıf isimlerini kaydet
class_names = list(train_generator.class_indices.keys())
with open('models/class_names.txt', 'w') as f:
    f.write('\n'.join(class_names))

print("\n✅ Model eğitimi tamamlandı ve kaydedildi!")
