import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import random

def augment_image(image_path, output_dir, num_augmentations=5):
    # Görüntüyü yükle
    img = Image.open(image_path)
    img = np.array(img)
    
    # Veri artırma ayarları
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2],
        zoom_range=[0.8, 1.2]
    )
    
    # Görüntüyü yeniden şekillendir
    img = img.reshape((1,) + img.shape)
    
    # Artırılmış görüntüleri oluştur ve kaydet
    i = 0
    for batch in datagen.flow(img, batch_size=1, save_to_dir=output_dir,
                            save_prefix='aug', save_format='jpg'):
        i += 1
        if i >= num_augmentations:
            break

def process_directory(input_dir, output_dir):
    # Çıktı dizinini oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    # Her alt dizin için
    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
        if os.path.isdir(class_dir):
            # Sınıf için çıktı dizini oluştur
            class_output_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_output_dir, exist_ok=True)
            
            # Her görüntü için
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    # Her görüntü için 5 artırılmış versiyon oluştur
                    augment_image(img_path, class_output_dir, num_augmentations=5)

if __name__ == "__main__":
    input_dir = "data/train"
    output_dir = "data/augmented"
    
    print("Veri artırma başlıyor...")
    process_directory(input_dir, output_dir)
    print("Veri artırma tamamlandı!") 