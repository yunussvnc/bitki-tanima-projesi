import os
import shutil
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dizin yolları
TRAIN_DIR = 'data/train'
VALID_DIR = 'data/valid'
TEST_DIR = 'data/test'
AUGMENTED_DIR = 'data/augmented'

# Hedef görüntü boyutu
IMG_SIZE = 224

def create_directories():
    """Gerekli dizinleri oluştur"""
    for dir_path in [VALID_DIR, TEST_DIR, AUGMENTED_DIR]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

def resize_and_save_image(img_path, target_size=(IMG_SIZE, IMG_SIZE)):
    """Görüntüyü yeniden boyutlandır ve kaydet"""
    try:
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        print(f"Hata: {img_path} - {str(e)}")
        return None

def prepare_dataset():
    """Veri setini hazırla"""
    create_directories()
    
    # Her sınıf için görüntüleri topla
    for class_name in os.listdir(TRAIN_DIR):
        class_dir = os.path.join(TRAIN_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        # Görüntü yollarını topla
        image_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Eğitim/doğrulama/test olarak böl
        train_paths, temp_paths = train_test_split(image_paths, test_size=0.3, random_state=42)
        valid_paths, test_paths = train_test_split(temp_paths, test_size=0.5, random_state=42)
        
        # Görüntüleri işle ve kaydet
        for paths, target_dir in [(train_paths, TRAIN_DIR), 
                                (valid_paths, VALID_DIR), 
                                (test_paths, TEST_DIR)]:
            target_class_dir = os.path.join(target_dir, class_name)
            if not os.path.exists(target_class_dir):
                os.makedirs(target_class_dir)
                
            for img_path in paths:
                img = resize_and_save_image(img_path)
                if img is not None:
                    img.save(os.path.join(target_class_dir, os.path.basename(img_path)))

def augment_data():
    """Veri artırma işlemlerini uygula"""
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.6, 1.4]
    )
    
    # Her sınıf için veri artırma
    for class_name in os.listdir(TRAIN_DIR):
        class_dir = os.path.join(TRAIN_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        # Hedef dizini oluştur
        target_dir = os.path.join(AUGMENTED_DIR, class_name)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        # Görüntüleri yükle ve artır
        for img_name in os.listdir(class_dir):
            if not img_name.endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(class_dir, img_name)
            img = Image.open(img_path)
            img = img.convert('RGB')
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
            img_array = np.array(img)
            img_array = img_array.reshape((1,) + img_array.shape)
            
            # Her görüntü için 5 artırılmış versiyon oluştur
            i = 0
            for batch in datagen.flow(img_array, batch_size=1,
                                    save_to_dir=target_dir,
                                    save_prefix=f'aug_{class_name}',
                                    save_format='jpg'):
                i += 1
                if i >= 5:
                    break

if __name__ == '__main__':
    print("Veri seti hazırlanıyor...")
    prepare_dataset()
    print("Veri artırma işlemi yapılıyor...")
    augment_data()
    print("İşlem tamamlandı!") 