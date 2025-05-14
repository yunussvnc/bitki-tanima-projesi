import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json
from PIL import Image
import matplotlib.pyplot as plt

# JSON dosyalarını yükle
def load_json_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

# Tüm JSON dosyalarını yükle
json_files = {
    'bitki_bilgileri': 'bitki_bilgileri.json',
    'bitki_iliskileri': 'bitki_iliskileri.json',
    'bitki_bakim_takvimi': 'bitki_bakim_takvimi.json',
    'bitki_zorluk_seviyeleri': 'bitki_zorluk_seviyeleri.json',
    'bitki_hastaliklari': 'bitki_hastaliklari.json',
    'bitki_yetistirme_teknikleri': 'bitki_yetistirme_teknikleri.json',
    'bitki_besin_degerleri': 'bitki_besin_degerleri.json',
    'bitki_iklim_bolgeleri': 'bitki_iklim_bolgeleri.json',
    'bitki_uretim_teknikleri': 'bitki_uretim_teknikleri.json',
    'bitki_hasat_depolama': 'bitki_hasat_depolama.json',
    'bitki_isleme_teknikleri': 'bitki_isleme_teknikleri.json'
}

data = {}
for key, filename in json_files.items():
    if os.path.exists(filename):
        data[key] = load_json_file(filename)

# Veri dizininden sınıf isimlerini al
data_dir = r"C:\Users\eruru\OneDrive\Belgeler\bitki_guncel_proje\data\train"
class_names = sorted(os.listdir(data_dir))

# Modeli yükle
model_path = r"C:\Users\eruru\OneDrive\Belgeler\bitki_guncel_proje\models\bitki_model.h5"
model = load_model(model_path)

def get_plant_info(plant_name):
    info = {}
    
    # Temel bilgiler
    if 'bitki_bilgileri' in data and plant_name in data['bitki_bilgileri']:
        info['temel_bilgiler'] = data['bitki_bilgileri'][plant_name]
    
    # İlişkili bitkiler
    if 'bitki_iliskileri' in data:
        for kategori, bitkiler in data['bitki_iliskileri']['plant_relationships']['companion_plants'].items():
            if plant_name in bitkiler:
                info['iliskili_bitkiler'] = bitkiler
    
    # Bakım takvimi
    if 'bitki_bakim_takvimi' in data:
        info['bakim_takvimi'] = data['bitki_bakim_takvimi']
    
    # Zorluk seviyesi
    if 'bitki_zorluk_seviyeleri' in data:
        for seviye, bilgi in data['bitki_zorluk_seviyeleri']['zorluk_seviyeleri'].items():
            if plant_name in bilgi['bitkiler']:
                info['zorluk_seviyesi'] = {
                    'seviye': seviye,
                    'bilgi': bilgi
                }
    
    # Hastalıklar
    if 'bitki_hastaliklari' in data:
        info['hastaliklar'] = data['bitki_hastaliklari']
    
    # Yetiştirme teknikleri
    if 'bitki_yetistirme_teknikleri' in data:
        info['yetistirme_teknikleri'] = data['bitki_yetistirme_teknikleri']
    
    # Besin değerleri
    if 'bitki_besin_degerleri' in data and plant_name in data['bitki_besin_degerleri']:
        info['besin_degerleri'] = data['bitki_besin_degerleri'][plant_name]
    
    return info

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """Görüntüyü yükle ve ön işle"""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_plant(img_path):
    """Bitki türünü tahmin et ve detaylı bilgileri göster"""
    # Model ve sınıf isimlerini yükle
    model = load_model('models/bitki_model.h5')
    with open('models/class_names.txt', 'r') as f:
        class_names = f.read().splitlines()
    
    # Görüntüyü yükle ve ön işle
    img_array = load_and_preprocess_image(img_path)
    
    # Tahmin yap
    predictions = model.predict(img_array)
    top_idx = np.argmax(predictions[0])
    confidence = predictions[0][top_idx] * 100
    plant_name = class_names[top_idx]
    
    # Sonuçları göster
    print("\n🔍 Tahmin Sonucu:")
    print("-" * 30)
    
    # Görüntüyü göster
    img = Image.open(img_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis('off')
    
    # Güven skoruna göre renk belirle
    if confidence >= 90:
        color = 'green'
    elif confidence >= 70:
        color = 'orange'
    else:
        color = 'red'
    
    plt.title(f"Tahmin: {plant_name}\nGüven: {confidence:.1f}%", 
              color=color, fontsize=12)
    plt.show()
    
    print(f"\n✅ En Yüksek Tahmin:")
    print(f"Bitki Türü: {plant_name}")
    print(f"Güven Oranı: {confidence:.1f}%")
    
    # Bitki bilgilerini göster
    plant_info = get_plant_info(plant_name)
    
    print("\n📚 Bitki Bilgileri:")
    print("-" * 30)
    
    if 'temel_bilgiler' in plant_info:
        print("\n🌱 Temel Bilgiler:")
        for key, value in plant_info['temel_bilgiler'].items():
            print(f"{key}: {value}")
    
    if 'zorluk_seviyesi' in plant_info:
        print("\n📊 Zorluk Seviyesi:")
        print(f"Seviye: {plant_info['zorluk_seviyesi']['seviye']}")
        print(f"Bilgi: {plant_info['zorluk_seviyesi']['bilgi']['aciklama']}")
    
    if 'bakim_takvimi' in plant_info:
        print("\n📅 Bakım Takvimi:")
        for ay, bilgi in plant_info['bakim_takvimi'].items():
            print(f"\n{ay}:")
            for key, value in bilgi.items():
                print(f"- {key}: {value}")
    
    if 'hastaliklar' in plant_info:
        print("\n⚠️ Yaygın Hastalıklar:")
        for hastalik, bilgi in plant_info['hastaliklar'].items():
            print(f"\n{hastalik}:")
            print(f"- Belirtiler: {bilgi['belirtiler']}")
            print(f"- Tedavi: {bilgi['tedavi']}")
    
    if 'yetistirme_teknikleri' in plant_info:
        print("\n🌿 Yetiştirme Teknikleri:")
        for teknik, bilgi in plant_info['yetistirme_teknikleri'].items():
            print(f"\n{teknik}:")
            print(f"- Açıklama: {bilgi['aciklama']}")
            print(f"- Öneriler: {bilgi['oneriler']}")
    
    return plant_name, confidence, plant_info

def predict_image(image_path):
    # Görüntüyü yükle ve ön işle
    img = Image.open(image_path)
    img = img.resize((300, 300))  # Modelin beklediği boyut
    img_array = np.array(img) / 255.0  # Normalize et
    img_array = np.expand_dims(img_array, axis=0)  # Batch boyutu ekle
    
    # Tahmin yap
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    # Bitki adını al
    plant_name = class_names[predicted_class]
    
    # Detaylı bilgileri al
    plant_info = get_plant_info(plant_name)
    
    return {
        'bitki_turu': plant_name,
        'guven_orani': float(confidence),
        'detayli_bilgiler': plant_info
    }

# Örnek kullanım
if __name__ == "__main__":
    # Test görüntüsü yolu
    test_image = "test_images/banana.jpg"  # Test edilecek görüntüyü buraya koyun
    
    if os.path.exists(test_image):
        print(f"\n📸 Görüntü analiz ediliyor: {test_image}")
        result, confidence, info = predict_plant(test_image)
    else:
        print(f"❌ Hata: {test_image} bulunamadı!") 