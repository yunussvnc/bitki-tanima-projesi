import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json

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

def load_and_preprocess_image(img_path, target_size=(300, 300)):
    """Görüntüyü yükle ve ön işleme yap"""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

def predict_plant(model_path, img_path):
    """Bitki türünü tahmin et"""
    try:
        # Model ve sınıf isimlerini yükle
        model = load_model(model_path)
        with open('models/class_names.txt', 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        
        # Görüntüyü yükle ve ön işleme yap
        img_array = load_and_preprocess_image(img_path)
        
        # Tahmin yap
        predictions = model.predict(img_array)
        
        # En yüksek 3 tahmini al
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [(class_names[idx], float(predictions[0][idx])) for idx in top_3_idx]
        
        # Sonuçları yazdır
        print("\nTahmin Sonuçları:")
        print("-" * 50)
        for plant_name, confidence in top_3_predictions:
            print(f"{plant_name}: %{confidence*100:.2f}")
        print("-" * 50)
        
        return top_3_predictions[0]  # En yüksek tahmini döndür
        
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        return None

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
    model_path = "models/bitki_model.h5"
    test_dir = "test"
    
    # Test klasöründeki tüm görüntüleri işle
    for filename in os.listdir(test_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(test_dir, filename)
            print(f"\nGörüntü: {filename}")
            predict_plant(model_path, img_path) 