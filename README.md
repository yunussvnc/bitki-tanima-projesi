# Bitki Tanıma Projesi

Bu proje, derin öğrenme kullanarak bitkileri tanımlayan bir yapay zeka modelidir. ResNet50 transfer learning kullanılarak geliştirilmiştir.

## Özellikler

- ResNet50 tabanlı transfer learning
- Gelişmiş veri artırma teknikleri
- İki aşamalı eğitim stratejisi
- %90+ doğruluk oranı
- GPU desteği
- Mixed precision training

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Veri setini hazırlayın:
- `data/train` klasörüne eğitim görüntülerini yerleştirin
- Her bitki türü için ayrı klasör oluşturun

3. Modeli eğitin:
```bash
python train.py
```

## Kullanım

Eğitilmiş modeli kullanmak için:
```python
from predict import predict_plant

# Görüntüyü tahmin et
result = predict_plant("test_image.jpg")
print(result)
```

## Model Mimarisi

- ResNet50 tabanlı transfer learning
- Global Average Pooling
- Dense katmanlar (1024 ve 512 nöronlu)
- Dropout (0.5)
- Batch Normalization

## Eğitim Stratejisi

1. İlk Aşama:
- Base model dondurulmuş
- 50 epoch
- Learning rate: 0.001

2. Fine-tuning:
- Son 30 katman çözülmüş
- 50 epoch
- Learning rate: 0.0001

## Gereksinimler

- Python 3.8+
- TensorFlow 2.x
- CUDA (GPU için)
- Diğer gereksinimler için requirements.txt dosyasına bakın

## Güncellemeler

- 13.05.2025: İlk sürüm yayınlandı
- 13.05.2025: README dosyası güncellendi

## Lisans

MIT License 