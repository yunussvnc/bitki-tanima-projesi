from flask import Flask, render_template, jsonify, request
import json
import os

app = Flask(__name__)

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/bitki/<bitki_adi>')
def get_bitki_info(bitki_adi):
    response = {}
    
    # Bitki bilgilerini topla
    if 'bitki_bilgileri' in data:
        for bitki, bilgi in data['bitki_bilgileri'].items():
            if bitki.lower() == bitki_adi.lower():
                response['temel_bilgiler'] = bilgi
                break
    
    # İlişkili bitkileri bul
    if 'bitki_iliskileri' in data:
        for kategori, bitkiler in data['bitki_iliskileri']['plant_relationships']['companion_plants'].items():
            if bitki_adi.lower() in bitkiler:
                response['iliskili_bitkiler'] = bitkiler
    
    # Bakım takvimini ekle
    if 'bitki_bakim_takvimi' in data:
        response['bakim_takvimi'] = data['bitki_bakim_takvimi']
    
    # Zorluk seviyesini bul
    if 'bitki_zorluk_seviyeleri' in data:
        for seviye, bilgi in data['bitki_zorluk_seviyeleri']['zorluk_seviyeleri'].items():
            if bitki_adi.lower() in bilgi['bitkiler']:
                response['zorluk_seviyesi'] = {
                    'seviye': seviye,
                    'bilgi': bilgi
                }
    
    return jsonify(response)

@app.route('/api/arama')
def search():
    query = request.args.get('q', '').lower()
    results = {}
    
    # Tüm dosyalarda arama yap
    for key, content in data.items():
        if isinstance(content, dict):
            for bitki, bilgi in content.items():
                if query in str(bilgi).lower():
                    if key not in results:
                        results[key] = []
                    results[key].append({
                        'bitki': bitki,
                        'bilgi': bilgi
                    })
    
    return jsonify(results)

def load_plant_data():
    with open('bitki_bilgileri.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['plants']

if __name__ == '__main__':
    app.run(debug=True) 