# Bu kod, oluşturduğun veri kümesindeki yüzleri okur
# ve bir yüz tanıma modeli (örneğin, LBPHFaceRecognizer) eğitir.
# Eğitilen model, bir .yml dosyası olarak kaydedilir.

import cv2
import os
import numpy as np
from PIL import Image

# Veri kümesi klasörünün yolu
path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def get_images_and_labels(path):
    image_paths = [os.path.join(dirpath, f) for dirpath, dirnames, filenames in os.walk(path) for f in filenames]
    face_samples = []
    ids = []
    
    id_map = {}
    current_id = 0
    
    for image_path in image_paths:
        img_pil = Image.open(image_path).convert('L')
        img_np = np.array(img_pil, 'uint8')
        
        name = os.path.basename(os.path.dirname(image_path))
        if name not in id_map:
            id_map[name] = current_id
            current_id += 1
            
        face_id = id_map[name]
        faces = face_detector.detectMultiScale(img_np)
        
        for (x, y, w, h) in faces:
            face_samples.append(img_np[y:y+h, x:x+w])
            ids.append(face_id)
            
    return face_samples, ids, id_map

print("\n[INFO] Yüzler eğitiliyor. Lütfen bekleyin...")
faces, ids, id_map = get_images_and_labels(path)
recognizer.train(faces, np.array(ids))

# Modeli kaydet
recognizer.write('trainer/trainer.yml')

# ID-isim eşleşmesini kaydet
with open('trainer/labels.txt', 'w') as f:
    for name, id_val in id_map.items():
        f.write(f"{id_val}:{name}\n")

print(f"\n[INFO] {len(np.unique(ids))} yüz eğitildi. Çıkış yapılıyor.")