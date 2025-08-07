# Bu kod, kameradan canlı görüntü alarak yüzleri algılar,
# keser ve veri kümesi (dataset) klasörüne kaydeder. 
# Böylece veri toplama işlemini otomatikleştirebilirsin.

import cv2
import os

# Yüz algılayıcı modelini yükle
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Veri kümesi klasörünü oluştur
dataset_path = "dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Terminalden isim girişi
face_id = input('\n[INFO] Lütfen tanınacak kişinin ismini girin: ')
print(f"\n[INFO] Yüz örneği yakalanıyor. Kameraya bakın ve bekleyin...")

# Yeni bir klasör oluştur
person_path = os.path.join(dataset_path, face_id)
if not os.path.exists(person_path):
    os.makedirs(person_path)

# Kamera bağlantısını aç
cam = cv2.VideoCapture(0)
sayac = 0

while True:
    ret, img = cam.read()
    if not ret:
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        sayac += 1
        
        # Yüzü kaydet
        cv2.imwrite(f"{person_path}/{face_id}_{sayac}.jpg", gray[y:y+h, x:x+w])
        
        # Ekrandaki yüzün üzerine ismini yaz
        cv2.putText(img, face_id, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff
    if k == 27: # ESC tuşuna basıldığında
        break
    elif sayac >= 30: # 30 yüz fotoğrafı çekildiğinde
         break

print("\n[INFO] Çıkış yapılıyor...")
cam.release()
cv2.destroyAllWindows()