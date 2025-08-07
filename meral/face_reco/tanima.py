#Bu kod, eğitilmiş modeli yükler ve canlı video akışında yüzleri tanımaya çalışır. Tanıdığı yüzün üzerine ismini yazdırır.

import cv2
import os

# Yüz tanıma modelini yükle
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

# Yüz algılayıcı modelini yükle
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# ID-isim eşleşmesini yükle
labels = {}
with open('trainer/labels.txt', 'r') as f:
    for line in f.readlines():
        id_val, name = line.strip().split(':')
        labels[int(id_val)] = name

# Ekrana yazı yazmak için kullanılacak font
font = cv2.FONT_HERSHEY_SIMPLEX

# Kamera bağlantısını aç
cam = cv2.VideoCapture(0)

print("\n[INFO] Yüz tanıma sistemi başlatıldı. Çıkmak için ESC tuşuna basın.")

while True:
    ret, img = cam.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        # Başlangıçta kutu rengi kırmızı (Bilinmeyen yüz)
        box_color = (0, 0, 255) # BGR formatında

        id_pred, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < 70:
            name = labels[id_pred]
            confidence_str = "  {0}%".format(round(100 - confidence))
            # Yüz tanındığında kutu rengi yeşil olsun
            box_color = (0, 255, 0) # BGR formatında
        else:
            name = "Bilinmeyen"
            confidence_str = "  {0}%".format(round(100 - confidence))
            
        # Belirlenen renkte kutuyu çiz
        cv2.rectangle(img, (x, y), (x+w, y+h), box_color, 2)

        # İsmi ve güven seviyesini ekrana yaz
        cv2.putText(img, str(name), (x+5, y-5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence_str), (x+5, y+h-5), font, 1, box_color, 1)

    cv2.imshow('Kamera', img)
    k = cv2.waitKey(10) & 0xff
    if k == 27:  # ESC tuşuna basıldığında döngüden çık
        break

print("\n[INFO] Uygulama sonlandırılıyor.")
cam.release()
cv2.destroyAllWindows()