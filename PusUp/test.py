import cv2

video_path = 'videos/v3.mp4' 

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("HATA: Video dosyası açılamadı. Lütfen dosya yolunu ve adını kontrol edin.")
else:
    print("Video dosyası başarıyla açıldı. Pencere açılacak.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video bitti veya kare okunamadı.")
            break
        
        cv2.imshow('Video Test', frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Program sonlandı.")
