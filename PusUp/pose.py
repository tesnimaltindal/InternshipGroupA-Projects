import cv2
import mediapipe as mp
import os
import numpy as np

# Açı hesaplama fonksiyonu
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Video kaynağını aç (Hata düzeltildi: cap artık bir VideoCapture nesnesi)
video_source = 'videos/v2.mp4'
cap = cv2.VideoCapture(video_source)

# Hata kontrolü
if not cap.isOpened():
    print("HATA: Video kaynağı açılamadı. Dosya yolunu kontrol edin.")
    exit()

# *** Video Kaydetme Eklentisi ***
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
output_video_path = os.path.join(output_dir, 'pose_video.mp4')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30

out = None

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video bitti veya kare okunamadı. Döngüden çıkılıyor.")
            break
        
        # Video boyutunu yeniden boyutlandır
        original_height, original_width, _ = frame.shape
        new_width = 800
        ratio = new_width / original_width
        new_height = int(original_height * ratio)
        frame = cv2.resize(frame, (new_width, new_height))
        
        # Eğer video yazıcı henüz başlatılmadıysa, yeniden boyutlandırılmış kare boyutlarıyla başlat
        if out is None:
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (new_width, new_height))
            if not out.isOpened():
                print("HATA: Video kaydedici başlatılamadı.")
                exit()
        
        # Görüntüyü işle
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Eğer iskelet noktaları tespit edildiyse, onları videonun üzerine çiz
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

        # Ekranda görüntüyü göster
        cv2.imshow('Pushup Pose', image)

        # İşlenmiş kareyi videoya yaz
        out.write(image)

        # 'q' tuşuna basıldığında döngüden çık
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Kaynakları serbest bırak
cap.release()
out.release()
cv2.destroyAllWindows()