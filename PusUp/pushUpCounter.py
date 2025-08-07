import cv2
import mediapipe as mp
import numpy as np

# Açı hesaplama fonksiyonu 
#şınav formunu belirlemek için iskelet noktalarını kullan
# Üç nokta (örneğin omuz-dirsek-bilek) arasındaki açıyı hesaplayarak,
#kol ne kadar büküldü
def calculate_angle(a, b, c):
    a = np.array(a)  # İlk nokta (omuz)
    b = np.array(b)  # Orta nokta (dirsek)
    c = np.array(c)  # Son nokta (bilek)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    #arctan2 fonksiyonu, iki nokta arasında bir vektör oluşturur 
    #1>b' noktasından 'c' noktasına 
    #2>'b' noktasından 'a' noktasına 
    #radyandan dereceye>np.degrees()
    if angle > 180.0:
        angle = 360 - angle
    #açı her zaman 0 ile 180 derece arasında
    return angle

mp_drawing = mp.solutions.drawing_utils #çizim araçlarını içeri # MediaPipe'ın tespit ettiği iskelet noktalarını alıp, onları videonun üzerine çizmek ve birbirine bağlamak için  
mp_pose = mp.solutions.pose#vücut pozisyonu tahmin modeli

# Sayaç değişkenleri
counter = 0
stage = "up"
feedback = ""  # Kullanıcıya geri bildirim verecek değişken

# Video Kaynağı Seçimi
# Canlı kamera kullanmak için '0'
# Video dosyası kullanmak için dosya yolu
video_source = 'videos/v6.mp4' # Örnek: 'video.mp4' veya 0

cap = cv2.VideoCapture(video_source)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose: #minimum güvenilirlik 0.5
    while cap.isOpened(): #her kareyi sırayla işlemek için
        ret, frame = cap.read() #kare başarıyla okunursa, ret True olur ve frame record
        if not ret:
            break

        height, width, _ = frame.shape
        new_width = 800
        ratio = new_width / width
        new_height = int(height * ratio)
        frame = cv2.resize(frame, (new_width, new_height))

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #bgr to rgb 
        image.flags.writeable = False#salt okunur 
        results = pose.process(image) #görüntüyü MediaPipe poz tahmin modeline gönder > tüm işi yaptığı asıl satırdır.
        #Model,görüntüdeki iskelet noktalarını tespit eder ve sonuçları results değişkenine kaydeder.
        image.flags.writeable = True # görüntü tekrar yazılabilir çevir
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            
            # Gerekli iskelet noktalarını al (sol kol)
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Duruş analizi için gerekli ekstra noktaları al (kalça, ayak bileği)
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Dirsek açısı
            elbow_angle = calculate_angle(shoulder, elbow, wrist)

            # Vücut duruş açısı (omuz-kalça-ayak bileği)
            body_angle = calculate_angle(shoulder, hip, ankle)

            # Duruş kontrolü ve geri bildirim
            if body_angle < 170:  # Eğer vücut düz değilse (bel bükülmüşse)
                feedback = "Sirtini duz tut!"
            else:
                feedback = ""

            # Sayma mantığı: Dirsek açısını ve vücut duruşunu kullanarak say
            if body_angle >= 170:  # Sadece duruş doğruysa say
                if elbow_angle > 160:  # Dirsek düz (yukarı pozisyon)
                    stage = "up"
                if elbow_angle < 90 and stage == "up":  # Dirsek bükülmüş (aşağı pozisyon) ve önceki durum "yukarı" ise
                    stage = "down"
                    counter += 1
                       
        except:
            pass

        # Geri bildirimi ekrana yazdır
        cv2.putText(image, feedback, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Sayaç ve durumu ekrana yazdır
        cv2.putText(image, 'Counter: ' + str(counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(image, 'Stage: ' + stage, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

        # İskelet noktalarını videonun üzerine çiz
        if results.pose_landmarks: #boşsa, yani model bir kişi bulamamışsa, kod çizim yapmaya çalışmaz
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks, # koordinatları
                mp_pose.POSE_CONNECTIONS, #skeletin hangi noktalarının birbirine bağlanacağını belirten önceden tanımlanmış bir şablondur.
                 # Bu sayede sadece noktalar değil, aynı zamanda vücut iskeletini gösteren çizgiler de çizilir.
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),#özelleştirme
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        cv2.imshow('Pushup Counter', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()