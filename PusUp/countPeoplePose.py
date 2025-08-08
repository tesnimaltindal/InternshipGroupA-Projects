import cv2
import mediapipe as mp
import numpy as np
import csv
from datetime import datetime
from ultralytics import YOLO
import supervision as sv

# Açı hesaplama fonksiyonu
#şınav formunu belirlemek için iskelet noktalarını kullan
# Üç nokta (örneğin omuz-dirsek-bilek) arasındaki açıyı hesaplayarak,
#kol ne kadar büküldü
def calculate_angle(a, b, c):
    a = np.array(a) # İlk nokta (omuz
    b = np.array(b) # Orta nokta (dirsek)
    c = np.array(c) # Son nokta (bilek)
    #arctan2 fonksiyonu, iki nokta arasında bir vektör oluşturur 
    #1>b' noktasından 'c' noktasına 
    #2>'b' noktasından 'a' noktasına 
    #radyandan dereceye>np.degrees()


    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    
    if angle > 180.0:
        angle = 360 - angle
     #açı her zaman 0 ile 180 derece arasında
    return angle

# MediaPipe ve YOLO modellerini yükle
mp_drawing = mp.solutions.drawing_utils#çizim araçlarını içeri # MediaPipe'ın tespit ettiği iskelet noktalarını alıp,
#onları videonun üzerine çizmek ve birbirine bağlamak için  
mp_pose = mp.solutions.pose #vücut pozisyonu tahmin modeli
yolo_model = YOLO('yolov8n.pt')

# Supervision annotator'ları
box_annotator = sv.BoxAnnotator(thickness=2, color=sv.Color(255, 0, 0))
label_annotator = sv.LabelAnnotator(
    text_thickness=2, 
    text_scale=1, 
    text_color=sv.Color(255, 255, 255), 
    color=sv.Color(0, 0, 0), 
    text_padding=5
)

# Sayaç değişkenleri
counter = 0## Başarılı şınav sayısı
rep_attempt_counter = 0 # Toplam şınav denemesi sayısı
stage = "up"
feedback = "" # Kullanıcıya geri bildirim verecek değişken
log_data = [] # her şınavın bir kaydı olacak şekilde liste tutulur  
rep_start_time = None# hareketin 'yukarı' durumundan 'aşağı' durumuna geçtiği an, yani dirsekler bükülmeye başladığında güncellenir
 #Her şınavın başlangıç zamanı için 

person_count = 0

# Her N karede bir YOLO tespiti yap
#CPU zaman bölme , iş yükü azaltma amaçlı 
frame_counter = 0
yolo_detection_interval = 3

# Video Kaynağı Seçimi
video_source = 'videos/m2.mp4'

cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("HATA: Video kaynağı açılamadı.")
    exit()

# Video Kaydetme Eklentisi 
# Kaydedilecek video için dosya yolunu belirle
output_video_path = 'output/recorded_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Video codec ayarı
# VideoWriter nesnesini başlat (FPS ve boyut dinamik olarak ayarlanıyor)
fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
out = cv2.VideoWriter(output_video_path, fourcc, fps, (800, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 800 / cap.get(cv2.CAP_PROP_FRAME_WIDTH))))

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose: #minimum güvenilirlik 0.5
    #MediaPipe modeli kişi "tespit ettim" demesi için gereken minimum güvenilirlik seviyesi
    #min_tracking_confidence=0.5 kişiyi ilk kez tespit ettikten sonra, onu takip ederken de minimum %50 güvenilirlik seviyesini korumasını sağlar.
    while cap.isOpened():
        ret, frame = cap.read() #kare başarıyla okunursa, ret True olur ve frame record
        if not ret:
            print("Video bitti veya kare okunamadı. Döngüden çıkılıyor.")
            break

        height, width, _ = frame.shape
        new_width = 800
        ratio = new_width / width
        new_height = int(height * ratio)
        frame = cv2.resize(frame, (new_width, new_height))

        # Kişi Sayımı (YOLO)
        if frame_counter % yolo_detection_interval == 0:  #modelin her 3 karede bir çalışması için #akıcılık

            results_yolo = yolo_model(frame, classes=0, conf=0.25, verbose=False)[0] #YOLO modelini o anki video karesi üzerinde çalıştırır
            #classes=0 insanlar içn 
            #classes=0
            #verbose kelime anlamı olarak "ayrıntılı" demektir > gereksiz ayrıntıları terminale yazdırmıyoruz örn:inferencetime 12.3 ms
            detections = sv.Detections.from_ultralytics(results_yolo)#daha düzenli bir format yolo>supervision 
            person_count = len(detections) #Tespitten gelen nesne sayısını alarak anlık kişi sayısı
        
        labels = [ #Tespit edilen her kişi için ekranda görünecek etiket
            f"Kisi {idx+1} ({confidence:.2f})"
            for idx, (class_id, confidence) in enumerate(zip(detections.class_id, detections.confidence))
        ]
        #YOLO'nun tespit ettiği sınırlayıcı kutuları ve etiketleri videonun üzerine çiz
        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        #  Pose Tahmini ve Şınav Sayacı (MediaPipe)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)##bgr to rgb 
        image.flags.writeable = False #salt okunur 

        #görüntüyü MediaPipe poz tahmin modeline gönder > tüm işi yaptığı asıl satırdır.
        #Model,görüntüdeki iskelet noktalarını tespit eder ve sonuçları results değişkenine kaydeder.
        results_pose = pose.process(image)
        image.flags.writeable = True # görüntü tekrar yazılabilir çevir
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #rgb to bgr 
        
        frame_counter += 1

        try:
            if results_pose.pose_landmarks:
                landmarks = results_pose.pose_landmarks.landmark
                # Gerekli iskelet noktalarını al (sol kol)
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                # Duruş analizi için gerekli ekstra noktaları al (kalça, ayak bileği)
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                 # Duruş analizi için gerekli ekstra noktaları al (kalça, ayak bileği)

                # Dirsek açısı
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                # Vücut duruş açısı (omuz-kalça-ayak bileği)
                body_angle = calculate_angle(shoulder, hip, ankle)

                if elbow_angle > 160 and stage == "down":# Dirsek düz (yukarı pozisyon)
                    #Sayma mantığı: Dirsek açısını ve vücut duruşunu kullanarak say
                    # Sadece duruş doğruysa say
                    rep_attempt_counter += 1 #denemeyi bir arttır her seferinde  
                    if body_angle >= 160:
                        stage = "up"
                        counter += 1
                        rep_end_time = datetime.now()
                        rep_duration = (rep_end_time - rep_start_time).total_seconds() if rep_start_time else 0
                        log_data.append([rep_attempt_counter, "Basarili", rep_start_time.strftime("%Y-%m-%d %H:%M:%S.%f") if rep_start_time else '', rep_end_time.strftime("%Y-%m-%d %H:%M:%S.%f"), f"{rep_duration:.2f}", f"{body_angle:.2f}", f"{elbow_angle:.2f}"])
                        feedback = ""
                    else: # Eğer vücut düz değilse (bel bükülmüşse)
                        feedback = "Sirtini duz tut! Push Up sayilmadi."
                        log_data.append([rep_attempt_counter, "Basarisiz", rep_start_time.strftime("%Y-%m-%d %H:%M:%S.%f") if rep_start_time else '', datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), "", f"{body_angle:.2f}", f"{elbow_angle:.2f}"])
                        stage = "up"
                
                if elbow_angle < 90 and stage == "up":
                    stage = "down"
                    rep_start_time = datetime.now()
                # İskelet noktalarını videonun üzerine çiz
                mp_drawing.draw_landmarks(
                    image,
                    results_pose.pose_landmarks, # koordinatları
                    mp_pose.POSE_CONNECTIONS, #skeletin hangi noktalarının birbirine bağlanacağını belirten önceden tanımlanmış bir şablondur.
                    # Bu sayede sadece noktalar değil, aynı zamanda vücut iskeletini gösteren çizgiler de çizilir.
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
            else:
                feedback = "Kisi tespit edilemedi."
        except Exception as e:
            feedback = f"Hata: {e}"
            pass

        cv2.putText(image, 'Counter: ' + str(counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, 'Stage: ' + stage, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, 'Num.OfPerson: ' + str(person_count), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        if feedback:
            cv2.putText(image, feedback, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Pushup Counter and Person Detector', image)

        #İşlenmiş kareyi videoya yaz
        out.write(image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
# Video yazıcıyı serbest bırak
out.release()
cv2.destroyAllWindows()

if log_data:
    with open('log/many_log.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Rep', 'Durum', 'Baslangic Zamani', 'Bitis Zamani', 'Sure (saniye)', 'Vucut Acisi', 'Dirsek Acisi'])
        writer.writerows(log_data)
    print("Antrenman verileri 'many_log.csv' dosyasına kaydedildi.")