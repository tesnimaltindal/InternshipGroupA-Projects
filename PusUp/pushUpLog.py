import cv2
import mediapipe as mp
import numpy as np
import csv
from datetime import datetime
import os

# Açı hesaplama fonksiyonu

#şınav formunu belirlemek için iskelet noktalarını kullan
#Üç nokta (örneğin omuz-dirsek-bilek) arasındaki açıyı hesaplayarak,
#kol ne kadar büküldü tespiti

def calculate_angle(a, b, c):
    a = np.array(a) # İlk nokta (omuz)
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

mp_drawing = mp.solutions.drawing_utils #çizim araçlarını çağır # MediaPipe'ın tespit ettiği iskelet noktalarını alıp,
#onları videonun üzerine çizmek ve birbirine bağlamak için  
mp_pose = mp.solutions.pose #vücut pozisyonu tahmin modeli

# Sayaç değişkenleri
counter = 0# Başarılı şınav sayısı
rep_attempt_counter = 0 # Toplam şınav denemesi sayısı
stage = "up"
feedback = ""  # Kullanıcıya geri bildirim verecek değişken
log_data = [] # her şınavın kaydı olacak şekilde tutulacak liste  
rep_start_time = None # hareketin 'yukarı' durumundan 'aşağı' durumuna geçtiği an,
 #yani dirsekler bükülmeye başladığında güncellenir
 #Her şınavın başlangıç zamanı için 

#Video Kaynağı Seçimi
video_source = 'videos/v4.mp4'

cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("HATA: Video kaynağı açılamadı.")
    exit()

# video Kaydetme Eklentisi
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
output_video_path = os.path.join(output_dir, 'antrenman_kaydi.mp4')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30

# Video yazarını başlattık, ancak boyutları döngü içinde ayarlayacağız.
out = None  #VideoWriter nesnesini ilk başta boş bir şekilde tanımlar

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose: #minimum güvenilirlik 0.5
    #MediaPipe modeli kişi "tespit ettim" demesi için gereken minimum güvenilirlik seviyesi
    #min_tracking_confidence=0.5 kişiyi ilk kez tespit ettikten sonra, onu takip ederken de minimum %50 güvenilirlik seviyesini korumasını sağlar.
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
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
        
        # Video kaydetme kısmı ve diğer tüm işleme kodları
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #bgr to rgb 
        image.flags.writeable = False  #salt okunur
        results = pose.process(image) #görüntüyü MediaPipe poz tahmin modeline gönder > tüm işi yaptığı asıl satırdır.
        image.flags.writeable = True # görüntü tekrar yazılabilir çevir
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            if results.pose_landmarks:
                #if results_pose.pose_landmarks:güvenlik kontrolüdür. 
                # MediaPipe modeli, video karesinde bir kişi tespit edemezse, results_pose.pose_landmarks nesnesi boş olur. 
                # Bu if koşulu, kodun boş bir nesneye erişmeye çalışmasını engelleyerek programın çökmesini önler.
                landmarks = results.pose_landmarks.landmark
                #landmarks = results_pose.pose_landmarks.landmark modelin bulduğu tüm 33 iskelet noktasını içeren veri yapısına kolay erişim sağlar.
                #landmarks değişkeni, her bir noktanın x, y ve z koordinatlarını içerir.
                #Gerekli iskelet noktalarını al (sol kol)
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
                if elbow_angle > 160 and stage == "down":
                    rep_attempt_counter += 1 #denemeyi bir arttır her seferinde  
                    if body_angle >= 160:
                        stage = "up"
                        counter += 1
                        rep_end_time = datetime.now()#Şınavın bittiği anın zamanını kaydı
                        rep_duration = (rep_end_time - rep_start_time).total_seconds() if rep_start_time else 0 #Başlangıç ve bitiş zamanı arasındaki fark
                        #  log_data listesine eklenecekler
                        log_data.append([rep_attempt_counter, "Basarili", rep_start_time.strftime("%Y-%m-%d %H:%M:%S.%f") if rep_start_time else '', rep_end_time.strftime("%Y-%m-%d %H:%M:%S.%f"), f"{rep_duration:.2f}", f"{body_angle:.2f}", f"{elbow_angle:.2f}"]) 
                        feedback = ""
                    else:
                        #Eğer şınav tamamlandığında body_angle 160 derecenin altındaysa
                        feedback = "Sirtini duz tut! Push Up sayilmadi."# Eğer vücut düz değilse (bel bükülmüşse)
                        #Sayılmayan şınavın verilerini de, durumu 'başarısız' 
                        log_data.append([rep_attempt_counter, "Basarisiz", rep_start_time.strftime("%Y-%m-%d %H:%M:%S.%f") if rep_start_time else '', datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), "", f"{body_angle:.2f}", f"{elbow_angle:.2f}"])
                        stage = "up"
                        #rep attmepr counter , sayılmayan bir şınavı bile antrenman geçmişinde bir deneme olarak işaretlemek için
                        #counter değişkeni, yalnızca başarılı şınavları sayar
                        #Loglama dosyasındaki Rep sütunu ise, başarılı veya başarısız olmasına bakılmaksızın yapılan tüm denemeleri sırayla numaralandırır
                
                if elbow_angle < 90 and stage == "up":
                    stage = "down"
                    rep_start_time = datetime.now()
                 # Geri bildirimi ekrana yazdır
                if feedback:
                    cv2.putText(image, feedback, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # Sayaç ve durumu ekrana yazdır
                cv2.putText(image, 'Counter: ' + str(counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, 'Stage: ' + stage, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                # İskelet noktalarını videonun üzerine çiz
                mp_drawing.draw_landmarks( #boşsa, yani model bir kişi bulamamışsa, kod çizim yapmaya çalışmaz
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
            else:
                feedback = "Kisi tespit edilemedi."
                       
        except Exception as e:
            feedback = f"Hata: {e}"
            pass

        out.write(image)
        cv2.imshow('Pushup Log and Counter', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()

if log_data: #log_data listesi
    with open('log/antrenman_log.csv', 'w', newline='') as file:
        writer = csv.writer(file) #Python'ın csv kütüphanesini kullanarak yazıcı nesnesi #verileri virgülle ayrılmış dosya 
        writer.writerow(['Rep', 'Durum', 'Baslangic Zamani', 'Bitis Zamani', 'Sure (saniye)', 'Vucut Acisi', 'Dirsek Acisi'])
        writer.writerows(log_data)#program çalışırken toplanılan tüm verileri içeren log_data listesini alır ve her bir alt listeyi
        #dosyanın yeni bir satırına yazar
    print("Antrenman verileri 'antrenman_log.csv' dosyasına kaydedildi.")

    #uygulamanın anlık bir sayaç olmaktan çıkıp, kullanıcının ilerlemesini takip edebilmesi içn alan oluşturur 