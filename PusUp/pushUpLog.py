import cv2
import mediapipe as mp
import numpy as np
import csv
from datetime import datetime
import os

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

# Sayaç değişkenleri
counter = 0
stage = "up"
feedback = ""
log_data = [] 
rep_start_time = None

# --- Video Kaynağı Seçimi ---
video_source = 'videos/v1.mp4'

cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("HATA: Video kaynağı açılamadı.")
    exit()

# *** Video Kaydetme Eklentisi ***
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
output_video_path = os.path.join(output_dir, 'antrenman_kaydi.mp4')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30

# Video yazarını başlattık, ancak boyutları döngü içinde ayarlayacağız.
out = None 

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                body_angle = calculate_angle(shoulder, hip, ankle)

                if elbow_angle > 160 and stage == "down":
                    if body_angle >= 160:
                        stage = "up"
                        counter += 1
                        rep_end_time = datetime.now()
                        rep_duration = (rep_end_time - rep_start_time).total_seconds() if rep_start_time else 0
                        log_data.append([counter, "Basarili", rep_start_time.strftime("%Y-%m-%d %H:%M:%S.%f") if rep_start_time else '', rep_end_time.strftime("%Y-%m-%d %H:%M:%S.%f"), f"{rep_duration:.2f}", f"{body_angle:.2f}", f"{elbow_angle:.2f}"])
                        feedback = ""
                    else:
                        feedback = "Sirtini duz tut! Push Up sayilmadi."
                        log_data.append([counter + 1, "Basarisiz", rep_start_time.strftime("%Y-%m-%d %H:%M:%S.%f") if rep_start_time else '', datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), "", f"{body_angle:.2f}", f"{elbow_angle:.2f}"])
                        stage = "up"
                
                if elbow_angle < 90 and stage == "up":
                    stage = "down"
                    rep_start_time = datetime.now()
                
                if feedback:
                    cv2.putText(image, feedback, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.putText(image, 'Counter: ' + str(counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, 'Stage: ' + stage, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                mp_drawing.draw_landmarks(
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

if log_data:
    with open('log/antrenman_log.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Rep', 'Durum', 'Baslangic Zamani', 'Bitis Zamani', 'Sure (saniye)', 'Vucut Acisi', 'Dirsek Acisi'])
        writer.writerows(log_data)
    print("Antrenman verileri 'antrenman_log.csv' dosyasına kaydedildi.")