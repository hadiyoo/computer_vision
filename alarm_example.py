import cv2
import torch
from ultralytics import YOLO
import numpy as np
import time
import pygame
import threading
import winsound
import os


class RealtimeYOLODetectionWithAlerts:
    def __init__(self, model_path ="yolov5su.pt",target_object="person"):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(0)
        self.colors=np.random.randint(0, 255, size=(80,3), dtype="uint8")
        self.target_object = target_object
        self.alert_active = False

        # Initialize pygame mixer for sound alerts
        pygame.mixer.init()
        if os.path.exists("alarm.mp3"):
            self.alert_sound = pygame.mixer.Sound("alarm.mp3")
            self.use_file = True
        else:
            print("⚠️ alarm.mp3 not found, using default beep.")
            self.use_file = False


        #self.alert_sound = pygame.mixer.Sound("../DATA/alert.wav")  # Short alert sound

    def play_alarm(self):
           if self.use_file:
            self.alert_sound.play()
            time.sleep(2)  # Play alarm for 2 seconds
            self.alert_sound.stop()
           else:
            winsound.Beep(1000, 2000)  # 1000 Hz for 2 seconds
            self.alert_active = False

        #self.alarm_sound.play()
        #time.sleep(2)  # Play alarm for 2 seconds
        #self.alert_sound.stop()
        #self.alert_active = False

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            start_time = time.time()

            # Perform detection
            results = self.model(frame)

            # Run inference
            results = self.model(frame)

            target_detected = False

            # Process results
            for result in results:
              boxes = result.boxes
              for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{result.names[cls]}: {conf:.2f}"  

                # Draw detections
                color = [int(c) for c in self.colors[cls]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Check if the detected object is the target
                if result.names[cls].lower() == self.    target_object.lower():
                  target_detected = True
                  cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            

            #Trigger alarm if target object is detected
            if target_detected and not self.alert_active:
                self.alert_active = True
                threading.Thread(target=self.play_alarm).start()
                #visual alert
                cv2.putText(frame,f"{self.target_object.upper()} DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Calculate and display FPS
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            #display the frame
            cv2.imshow("YOLOv5 Real-time Detection with Alerts", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

#Usage
detector = RealtimeYOLODetectionWithAlerts(target_object="person")
detector.run()       