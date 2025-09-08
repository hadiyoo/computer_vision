import cv2
import torch
from ultralytics import YOLO
import numpy as np
import time
import pygame
import threading


class RealtimeYOLODetection:
    def __init__(self, model_path ="yolov5su.pt"):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(0)
        self.colors=np.random.randint(0, 255, size=(80,3), dtype="uint8")

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            start_time = time.time()

            # Perform detection
            results = self.model(frame)[0]


            # Run inference
            results = self.model(frame, stream=True)  # stream=True for real-time

            for r in results:  # each frame result
               boxes = r.boxes  # Boxes object

               for box in boxes:  # iterate over detections
                    # Bounding box
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()

                    # Class & confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{r.names[cls]}: {conf:.2f}"

                 # Draw
                color = [int(c) for c in self.colors[cls]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


            #process results
            #for result in results.boxes:
              # boxes = result.boxes.xyxy.cpu().numpy()
               #for box in boxes:
                  # x1, y1, x2, y2 = box.xyxy[0].astype(int)
                 #  cls = int(box.cls[0])
                  # conf = float(box.conf[0])
                  # label = f"{results.names[cls]}: {conf:.2f}"
                  # color = [int(c) for c in self.colors[cls]]
                  # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                  # cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Calculate and display FPS
            fps = 1.0 / (time.time() - start_time)   
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            #display the frame
            cv2.imshow("YOLOv5 Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        self.cap.release()
        cv2.destroyAllWindows()

#Usage
detector = RealtimeYOLODetection()
detector.run()        
