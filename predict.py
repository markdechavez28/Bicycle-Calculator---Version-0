import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

MODEL_PATH = r"C:\Users\Mark\Desktop\Bicycle_Detector\runs\detect\train8\weights\best.pt" 
model = YOLO(MODEL_PATH)

tracker = DeepSort(max_age=30)

VIDEO_PATH = r"C:\Users\Mark\Desktop\Bicycle_Detector\38_yolov3_train_custom_model\code\videos\bicycle.mp4"  
cap = cv2.VideoCapture(VIDEO_PATH)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

BICYCLE_CLASS_ID = 0  

bicycle_count = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    detections = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  
        confidences = result.boxes.conf.cpu().numpy()  
        class_ids = result.boxes.cls.cpu().numpy().astype(int) 

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            if class_id == BICYCLE_CLASS_ID and conf > 0.4:  
                x1, y1, x2, y2 = box
                detections.append(([x1, y1, x2, y2], conf, class_id))

    tracked_objects = tracker.update_tracks(detections, frame=frame)

    for track in tracked_objects:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_tlbr())

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Bicycle {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        bicycle_count.add(track_id)

    cv2.putText(frame, f"Total Bicycles: {len(bicycle_count)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Bicycle Detection & Counting", frame)

    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Total unique bicycles detected: {len(bicycle_count)}")
