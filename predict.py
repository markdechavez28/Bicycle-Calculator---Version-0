import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO model (Replace with your model path)
MODEL_PATH = r"C:\Users\Mark\Desktop\Bicycle_Detector\runs\detect\train8\weights\best.pt"  # Update with your trained YOLO model path
model = YOLO(MODEL_PATH)

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Load video
VIDEO_PATH = r"C:\Users\Mark\Desktop\Bicycle_Detector\38_yolov3_train_custom_model\code\videos\bicycle.mp4"  # Update with your video file
cap = cv2.VideoCapture(VIDEO_PATH)

# Video writer setup (Optional: Save output video)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# Bicycle class ID (Depends on your dataset - adjust if necessary)
BICYCLE_CLASS_ID = 0  # Check your YOLO dataset's class index for bicycles

# Unique bicycle count
bicycle_count = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO on the frame
    results = model(frame)

    detections = []

    # Process YOLO results
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            if class_id == BICYCLE_CLASS_ID and conf > 0.4:  # Adjust confidence threshold if needed
                x1, y1, x2, y2 = box
                detections.append(([x1, y1, x2, y2], conf, class_id))

    # Track objects with DeepSORT
    tracked_objects = tracker.update_tracks(detections, frame=frame)

    for track in tracked_objects:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_tlbr())

        # Draw bounding box & ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Bicycle {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Add to unique bicycle count
        bicycle_count.add(track_id)

    # Display count on frame
    cv2.putText(frame, f"Total Bicycles: {len(bicycle_count)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Show frame
    cv2.imshow("Bicycle Detection & Counting", frame)

    # Save frame to output video
    out.write(frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Total unique bicycles detected: {len(bicycle_count)}")
