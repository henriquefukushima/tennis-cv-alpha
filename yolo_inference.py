from ultralytics import YOLO

model = YOLO('models/yolov8x') # Load a pretrained YOLOv8 model

result = model.predict(
    'input_videos/input_video.mp4', 
    save=True,  # Save the output video with detections
    show=True,  # Display the output video with detections
) 

print(result)
print("BOXES: ")
for box in result[0].boxes:
    print(f"Box: {box.xyxy}, Confidence: {box.conf}, Class: {box.cls}")