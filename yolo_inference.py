from ultralytics import YOLO

model = YOLO('yolov8x') # Load a pretrained YOLOv8 model

model.predict(
    'input_videos/input_video.mp4',
    save=True,  # Save the output video with detections
    show=True,  # Display the output video with detections
)  # Path to the input video)