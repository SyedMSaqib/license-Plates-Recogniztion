from ultralytics import YOLO

# Load the trained model
model_path = 'runs/detect/train3/weights/best.pt'
model = YOLO(model_path)

# Perform inference on a video file
video_path = 'videos/cars.mp4'
results = model.predict(source=video_path)

# Filter results to keep only license plate detections
for result in results:
    detections = result.pandas().xyxy[0]  # Get the detection results in pandas DataFrame
    license_plates = detections[detections['name'] == 'license-plate']  # Filter for license plates
    
    # Save or display the filtered results
    # Annotate or process `license_plates` as needed
