import cv2
import numpy as np
from ultralytics import YOLO

# Load the trained YOLOv8 model
model_path = 'runs/detect/train3/weights/best.pt'
model = YOLO(model_path)

# Define class ID for 'license-plate'
license_plate_class_id = 0  # This corresponds to 'license-plate'

# Open the video file
video_path = 'videos/cars.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_with_plates.avi', fourcc, fps, (width, height))

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the current frame
    results = model(frame)

    # Process the results
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Get boxes as numpy array
        classes = result.boxes.cls.cpu().numpy()  # Get class IDs as numpy array

        for i, box in enumerate(boxes):
            if int(classes[i]) == license_plate_class_id:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green rectangle

    # Write the frame with bounding boxes to the output video
    out.write(frame)

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing complete. Output saved to 'output_with_plates.avi'.")
