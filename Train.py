import cv2
import numpy as np
from easyocr import Reader
from ultralytics import YOLO
import os
import uuid

# Initialize EasyOCR
reader = Reader(['en'])

# Load the trained YOLOv8 model
model_path = 'license_plate_detector.pt'
model = YOLO(model_path)

# Define class ID for 'license-plate'
license_plate_class_id = 0  # This corresponds to 'license-plate'

# Open the video file
video_path = 'videos/cars5.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_with_easyocr.avi', fourcc, fps, (width, height))

# Define paths for saving images
folder_path = './license_plate_images/'
os.makedirs(folder_path, exist_ok=True)

def extract_text_from_image(image):
    """Extract text from an image using EasyOCR and apply post-processing."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    results = reader.readtext(gray_image)
    if results:
        # Filter results by area to reduce false positives
        filtered_results = [result for result in results if np.prod(np.subtract(result[0][2], result[0][0])) > 100]
        if filtered_results:
            # Sort by confidence and take the most confident result
            filtered_results.sort(key=lambda x: x[2], reverse=True)
            return filtered_results[0][1].strip()  # Take the text from the highest confidence result
    return ""

def draw_text_with_background(image, text, position, font_scale=0.9, font_thickness=2, color=(0, 255, 0)):
    """Draw text with a background color on the image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    x, y = position
    background_top_left = (x, y - text_size[1] - 10)
    background_bottom_right = (x + text_size[0] + 10, y)
    cv2.rectangle(image, background_top_left, background_bottom_right, color, -1)
    cv2.putText(image, text, (x + 5, y - 5), font, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)

# Define the scale factor for resizing
scale_factor = 0.5  # 50% of the original size

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the current frame
    results = model(frame, conf=0.15)  # Lowered confidence threshold

    # Process the results
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for i, (box, cls) in enumerate(zip(boxes, classes)):
            if int(cls) == license_plate_class_id:
                x1, y1, x2, y2 = map(int, box)
                roi = frame[y1:y2, x1:x2]  # Region of interest (license plate)

                # Extract text from the license plate using EasyOCR
                plate_text = extract_text_from_image(roi)

                # Save cropped license plate image
                img_name = f'{uuid.uuid1()}.jpg'
                cv2.imwrite(os.path.join(folder_path, img_name), roi)

                # Draw bounding box and text on the frame
                if plate_text:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    draw_text_with_background(frame, plate_text, (x1, y1 - 10))

    # Resize the frame
    resized_frame = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))

    # Write the frame with bounding boxes and text to the output video
    out.write(frame)

    # Display the resized frame
    cv2.imshow('Frame', resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
print("Video processing complete. Output saved to 'output_with_easyocr.avi'")