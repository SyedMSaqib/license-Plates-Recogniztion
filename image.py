import cv2
import numpy as np
from ultralytics import YOLO
from easyocr import Reader

# Load the YOLO model
model_path = 'runs/detect/train3/weights/best.pt'
model = YOLO(model_path)

# Initialize EasyOCR reader
ocr_reader = Reader(['en'])

# Read the image
image_path = "Images/3.JPG"
image = cv2.imread(image_path)

# Run YOLO model to detect objects
results = model(image_path)

# Get the detected objects
for detection in results[0].boxes:
    # Extract bounding box coordinates
    x1, y1, x2, y2 = map(int, detection.xyxy[0])
    label = detection.cls[0]
    
    # Check if the detected object is a license plate (assuming class 0 is license plates)
    if label == 0:
        # Crop the license plate region
        plate_img = image[y1:y2, x1:x2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to smooth out noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Perform OCR directly on the blurred grayscale image
        ocr_result = ocr_reader.readtext(blurred, detail=0)
        
        # Extract text from OCR results
        plate_text = ' '.join(ocr_result)
        
        # Draw bounding box on the original image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Set font parameters
        font_scale = 1
        font_thickness = 3
        font_color = (255, 255, 255)  # White color for text
        background_color = (0, 255, 0)  # Green color for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get the text size to create a background rectangle
        text_size = cv2.getTextSize(plate_text, font, font_scale, font_thickness)[0]
        text_x = x1
        text_y = y1 - 10
        background_top_left = (text_x, text_y - text_size[1] - 10)
        background_bottom_right = (text_x + text_size[0] + 10, text_y)
        
        # Draw the background rectangle
        cv2.rectangle(image, background_top_left, background_bottom_right, background_color, -1)
        
        # Draw the text on top of the background rectangle
        cv2.putText(image, plate_text, (text_x + 5, text_y - 5), font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)

# Save or display the result
cv2.imwrite('output_image.jpg', image)
# cv2.imshow('Detected Plates', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
