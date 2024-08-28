import cv2
import numpy as np
import pytesseract
from PIL import Image
from ultralytics import YOLO

# Configure Tesseract executable path if necessary (e.g., on Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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
out = cv2.VideoWriter('output_with_plates_and_text.avi', fourcc, fps, (width, height))

def extract_text_from_image(image):
    """Extract text from an image using Tesseract OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    pil_image = Image.fromarray(gray)  # Convert to PIL image for Tesseract
    text = pytesseract.image_to_string(pil_image, config='--psm 8')  # OCR
    return text.strip()

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
                roi = frame[y1:y2, x1:x2]  # Region of interest (license plate)

                # Extract text from the license plate
                plate_text = extract_text_from_image(roi)

                # Draw rectangle and put text on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green rectangle
                cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Put text

    # Write the frame with bounding boxes and text to the output video
    out.write(frame)

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing complete. Output saved to 'output_with_plates_and_text.avi'.")
