import cv2
import pytesseract
from PIL import Image
from ultralytics import YOLO

# Initialize Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Update with your Tesseract path if necessary

# Load the trained YOLOv8 model
model_path = 'runs/detect/train3/weights/best.pt'
model = YOLO(model_path)

# Define class ID for 'license-plate'
license_plate_class_id = 0  # This corresponds to 'license-plate'

# Open the video file
video_path = 'videos/cars2.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_with_plates_and_text2.avi', fourcc, fps, (width, height))

def extract_text_from_image(image):
    """Extract text from an image using Tesseract OCR with a character whitelist."""
    pil_image = Image.fromarray(image)
    custom_config = r'--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    text = pytesseract.image_to_string(pil_image, config=custom_config)
    return text.strip()

def draw_text_with_background(image, text, position, font_scale=0.9, font_thickness=2):
    """Draw text with a background color on the image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    
    x, y = position
    background_top_left = (x, y - text_size[1] - 10)
    background_bottom_right = (x + text_size[0] + 10, y)
    
    # Draw the background rectangle
    cv2.rectangle(image, background_top_left, background_bottom_right, (0, 255, 0), -1)  # Green background
    
    # Draw the text
    cv2.putText(image, text, (x + 5, y - 5), font, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)  # White text

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
                draw_text_with_background(frame, plate_text, (x1, y1 - 10))  # Draw text with background

    # Write the frame with bounding boxes and text to the output video
    out.write(frame)

# Release everything if job is finished
cap.release()
out.release()
print("Video processing complete. Output saved to 'output_with_plates_and_text.avi'.")
