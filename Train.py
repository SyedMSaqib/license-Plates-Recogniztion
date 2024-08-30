import cv2
import numpy as np
import pytesseract
from PIL import Image
from ultralytics import YOLO
from collections import deque

# Initialize Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Update with your Tesseract path if necessary

# Load the trained YOLOv8 model
model_path = 'runs/detect/train3/weights/best.pt'
model = YOLO(model_path)

# Define class ID for 'license-plate'
license_plate_class_id = 0  # This corresponds to 'license-plate'

# Open the video file
video_path = 'videos/cars4.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_with_multiple_stable_plates_debug.avi', fourcc, fps, (width, height))

# Define debug window size
debug_width = 800
debug_height = int(height * (debug_width / width))

def preprocess_license_plate(image):
    """Preprocess the license plate image to improve OCR accuracy."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_text_from_image(image):
    """Extract text from an image using Tesseract OCR with improved configuration."""
    preprocessed = preprocess_license_plate(image)
    pil_image = Image.fromarray(preprocessed)
    custom_config = r'--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(pil_image, config=custom_config)
    return text.strip()

def is_valid_plate(text):
    """Check if the extracted text is likely to be a valid license plate."""
    if len(text) < 5 or len(text) > 8:
        return False
    letters = sum(c.isalpha() for c in text)
    numbers = sum(c.isdigit() for c in text)
    return letters >= 2 and numbers >= 2

def draw_text_with_background(image, text, position, font_scale=0.9, font_thickness=2, color=(0, 255, 0)):
    """Draw text with a background color on the image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    x, y = position
    background_top_left = (x, y - text_size[1] - 10)
    background_bottom_right = (x + text_size[0] + 10, y)
    cv2.rectangle(image, background_top_left, background_bottom_right, color, -1)
    cv2.putText(image, text, (x + 5, y - 5), font, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)

class PlateTracker:
    def __init__(self, max_distance=50, max_frames=30):
        self.plates = {}
        self.max_distance = max_distance
        self.max_frames = max_frames
        self.next_id = 0

    def update(self, detections):
        new_plates = {}
        used_detections = set()

        # Update existing plates
        for plate_id, plate_info in self.plates.items():
            best_match = None
            best_iou = 0
            for i, (box, text) in enumerate(detections):
                if i in used_detections:
                    continue
                iou = self.calculate_iou(box, plate_info['box'])
                if iou > 0.3 and iou > best_iou:  # Lowered IOU threshold
                    best_match = (i, box, text)
                    best_iou = iou

            if best_match:
                i, box, text = best_match
                plate_info['box'] = box
                plate_info['text_history'].append(text)
                plate_info['frames_since_last_detection'] = 0
                new_plates[plate_id] = plate_info
                used_detections.add(i)
            else:
                plate_info['frames_since_last_detection'] += 1
                if plate_info['frames_since_last_detection'] < self.max_frames:
                    new_plates[plate_id] = plate_info

        # Add new plates
        for i, (box, text) in enumerate(detections):
            if i not in used_detections:
                new_plates[self.next_id] = {
                    'box': box,
                    'text_history': deque([text], maxlen=10),
                    'frames_since_last_detection': 0
                }
                self.next_id += 1

        self.plates = new_plates

    def get_stable_plates(self):
        stable_plates = []
        for plate_id, plate_info in self.plates.items():
            if len(plate_info['text_history']) > 3:  # Lowered stability threshold
                most_common_text = max(set(plate_info['text_history']), key=plate_info['text_history'].count)
                stable_plates.append((plate_info['box'], most_common_text))
        return stable_plates

    @staticmethod
    def calculate_iou(box1, box2):
        # Calculate IoU between two bounding boxes
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        inter_x1 = max(x1, x3)
        inter_y1 = max(y1, y3)
        inter_x2 = min(x2, x4)
        inter_y2 = min(y2, y4)
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            box1_area = (x2 - x1) * (y2 - y1)
            box2_area = (x4 - x3) * (y4 - y3)
            iou = inter_area / float(box1_area + box2_area - inter_area)
            return iou
        return 0

# Initialize the plate tracker
plate_tracker = PlateTracker()

frame_count = 0
# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"Processing frame {frame_count}")

    # Perform inference on the current frame
    results = model(frame, conf=0.3)  # Lowered confidence threshold

    # Process the results
    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
            if int(cls) == license_plate_class_id:
                x1, y1, x2, y2 = map(int, box)
                roi = frame[y1:y2, x1:x2]  # Region of interest (license plate)

                # Extract text from the license plate
                plate_text = extract_text_from_image(roi)

                # Validate the plate text
                if is_valid_plate(plate_text):
                    detections.append((box, plate_text))
                    print(f"Detected plate: {plate_text} at {box} with confidence {conf}")

    print(f"Total detections in this frame: {len(detections)}")

    # Update the plate tracker
    plate_tracker.update(detections)

    # Get stable plates and draw them
    stable_plates = plate_tracker.get_stable_plates()
    print(f"Stable plates in this frame: {len(stable_plates)}")
    for box, text in stable_plates:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        draw_text_with_background(frame, text, (x1, y1 - 10))

    # Draw all detections (including unstable ones) in red
    for box, _ in detections:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Draw frame number on the image
    draw_text_with_background(frame, f"Frame: {frame_count}", (10, 30), color=(255, 0, 0))

    # Write the frame with bounding boxes and text to the output video
    out.write(frame)

    # Resize the frame for display
    debug_frame = cv2.resize(frame, (debug_width, debug_height))

    # Display the resized frame
    cv2.imshow('Debug Frame', debug_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
print("Video processing complete. Output saved to 'output_with_multiple_stable_plates_debug.avi'.")