import cv2
import numpy as np
from easyocr import Reader
from ultralytics import YOLO
from collections import defaultdict, Counter
import re
import csv
import datetime

# Initialize EasyOCR
reader = Reader(['en'])

# Load the trained YOLOv8 model
model_path = 'license_plate_detector.pt'
model = YOLO(model_path)

# Define class IDs
license_plate_class_id = 0  # This corresponds to 'license-plate'

# Open the video file (replace with your traffic feed source)
video_path = 'videos/CC.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4 format
out = cv2.VideoWriter('traffic_analysis_output2.mp4', fourcc, fps, (width, height))

# CSV file setup
csv_file = 'vehicle_crossings.csv'
csv_columns = ['Vehicle ID', 'License Plate', 'Time (UTC)']

with open(csv_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()

# Dictionary to store license plates for each vehicle
vehicle_license_plates = defaultdict(Counter)

# Set to keep track of unique vehicles
unique_vehicles = set()
vehicle_id_counter = 1  # Sequential vehicle ID counter

# Define a region of interest (ROI) for counting vehicles
roi_start = int(height * 0.65)  # Moved to 65% of the frame height
roi_end = int(height * 0.75)    # Moved to 75% of the frame height

def extract_text_from_image(image):
    """Extract text from an image using EasyOCR and apply post-processing."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray_image)
    if results:
        filtered_results = [result for result in results if np.prod(np.subtract(result[0][2], result[0][0])) > 100]
        if filtered_results:
            filtered_results.sort(key=lambda x: x[2], reverse=True)
            return filtered_results[0][1].strip()
    return ""

def clean_plate_text(plate_text):
    """Clean and correct the license plate text to include only 0-9 and A-Z."""
    if plate_text:
        plate_text = plate_text[1:].strip()
        if plate_text and plate_text[-1] in ['5', 'S']:
            plate_text = plate_text[:-1] + 'S'
        plate_text = ''.join(plate_text.split()) 
        plate_text = plate_text.upper()  # Ensure uppercase letters
        plate_text = re.sub(r'[^A-Z0-9]', '', plate_text)  # Remove any non-alphanumeric characters
    return plate_text

def draw_text_with_background(image, text, position, font_scale=0.9, font_thickness=2, color=(0, 255, 0)):
    """Draw text with a background color on the image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    x, y = position
    background_top_left = (x, y - text_size[1] - 10)
    background_bottom_right = (x + text_size[0] + 10, y)
    cv2.rectangle(image, background_top_left, background_bottom_right, color, -1)
    cv2.putText(image, text, (x + 5, y - 5), font, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)

# Process each frame
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Get current UTC time
    utc_time = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

    # Perform inference and tracking on the current frame
    results = model.track(frame, conf=0.15, persist=True)

    # Draw ROI lines
    cv2.line(frame, (0, roi_start), (width, roi_start), (255, 0, 0), 2)
    cv2.line(frame, (0, roi_end), (width, roi_end), (255, 0, 0), 2)

    # Extract bounding boxes, class IDs, and tracking IDs
    for result in results:
        if result.boxes.id is not None:
            for box, cls, track_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.id.cpu().numpy()):
                if int(cls) == license_plate_class_id:
                    x1, y1, x2, y2 = map(int, box)
                    roi = frame[y1:y2, x1:x2]

                    plate_text = extract_text_from_image(roi)
                    plate_text = clean_plate_text(plate_text)

                    if plate_text:
                        # Check if vehicle is crossing the ROI
                        vehicle_center_y = (y1 + y2) / 2
                        if roi_start <= vehicle_center_y <= roi_end and track_id not in unique_vehicles:
                            unique_vehicles.add(track_id)

                            # Assign a sequential vehicle ID
                            vehicle_id = vehicle_id_counter
                            vehicle_id_counter += 1

                            # Record the vehicle's license plate and crossing time (UTC) in the CSV
                            with open(csv_file, 'a', newline='') as csvfile:
                                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                                writer.writerow({
                                    'Vehicle ID': vehicle_id,
                                    'License Plate': plate_text,
                                    'Time (UTC)': utc_time
                                })

                        vehicle_license_plates[track_id][plate_text] += 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if plate_text:
                        most_common_plate = vehicle_license_plates[track_id].most_common(1)
                        if most_common_plate:
                            display_text = most_common_plate[0][0]
                            draw_text_with_background(frame, display_text, (x1, y1 - 10))

    # Display total vehicle count
    total_vehicles_text = f"Total Vehicles Passed: {vehicle_id_counter - 1}"
    draw_text_with_background(frame, total_vehicles_text, (10, 30), font_scale=0.7, color=(0, 0, 255))

    out.write(frame)

    cv2.imshow('Traffic Analysis', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Print final statistics
print("\nFinal Statistics:")
print(f"Total Vehicles Passed: {vehicle_id_counter - 1}")

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
