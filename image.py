import cv2
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
    
    # Check if the detected object is a license plate
    if label == 0:  # Assuming 0 is the class index for license plates
        # Crop the license plate region
        plate_img = image[y1:y2, x1:x2]
        
        # Perform OCR on the cropped region
        ocr_result = ocr_reader.readtext(plate_img)
        
        # Extract text from OCR results
        plate_text = ' '.join([text[1] for text in ocr_result])
        
        # Draw bounding box on the original image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Set font parameters for larger and bolder text
        font_scale = 1.5
        font_thickness = 3
        font_color = (0, 255, 0)  # Green color
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Draw text on the original image
        cv2.putText(image, plate_text, (x1, y1 - 10), font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)

# Save or display the result
cv2.imwrite('output_image.jpg', image)
# cv2.imshow('Detected Plates', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
