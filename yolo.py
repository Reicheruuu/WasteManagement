# Import necessary libraries
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import resize

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Check and set the device to GPU if available, else use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Open a video capture object for camera 0 (you can change the camera index if needed)
cap = cv2.VideoCapture(2)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = frame.shape
    max_size = 600
    if width > height:
        new_width = max_size
        new_height = int(height / (width / max_size))
    else:
        new_height = max_size
        new_width = int(width / (height / max_size))
    frame = resize(Image.fromarray(frame), (new_width, new_height))  # Fixed a mistake here

    # Convert frame to numpy array
    frame = np.array(frame)

    # Perform object detection
    results = model(frame)

    # Perform object detection
    results = model(frame)

    # Extract results
    pred = results.pred[0]

    # Get bounding boxes, confidence scores, and class labels
    boxes = pred[:, :4]  # Bounding boxes
    scores = pred[:, 4]  # Confidence scores
    class_ids = pred[:, 5]  # Class labels

    # You can set a threshold to filter out low-confidence detections
    threshold = 0.5
    mask = scores > threshold
    boxes = boxes[mask]
    class_ids = class_ids[mask]

    # Loop through the detected objects
    for box, class_id in zip(boxes, class_ids):
        x1, y1, x2, y2 = map(int, box)
        class_name = model.names[int(class_id)]

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle
        cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Label

    # Display the frame with detections
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()


