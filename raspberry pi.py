import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import resize


# Create categories for model2 based on the specific classes
model2_category_map = {
    'Leaves': 'Biodegradable',
    'crumpled paper': 'Biodegradable',
    'tissue': 'Biodegradable',
    'banana': 'Biodegradable',
    'apple': 'Biodegradable',
    'sandwich': 'Biodegradable',
    'orange': 'Biodegradable',
    'carrot': 'Biodegradable',
    'broccoli': 'Biodegradable',
    'hot dog': 'Biodegradable',
    'pizza': 'Biodegradable',
    'donut': 'Biodegradable',
    'cake': 'Biodegradable',
    'junk food wrapper' : 'Non-Biodegradable',
    'styrofoam': 'Non-Biodegradable',
    'diaper': 'Non-Biodegradable',
    'napkin': 'Non-Biodegradable',
    'straw': 'Non-Biodegradable',
    'stick': 'Non-Biodegradable',
    'CARDBOARD': 'Recyclable',
    'glass bottle': 'Recyclable',
    'plastic bottle': 'Recyclable',
    'plastic cup': 'Recyclable',
    'can': 'Recyclable',
    'spoon': 'Recyclable',
    'fork': 'Recyclable',
    'bottle': 'Recyclable',
    'cup': 'Recyclable'
}

# Set the minimum confidence threshold (adjust this value as needed)
confidence_threshold = 0.8

# Load YOLOv5 models
model1 = torch.hub.load('ultralytics/yolov5', 'custom', path='model4/weights/best.pt')
model2 = torch.hub.load('ultralytics/yolov5', 'yolov5s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model1 = model1.to(device).eval()
model2 = model2.to(device).eval()

# Define frame skipping parameters
frame_counter = 0
frame_skip = 1  # Process every frame (1), every other frame (2), etc.

# Define classes to exclude from detection in the pretrained model
classes_to_exclude = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'wine glass',
    'knife', 'bowl', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors',  'teddy bear', 'hair drier', 'toothbrush'
]


# Create a resizable window
cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)

# Open the webcam (adjust the camera index if needed)
cap = cv2.VideoCapture(2)  # 0 represents the default camera, you can change it if you have multiple cameras.

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    frame_counter += 1

    if not ret:
        break

    if frame_counter % frame_skip != 0:
        continue  # Skip frames based on the frame_skip value

    # Resize the frame (similar to the previous code)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = frame.shape
    max_size = 320  # Adjust the resolution (e.g., 320x240)
    if width > height:
        new_width = max_size
        new_height = int(height / (width / max_size))
    else:
        new_height = max_size
        new_width = int(width / (height / max_size))
    frame = resize(Image.fromarray(frame), (new_height, new_width))
    frame = np.array(frame)
    # Perform inference with the first YOLOv5 model
    results_model1 = model1(frame)
    detections1 = results_model1.xyxy[0].cpu().numpy()
    confidence_model1 = detections1[:, 4]

    # Filter out objects with excluded classes
    filtered_detections1 = [d for d in detections1 if model1.names[int(d[5])] not in classes_to_exclude]

    # Perform inference with the second YOLOv5 model
    results_model2 = model2(frame)
    detections2 = results_model2.xyxy[0].cpu().numpy()
    confidence_model2 = detections2[:, 4]

    # Filter out objects with excluded classes
    filtered_detections2 = [d for d in detections2 if model2.names[int(d[5])] not in classes_to_exclude]

    # Loop through detected objects and their confidence scores for model 1
    for i in range(len(filtered_detections1)):
        class_index = int(filtered_detections1[i][5])
        class_name = model1.names[class_index]
        category = model2_category_map.get(class_name, 'Unknown')

        x_min, y_min, x_max, y_max, confidence = filtered_detections1[i][:5]

        if confidence >= confidence_threshold:
            color = (0, 255, 0) if category == 'Biodegradable' else (255, 255, 0) if category == 'Non-Biodegradable' else (255, 0, 0)
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
            label = f'{class_name} ({category}, {confidence:.2f})'
            cv2.putText(frame, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Loop through detected objects and their confidence scores for model 2
    for i in range(len(filtered_detections2)):
        class_index = int(filtered_detections2[i][5])
        class_name = model2.names[class_index]
        category = model2_category_map.get(class_name, 'Unknown')

        x_min, y_min, x_max, y_max, confidence = filtered_detections2[i][:5]

        if confidence >= confidence_threshold:
            color = (0, 255, 0) if category == 'Biodegradable' else (255, 255, 0) if category == 'Non-Biodegradable' else (255, 0, 0)
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
            label = f'{class_name} ({category}, {confidence:.2f})'
            cv2.putText(frame, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the image with bounding boxes
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Object Detection', frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()