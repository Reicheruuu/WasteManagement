import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import resize
import os

# Define class categories for model1
biodegradable_classes = ['Leaves', 'crumpled paper', 'kitchentissue', 'banana', 'apple', 'sandwich', 'orange', 'carrot', 'broccoli']
non_biodegradable_classes = ['junk food wrapper', 'styrofoam', 'diaper', 'napkin', 'toothbrush']
recyclable_classes = ['CARDBOARD', 'glass', 'can', 'spoon', 'fork', 'bottle', 'cup']

# Define the specific classes you want to detect with model2
model2_classes_to_detect = ['banana', 'apple', 'sandwich', 'orange', 'carrot', 'broccoli', 'toothbrush', 'spoon', 'fork', 'bottle', 'cup']

# Create categories for model2 based on the specific classes
model2_category_map = {
    'banana': 'Biodegradable',
    'apple': 'Biodegradable',
    'sandwich': 'Biodegradable',
    'orange': 'Biodegradable',
    'broccoli': 'Biodegradable',
    'toothbrush': 'Non-Biodegradable',
    'spoon': 'Recyclable',
    'fork': 'Recyclable',
    'bottle': 'Recyclable',
    'cup': 'Recyclable'
}

# Set the minimum confidence threshold (adjust this value as needed)
confidence_threshold = 0.5
# Load YOLOv5 models
model1 = torch.hub.load('ultralytics/yolov5', 'custom', path='model/weights/best.pt')
model2 = torch.hub.load('ultralytics/yolov5', 'yolov5s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model1 = model1.to(device)
model2 = model2.to(device)
model1.eval()
model2.eval()

# Define classes to exclude from detection in the pretrained model
classes_to_exclude = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'wine glass',
    'knife',
    'bowl',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier'
]

# Specify the image directory
image_directory = 'Waste-Management-3/test/images'
image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    frame = cv2.imread(image_path)

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
    frame = resize(Image.fromarray(frame), (new_height, new_width))

    # Convert frame to numpy array
    frame = np.array(frame)

    # Perform inference with the first YOLOv5 model
    results_model1 = model1(frame)

    # Get detected objects and their positions for model 1
    detections1 = results_model1.xyxy[0].cpu().numpy()
    confidence_model1 = results_model1.xyxy[0].cpu().numpy()[:, 4]

    # Filter out objects with excluded classes
    filtered_detections1 = [d for d in detections1 if model1.names[int(d[5])] not in classes_to_exclude]

    # Perform inference with the second YOLOv5 model
    results_model2 = model2(frame)

    # Get detected objects and their positions for model 2
    detections2 = results_model2.xyxy[0].cpu().numpy()
    confidence_model2 = results_model2.xyxy[0].cpu().numpy()[:, 4]

    # Filter out objects with excluded classes
    filtered_detections2 = [d for d in detections2 if model2.names[int(d[5])] not in classes_to_exclude]

    # Loop through detected objects and their confidence scores for model 1
    for i in range(len(filtered_detections1)):

        # Get class index
        class_index = int(filtered_detections1[i][5])
        class_name = model1.names[class_index]  # Get the class name from the model

        # Determine the category for model1
        if class_name in biodegradable_classes:
            category = 'Biodegradable'
        elif class_name in non_biodegradable_classes:
            category = 'Non-Biodegradable'
        elif class_name in recyclable_classes:
            category = 'Recyclable'
        else:
            category = 'Unknown'

        # Extract bounding box coordinates
        x_min, y_min, x_max, y_max, confidence = filtered_detections1[i][:5]

        # Check if the confidence is above the threshold
        if confidence >= confidence_threshold:
            # Draw bounding box on the image
            if category == 'Biodegradable':
                color = (0, 255, 0)  # Green for biodegradable
            elif category == 'Non-Biodegradable':
                color = (255, 255, 0)  # Yellow for non-biodegradable
            else:
                color = (255, 0, 0)  # Red for recyclable (can be adjusted))

            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)

        # Put text with class label, category, and confidence score on the image
            label = f'{class_name} ({category}, {confidence:.2f})'
            cv2.putText(frame, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Loop through detected objects and their confidence scores for model 2
    for i in range(len(filtered_detections2)):
        # Get class index
        class_index = int(filtered_detections2[i][5])
        class_name = model2.names[class_index]  # Get the class name from the model

        # Determine the category for model 1
        if class_name in model2_classes_to_detect:
            category = model2_category_map[class_name]
        else:
            category = 'Unknown'

        # Extract bounding box coordinates
        x_min, y_min, x_max, y_max, confidence = filtered_detections2[i][:5]

        # Check if the confidence is above the threshold
        if confidence >= confidence_threshold:
            # Draw bounding box on the image
            if category == 'Biodegradable':
                color = (0, 255, 0)  # Green for biodegradable
            elif category == 'Non-Biodegradable':
                color = (255, 255, 0)  # Yellow for non-biodegradable
            else:
                color = (255, 0, 0)  # Red for recyclable (can be adjusted)

            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)

            # Put text with class label, category, and confidence score on the image
            label = f'{class_name} ({category}, {confidence:.2f})'
            cv2.putText(frame, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the image with bounding boxes
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Object Detection', frame)
    cv2.waitKey(0)

# Release the OpenCV window
cv2.destroyAllWindows()
