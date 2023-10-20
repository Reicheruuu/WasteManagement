import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import resize
import os

# Define class categories
biodegradable_classes = ['Leaves', 'crumpled paper', 'tissue']
non_biodegradable_classes = ['junk food wrapper', 'styrofoam', 'diaper', 'napkin']
recyclable_classes = ['CARDBOARD', 'glass', 'can']

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='main model/weights/best.pt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

model.eval()

# Specify the image directory
image_directory = 'Waste-Management-3/test/images'
image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
cap = cv2.VideoCapture(0)

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

    # Perform inference with YOLOv5 model
    results_model = model(frame)

    # Get detected objects and their positions
    detections = results_model.xyxy[0].cpu().numpy()
    confidence_model = results_model.xyxy[0].cpu().numpy()[:, 4]

    # Loop through detected objects and their confidence scores
    for i in range(len(detections)):
        if confidence_model[i] > 0.5:  # You can adjust the confidence threshold as needed
            # Get class index
            class_index = int(detections[i][5])
            class_name = model.names[int(class_index)]  # Get the class name from the model

            # Determine the category
            if class_name in biodegradable_classes:
                category = 'Biodegradable'
            elif class_name in non_biodegradable_classes:
                category = 'Non-Biodegradable'
            elif class_name in recyclable_classes:
                category = 'Recyclable'
            else:
                category = 'Unknown'

            # Extract bounding box coordinates
            x_min, y_min, x_max, y_max, confidence = detections[i][:5]

            # Draw bounding box on the image
            if category == 'Biodegradable':
                color = (0, 255, 0)  # Green for biodegradable
            elif category == 'Non-Biodegradable':
                color = (255, 255, 0)  # Yellow for non-biodegradable
            else:
                color = (255, 0, 0)  # red for recyclable (can be adjusted)

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
