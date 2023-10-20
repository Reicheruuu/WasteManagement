import cv2
import torch
import os

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
confidence_threshold = 0.5

# Load YOLOv5 models
model1 = torch.hub.load('ultralytics/yolov5', 'custom', path='model4/weights/best.pt')
model2 = torch.hub.load('ultralytics/yolov5', 'yolov5s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model1 = model1.to(device).eval()
model2 = model2.to(device).eval()

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

# Specify the image directory
image_directory = 'Waste-Management-3/test/images'
image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
current_image_index = 0  # Start with the first image

while True:
    image_file = image_files[current_image_index]
    image_path = os.path.join(image_directory, image_file)
    frame = cv2.imread(image_path)

    # Get the size of the window
    window_size = cv2.getWindowImageRect('Object Detection')

    # Resize the frame based on the window size
    width, height = window_size[2], window_size[3]
    frame = cv2.resize(frame, (width, height))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
    key = cv2.waitKey(0)

    if key == ord('q'):
        # Quit the loop if 'q' key is pressed
        break
    elif key == ord('d'):
        # Move to the next image if 'n' key is pressed
        current_image_index = (current_image_index + 1) % len(image_files)
    elif key == ord('a'):
        # Move to the previous image if 'p' key is pressed
        current_image_index = (current_image_index - 1) % len(image_files)

# Release the OpenCV window
cv2.destroyAllWindows()
