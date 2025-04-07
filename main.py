import gradio as gr
import cv2
import numpy as np
import os
import random
from pathlib import Path
from ultralytics import YOLO

# Paths to trained models
models = {
    "disease": "runs/detect/quadrant_enumeration_disease_train/weights/best.pt",
    "enumeration": "runs/detect/quadrant_enumeration_train/weights/best.pt",
    "quadrant": "runs/detect/quadrant_train/weights/best.pt",
}

# Load all models
yolo_models = {name: YOLO(path) for name, path in models.items()}

# Function to generate distinct colors
def get_random_color():
    return tuple(random.randint(0, 255) for _ in range(3))

# Function to calculate IoU
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
    return iou

# Function for model inference and labeling
def yolo_predict(image):
    detections = []
    detection_summary = []

    # Save temporary input image
    temp_input_path = "temp_input.png"
    cv2.imwrite(temp_input_path, image)

    # Run inference on all models
    disease_results = yolo_models["disease"](temp_input_path)[0]
    quadrant_results = yolo_models["quadrant"](temp_input_path)[0]
    enumeration_results = yolo_models["enumeration"](temp_input_path)[0]

    # Extract quadrant & enumeration boxes
    quadrant_boxes = [box.xyxy[0].tolist() for box in quadrant_results.boxes]
    quadrant_labels = [yolo_models["quadrant"].names[int(cls)] for cls in quadrant_results.boxes.cls]

    enumeration_boxes = [box.xyxy[0].tolist() for box in enumeration_results.boxes]
    enumeration_labels = [yolo_models["enumeration"].names[int(cls)] for cls in enumeration_results.boxes.cls]

    for box, cls in zip(disease_results.boxes.xyxy, disease_results.boxes.cls):
        x1, y1, x2, y2 = map(int, box[:4])
        disease_label = yolo_models["disease"].names[int(cls)]

        best_q_label, best_e_label = "?", "?"
        max_iou_q, max_iou_e = 0, 0

        for q_box, q_label in zip(quadrant_boxes, quadrant_labels):
            iou = calculate_iou([x1, y1, x2, y2], q_box)
            if iou > max_iou_q:
                max_iou_q = iou
                best_q_label = q_label

        for e_box, e_label in zip(enumeration_boxes, enumeration_labels):
            iou = calculate_iou([x1, y1, x2, y2], e_box)
            if iou > max_iou_e:
                max_iou_e = iou
                best_e_label = e_label

        label = f"Q: {best_q_label} N: {best_e_label} D: {disease_label}"
        detections.append((x1, y1, x2, y2, label))
        detection_summary.append(f"Quadrant: {best_q_label} | Tooth: {best_e_label} | Disease: {disease_label}")

    output_image = image.copy()
    random.shuffle(detections)
    box_colors = [get_random_color() for _ in detections]

    for (x1, y1, x2, y2, label), color in zip(detections, box_colors):
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return output_image, detection_summary

# Gradio processing function
def process_xray(patient_name, xray_image, comments=""):
    output_image, summary = yolo_predict(xray_image)

    if not summary:
        result_text = "No detections found."
    else:
        result_text = "\n".join(summary)

    return output_image, result_text

# Gradio interface
interface = gr.Interface(
    fn=process_xray,
    inputs=[
        gr.Textbox(label="Patient Name"),
        gr.Image(label="Upload X-ray", type="numpy"),
        gr.Textbox(label="Add Comments (Optional)", lines=2)
    ],
    outputs=[
        gr.Image(label="Labeled X-ray"),
        gr.Textbox(label="Detection Summary")
    ],
    title="X-ray Disease Detection with YOLO",
    description="Upload a dental X-ray to detect quadrant, tooth number, and dental disease using trained YOLO models."
)

# Launch the interface
interface.launch()
