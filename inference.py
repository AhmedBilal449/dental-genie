# inference.py
import torch
import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import random

class DentalInference:
    def __init__(self):
        self.models = {
            "disease": "runs/detect/quadrant_enumeration_disease_train/weights/best.pt",
            "enumeration": "runs/detect/quadrant_enumeration_train/weights/best.pt",
            "quadrant": "runs/detect/quadrant_train/weights/best.pt",
        }
        self.yolo_models = {name: YOLO(path) for name, path in self.models.items()}

    def calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter_area / (box1_area + box2_area - inter_area)

    def get_random_color(self):
        return tuple(random.randint(0, 255) for _ in range(3))

    def predict(self, image_np):
        detections = []
        img = image_np.copy()
        h, w = img.shape[:2]

        temp_path = "temp.jpg"
        cv2.imwrite(temp_path, img)

        disease_results = self.yolo_models["disease"](temp_path)[0]
        quadrant_results = self.yolo_models["quadrant"](temp_path)[0]
        enumeration_results = self.yolo_models["enumeration"](temp_path)[0]

        quadrant_boxes = [box.xyxy[0].tolist() for box in quadrant_results.boxes]
        quadrant_labels = [self.yolo_models["quadrant"].names[int(cls)] for cls in quadrant_results.boxes.cls]

        enumeration_boxes = [box.xyxy[0].tolist() for box in enumeration_results.boxes]
        enumeration_labels = [self.yolo_models["enumeration"].names[int(cls)] for cls in enumeration_results.boxes.cls]

        for box, cls in zip(disease_results.boxes.xyxy, disease_results.boxes.cls):
            x1, y1, x2, y2 = map(int, box[:4])
            disease_label = self.yolo_models["disease"].names[int(cls)]

            best_quadrant_label = "?"
            best_enumeration_label = "?"
            max_iou_quadrant = 0
            max_iou_enumeration = 0

            for q_box, q_label in zip(quadrant_boxes, quadrant_labels):
                iou = self.calculate_iou([x1, y1, x2, y2], q_box)
                if iou > max_iou_quadrant:
                    max_iou_quadrant = iou
                    best_quadrant_label = q_label

            for e_box, e_label in zip(enumeration_boxes, enumeration_labels):
                iou = self.calculate_iou([x1, y1, x2, y2], e_box)
                if iou > max_iou_enumeration:
                    max_iou_enumeration = iou
                    best_enumeration_label = e_label

            # Clean label format
            quadrant_clean = best_quadrant_label.replace("Quadrant_", "")
            tooth_clean = best_enumeration_label.replace("Tooth_", "")
            label = f"Quadrant {quadrant_clean} | Tooth {tooth_clean} | {disease_label}"

            detections.append((x1, y1, x2, y2, label))

        box_colors = [self.get_random_color() for _ in detections]
        result_labels = []

        for (x1, y1, x2, y2, label), color in zip(detections, box_colors):
            result_labels.append(label)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return img, "\n".join(result_labels)
