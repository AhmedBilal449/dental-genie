import os
import cv2
import random
import torch
from ultralytics import YOLO


class DentalInference:
    def __init__(self):
        self.models = {
            "disease": "runs/detect/quadrant_enumeration_disease_train_m/weights/best.pt",
            "enumeration": "runs/detect/quadrant_enumeration_train_m/weights/best.pt",
            "quadrant": "runs/detect/quadrant_train_m/weights/best.pt",
        }
        self.yolo_models = {name: YOLO(path) for name, path in self.models.items()}

    def calculate_iou_tensor(self, boxes1, boxes2):
        """
        boxes1: (N, 4), boxes2: (M, 4) => returns (N, M) IoU tensor
        """
        x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
        y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])

        inter_area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        union_area = area1[:, None] + area2 - inter_area
        return inter_area / union_area.clamp(min=1e-6)

    def get_random_color(self):
        return tuple(random.randint(0, 255) for _ in range(3))

    def predict(self, image_np):
        detection_data = []
        img = image_np.copy()

        temp_path = "temp.jpg"
        cv2.imwrite(temp_path, img)

        # Run inference
        disease_results = self.yolo_models["disease"](temp_path)[0]
        quadrant_results = self.yolo_models["quadrant"](temp_path)[0]
        enumeration_results = self.yolo_models["enumeration"](temp_path)[0]

        # Extract boxes and classes
        disease_boxes = disease_results.boxes.xyxy.cpu()
        disease_classes = disease_results.boxes.cls.cpu()

        quadrant_boxes = quadrant_results.boxes.xyxy.cpu()
        quadrant_classes = quadrant_results.boxes.cls.cpu()

        enumeration_boxes = enumeration_results.boxes.xyxy.cpu()
        enumeration_classes = enumeration_results.boxes.cls.cpu()

        # Compute IoUs
        iou_quadrants = self.calculate_iou_tensor(disease_boxes, quadrant_boxes)
        best_q_indices = torch.argmax(iou_quadrants, dim=1)

        iou_enums = self.calculate_iou_tensor(disease_boxes, enumeration_boxes)
        best_e_indices = torch.argmax(iou_enums, dim=1)

        # Get label names
        quadrant_names = self.yolo_models["quadrant"].names
        enumeration_names = self.yolo_models["enumeration"].names
        disease_names = self.yolo_models["disease"].names

        for i in range(len(disease_boxes)):
            x1, y1, x2, y2 = map(int, disease_boxes[i].tolist())
            disease_label = disease_names[int(disease_classes[i])]

            best_quadrant_label = quadrant_names[int(quadrant_classes[best_q_indices[i]])]
            best_enum_label = enumeration_names[int(enumeration_classes[best_e_indices[i]])]

            quadrant_clean = best_quadrant_label.replace("Quadrant_", "")
            tooth_clean = best_enum_label.replace("Tooth_", "")
            label = f"Quadrant {quadrant_clean} | Tooth {tooth_clean} | {disease_label}"

            detection_data.append({
                "quadrant": quadrant_clean,
                "tooth": tooth_clean,
                "disease": disease_label,
                "bbox": (x1, y1, x2, y2)
            })

            color = self.get_random_color()
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return img, detection_data
