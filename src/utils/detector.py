"""Module for object detection using YOLOv8."""

from ultralytics import YOLO
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)

class ObjectDetector:
    """Wrapper for YOLOv8 object detection model."""

    def __init__(self, model_name='yolov8n.pt'):
        """
        Initialise the detector.
        
        Args:
            model_name (str): Name of the pre-trained YOLOv8 model (e.g., 'yolov8n.pt', 'yolov8s.pt').
        """
        logger.info("Initialising YOLOv8 detector with model: %s", model_name)
        self.model = YOLO(model_name)
        
        # COCO class IDs: 0: person, 1: bicycle, 3: motorcycle
        self.target_classes = [0, 1, 3] 

    def detect(self, image, conf=0.20):
        """
        Perform detection on a single image.
        
        Args:
            image (numpy.ndarray): RGB image as a NumPy array.
            conf (float): Confidence threshold for tracking/detection.
            
        Returns:
            ultralytics.engine.results.Results: Detection results.
        """
        # Usiamo track invece di predict per avere gli ID degli oggetti (necessari per Kalman)
        results = self.model.track(image, classes=self.target_classes, persist=True, verbose=False, conf=conf)
        return results[0]

    def get_detected_objects(self, results):
        """
        Extract bounding boxes, classes, and confidences from results.
        
        Args:
            results: Results object from self.detect()
            
        Returns:
            list: List of dictionaries containing detection info.
        """
        objects = []
        for box in results.boxes:
            objects.append({
                'bbox': box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                'confidence': float(box.conf[0]),
                'class_id': int(box.cls[0]),
                'label': self.model.names[int(box.cls[0])]
            })
        return objects
