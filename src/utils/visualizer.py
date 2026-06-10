import pygame
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)

class Visualizer:
    """Component to handle real-time visualization of CARLA simulation and detections."""

    def __init__(self, width=800, height=600, title="SmartVehicular Detection"):
        """
        Initialise the pygame display.
        
        Args:
            width (int): Window width.
            height (int): Window height.
            title (str): Window title.
        """
        pygame.init()
        self.display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont('Arial', 18)
        logger.info("Visualizer initialised: %dx%d", width, height)

    def update(self, image, results=None, kalman_predictions=None):
        """
        Update the display with a new frame and optional detection results.
        
        Args:
            image (numpy.ndarray): RGB image array.
            results: Results object from YOLOv8 detector.
            kalman_predictions (dict): Dictionary of predicted boxes.
        """
        surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
        
        color_yolo = (0, 255, 0)
        color_kalman = (255, 165, 0)

        if results is not None:
            for i, box in enumerate(results.boxes):
                coords = box.xyxy[0].tolist()
                rect = pygame.Rect(coords[0], coords[1], coords[2]-coords[0], coords[3]-coords[1])
                pygame.draw.rect(surface, color_yolo, rect, 2)
                
                label = f"{results.names[int(box.cls[0])]} {float(box.conf[0]):.2f}"
                text_surface = self.font.render(label, True, color_yolo)
                surface.blit(text_surface, (coords[0], coords[1] - 22))
                
                if kalman_predictions and results.boxes.id is not None:
                    obj_id = int(results.boxes.id[i])
                    if obj_id in kalman_predictions:
                        pred_data = kalman_predictions[obj_id]
                        if isinstance(pred_data, dict):
                            p_coords = pred_data["box"]
                            trajectory = pred_data.get("trajectory", [])
                            time_horizon = pred_data.get("time_horizon", 1.0)
                        else:
                            p_coords = pred_data
                            trajectory = []
                            time_horizon = 1.0
                        
                        # Draw the prediction box
                        p_rect = pygame.Rect(p_coords[0], p_coords[1], p_coords[2]-p_coords[0], p_coords[3]-p_coords[1])
                        pygame.draw.rect(surface, color_kalman, p_rect, 2)
                        
                        p_text = self.font.render(f"Kalman Prediction (+{time_horizon:.1f}s)", True, color_kalman)
                        surface.blit(p_text, (p_coords[0], p_coords[3] + 5))
                        
                        # Draw prediction trajectory points and connect them
                        if len(trajectory) > 0:
                            # Draw path lines
                            path_points = [(int(cx), int(cy)) for cx, cy in trajectory]
                            # Add the current center of the YOLO detection to start the trajectory line
                            yolo_cx = int((coords[0] + coords[2]) / 2.0)
                            yolo_cy = int((coords[1] + coords[3]) / 2.0)
                            path_points.insert(0, (yolo_cx, yolo_cy))
                            
                            if len(path_points) > 1:
                                pygame.draw.lines(surface, color_kalman, False, path_points, 1)
                            
                            # Draw dot at each predicted point
                            for pt in trajectory:
                                pygame.draw.circle(surface, color_kalman, (int(pt[0]), int(pt[1])), 3)

        # 3. Blit e update
        self.display.blit(surface, (0, 0))
        pygame.display.flip()
        
        # 4. Handle pygame events (to prevent window freezing)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True

    def close(self):
        """Close the pygame window."""
        pygame.quit()
