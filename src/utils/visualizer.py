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

    def update(self, image, results=None):
        """
        Update the display with a new frame and optional detection results.
        
        Args:
            image (numpy.ndarray): RGB image array.
            results: Results object from YOLOv8 detector.
        """
        # 1. Convert image to pygame surface
        # YOLO results.plot() returns BGR by default if used, 
        # but here we'll draw manually or use the plotted frame
        if results is not None:
            # results.plot() returns a BGR image with boxes drawn
            plotted_frame = results.plot()
            # BGR to RGB
            rgb_frame = plotted_frame[:, :, ::-1]
        else:
            rgb_frame = image

        # 2. Reshape/Transpose if necessary for pygame (W, H, 3)
        surface = pygame.surfarray.make_surface(rgb_frame.swapaxes(0, 1))
        
        # 3. Blit and update
        self.display.blit(surface, (0, 0))
        pygame.display.flip()
        
        # 4. Handle pygame events (to prevent window freezing)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def close(self):
        """Close the pygame window."""
        pygame.quit()
