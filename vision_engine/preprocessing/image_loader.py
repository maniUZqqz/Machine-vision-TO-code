import cv2
import numpy as np


class ImagePreprocessor:
    """
    Step 1: Load, validate, and normalize the input image.
    - Ensures RGB format
    - Optional denoising
    - Stores dimensions in context
    """

    def __init__(self, max_dimension: int = 1600, denoise: bool = True):
        self.max_dimension = max_dimension
        self.denoise = denoise

    def process(self, image: np.ndarray, context) -> object:
        img = image.copy()

        # Ensure RGB (handle RGBA)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # Resize if too large (preserve aspect ratio)
        h, w = img.shape[:2]
        # Store original dimensions before any resize
        context.original_height = h
        context.original_width = w
        if max(h, w) > self.max_dimension:
            scale = self.max_dimension / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Optional denoising
        if self.denoise:
            img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)

        context.working_image = img
        context.height, context.width = img.shape[:2]
        return context
