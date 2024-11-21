from sklearn.model_selection import train_test_split
from config.config import IMAGE_DIR
from keras.utils import to_categorical
import numpy as np
import logging
import os

from .image_processing import ImageProcessor

class DataLoader:
    def __init__(self, image_size=None):
        self.image_size = image_size
        self.image_processor = ImageProcessor()
        self._setup_logging()
    
    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
    
    def prepare_image(self, image_path):
        """
        Process single image with error handling and logging
        """
        try:
            ela_image = self.image_processor.convert_to_ela_image(image_path)
            processed_image = np.array(ela_image.resize(self.image_size))
            return processed_image / 255.0
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            return None

    def load_data(self):
        """
        Load and process all images with progress tracking
        """
        X, Y = [], []
        
        # Process authentic images
        auth_path = os.path.join(IMAGE_DIR, 'Au')
        self.logger.info("Processing authentic images...")
        for filename in os.listdir(auth_path):
            if filename.lower().endswith(('jpg', 'jpeg', 'png')):
                image = self.prepare_image(os.path.join(auth_path, filename))
                if image is not None:
                    X.append(image)
                    Y.append(1)
        
        # Process tampered images
        tamp_path = os.path.join(IMAGE_DIR, 'Tp')
        self.logger.info("Processing tampered images...")
        for filename in os.listdir(tamp_path):
            if filename.lower().endswith(('jpg', 'jpeg', 'png')):
                image = self.prepare_image(os.path.join(tamp_path, filename))
                if image is not None:
                    X.append(image)
                    Y.append(0)
        
        X = np.array(X)
        Y = to_categorical(np.array(Y), 2)
        
        self.logger.info(f"Processed {len(X)} images successfully")
        return train_test_split(X, Y, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED)