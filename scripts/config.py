import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
VIDEO_DIR = os.path.join(DATA_DIR, 'videos')
TEXT_DIR = os.path.join(DATA_DIR, 'text')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'fid_image_model.keras')

"""
Hyperparameters
"""
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
