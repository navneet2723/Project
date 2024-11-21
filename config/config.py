import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models', 'saved_models')
LOG_PATH = os.path.join(PROJECT_ROOT, 'logs', 'training_logs')
IMAGE_DIR = os.path.join(DATA_DIR, 'test')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Model configuration
IMAGE_SIZE = (128, 128) 
EPOCHS = 50
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Training configuration
HYPERPARAMETER_TRIALS = 50  # New: Number of trials for optimization
EARLY_STOPPING_PATIENCE = 5
LR_REDUCTION_PATIENCE = 3
MIN_LR = 1e-6
LR_REDUCTION_FACTOR = 0.2

# New: Hyperparameter search space configuration
HP_CONFIG = {
    'conv_filters': [32, 64, 128, 256],
    'kernel_sizes': [3, 5],
    'dense_units': [128, 256, 512],
    'dropout_rates': [0.1, 0.2, 0.3, 0.4, 0.5],
    'learning_rates': [1e-5, 1e-4, 1e-3, 1e-2],
    'batch_sizes': [16, 32, 64],
    'optimizers': ['adam', 'rmsprop']
}