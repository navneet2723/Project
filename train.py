import os
import json
import logging
from datetime import datetime
from scripts.data_loader import DataLoader
from scripts.hyperparameter_tuning import EnhancedHyperparameterTuner
from config.config import *

def setup_logging():
    """
    Enhanced logging setup with formatting
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(LOG_PATH, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    return log_dir

def main():
    try:
        # Setup logging and directories
        log_dir = setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("Starting training process")
        
        # Ensure directories exist
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        data_loader = DataLoader()
        X_train, X_val, Y_train, Y_val = data_loader.load_data()
        logger.info(f"Data loaded - Training: {X_train.shape}, Validation: {X_val.shape}")
        
        # Initialize and run hyperparameter tuning
        logger.info("Starting hyperparameter optimization")
        tuner = EnhancedHyperparameterTuner(X_train, Y_train, X_val, Y_val)
        best_params, best_model = tuner.run_optimization(n_trials=HYPERPARAMETER_TRIALS)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_dir = os.path.join(MODEL_SAVE_PATH, timestamp)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model and parameters
        model_path = os.path.join(model_dir, 'best_model.h5')
        params_path = os.path.join(model_dir, 'best_params.json')
        
        best_model.save(model_path)
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        
        # Generate and save optimization plots
        logger.info("Generating optimization visualizations")
        tuner.save_optimization_plots()
        
        logger.info(f"Training complete. Model saved to {model_path}")
        logger.info(f"Best parameters: {best_params}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()