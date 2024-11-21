import os
import json
import logging
from datetime import datetime
import optuna
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from config.config import *

class EnhancedHyperparameterTuner:
    def __init__(self, X_train, Y_train, X_val, Y_val):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.best_accuracy = 0
        self.best_model = None
        self.study = None
        
        # Create timestamp for this tuning session
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup directories
        self.model_dir = os.path.join(MODEL_SAVE_PATH, self.timestamp)
        self.log_dir = os.path.join(LOG_PATH, self.timestamp)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
    def create_model(self, trial):
        """
        Creates model architecture based on trial parameters
        """
        model = Sequential()
        
        # Number of convolutional blocks
        n_conv_blocks = trial.suggest_int('n_conv_blocks', 2, 4)
        
        # Input layer
        model.add(Conv2D(
            trial.suggest_categorical('conv0_filters', HP_CONFIG['conv_filters']),
            trial.suggest_categorical('conv0_kernel', HP_CONFIG['kernel_sizes']),
            padding='same',
            activation='relu',
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
        ))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(trial.suggest_float('conv0_dropout', 0.1, 0.4)))
        
        # Additional convolutional blocks
        for i in range(1, n_conv_blocks):
            model.add(Conv2D(
                trial.suggest_categorical(f'conv{i}_filters', HP_CONFIG['conv_filters']),
                trial.suggest_categorical(f'conv{i}_kernel', HP_CONFIG['kernel_sizes']),
                padding='same',
                activation='relu'
            ))
            model.add(BatchNormalization())
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Dropout(trial.suggest_float(f'conv{i}_dropout', 0.1, 0.4)))
        
        # Dense layers
        model.add(Flatten())
        
        # Number of dense layers
        n_dense_layers = trial.suggest_int('n_dense_layers', 1, 2)
        
        for i in range(n_dense_layers):
            model.add(Dense(
                trial.suggest_categorical(f'dense{i}_units', HP_CONFIG['dense_units']),
                activation='relu'
            ))
            model.add(BatchNormalization())
            model.add(Dropout(trial.suggest_float(f'dense{i}_dropout', 0.2, 0.5)))
        
        # Output layer
        model.add(Dense(2, activation='softmax'))
        
        # Compile model
        optimizer_name = trial.suggest_categorical('optimizer', HP_CONFIG['optimizers'])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        
        if optimizer_name == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        else:
            optimizer = RMSprop(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_callbacks(self, trial_number):
        """
        Creates training callbacks with proper paths
        """
        model_checkpoint_path = os.path.join(
            self.model_dir, 
            f'trial_{trial_number}_checkpoint.h5'
        )
        
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=LR_REDUCTION_FACTOR,
                patience=LR_REDUCTION_PATIENCE,
                min_lr=MIN_LR,
                verbose=1
            ),
            TensorBoard(
                log_dir=os.path.join(self.log_dir, f'trial_{trial_number}'),
                histogram_freq=1
            )
        ]
        return callbacks
    
    def objective(self, trial):
        """
        Objective function for optimization
        """
        self.logger.info(f"Starting trial {trial.number}")
        
        # Create and train model
        model = self.create_model(trial)
        callbacks = self.create_callbacks(trial.number)
        
        # Train model
        history = model.fit(
            self.X_train,
            self.Y_train,
            batch_size=trial.suggest_categorical('batch_size', HP_CONFIG['batch_sizes']),
            epochs=EPOCHS,
            validation_data=(self.X_val, self.Y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Get best validation accuracy
        val_accuracy = max(history.history['val_accuracy'])
        
        # Save if best model so far
        if val_accuracy > self.best_accuracy:
            self.best_accuracy = val_accuracy
            self.best_model = model
            
            # Save model
            self._save_best_model(trial, val_accuracy)
            
        self.logger.info(f"Trial {trial.number} finished with accuracy: {val_accuracy}")
        return val_accuracy
    
    def _save_best_model(self, trial, accuracy):
        """
        Saves the best model and its parameters
        """
        # Save model
        best_model_path = os.path.join(self.model_dir, 'best_model.h5')
        self.best_model.save(best_model_path)
        
        # Save metadata
        metadata = {
            'trial_number': trial.number,
            'accuracy': float(accuracy),
            'parameters': trial.params,
            'timestamp': self.timestamp,
            'image_size': IMAGE_SIZE,
            'model_architecture': self.best_model.get_config(),
            'training_history': {
                'best_accuracy': float(accuracy),
                'epochs_trained': len(self.best_model.history.history['accuracy'])
                if hasattr(self.best_model, 'history') else None
            }
        }
        
        metadata_path = os.path.join(self.model_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        self.logger.info(f"Saved best model (accuracy: {accuracy}) to {best_model_path}")
    
    def run_optimization(self, n_trials=HYPERPARAMETER_TRIALS):
        """
        Runs the optimization process
        """
        self.logger.info(f"Starting optimization with {n_trials} trials")
        
        # Create study
        self.study = optuna.create_study(direction='maximize')
        
        # Run optimization
        self.study.optimize(self.objective, n_trials=n_trials)
        
        # Save final results
        self._save_study_results()
        
        return self.study.best_params, self.best_model
    
    def _save_study_results(self):
        """
        Saves study results and statistics
        """
        study_stats = {
            'best_trial': {
                'number': self.study.best_trial.number,
                'value': self.study.best_trial.value,
                'params': self.study.best_trial.params
            },
            'n_trials': len(self.study.trials),
            'datetime': self.timestamp,
            'optimization_history': [
                {
                    'trial_number': t.number,
                    'value': t.value,
                    'params': t.params
                }
                for t in self.study.trials
            ]
        }
        
        stats_path = os.path.join(self.model_dir, 'optimization_results.json')
        with open(stats_path, 'w') as f:
            json.dump(study_stats, f, indent=4)
        
        self.logger.info(f"Saved optimization results to {stats_path}")
    
    def save_optimization_plots(self):
        """
        Saves visualization plots of the optimization process
        """
        if self.study is None:
            raise ValueError("No study available. Run optimization first.")
        
        plot_dir = os.path.join(self.model_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        # Save various visualization plots
        try:
            # Optimization history
            fig = optuna.visualization.plot_optimization_history(self.study)
            fig.write_html(os.path.join(plot_dir, 'optimization_history.html'))
            
            # Parameter importances
            fig = optuna.visualization.plot_param_importances(self.study)
            fig.write_html(os.path.join(plot_dir, 'parameter_importances.html'))
            
            # Parallel coordinate
            fig = optuna.visualization.plot_parallel_coordinate(self.study)
            fig.write_html(os.path.join(plot_dir, 'parameter_relationships.html'))
            
            # Slice plot
            fig = optuna.visualization.plot_slice(self.study)
            fig.write_html(os.path.join(plot_dir, 'slice_plot.html'))
            
            self.logger.info(f"Saved optimization plots to {plot_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving plots: {str(e)}")

def load_best_model(timestamp=None):
    """
    Loads the best model and its metadata from a specific training run
    """
    if timestamp is None:
        # Get most recent training directory
        all_timestamps = sorted(os.listdir(MODEL_SAVE_PATH))
        if not all_timestamps:
            raise ValueError("No trained models found")
        timestamp = all_timestamps[-1]
    
    model_path = os.path.join(MODEL_SAVE_PATH, timestamp, 'best_model.h5')
    metadata_path = os.path.join(MODEL_SAVE_PATH, timestamp, 'model_metadata.json')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return model, metadata