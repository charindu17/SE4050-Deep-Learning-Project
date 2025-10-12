# models/01_cnn/config.py
# models/02_lstm/config.py  
# models/03_bilstm/config.py
# models/04_hybrid_cnn_lstm/config.py

"""
Model Configuration File
Team members can modify these hyperparameters for their specific model
"""

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'early_stopping_patience': 15,
    'reduce_lr_patience': 10,
    'monitor_metric': 'val_loss'
}

# Model Architecture Configuration
# CNN-specific config
CNN_CONFIG = {
    'conv_filters': [64, 128, 256],
    'kernel_sizes': [7, 5, 3],
    'pool_sizes': [2, 2, 2],
    'dense_units': [512, 256],
    'dropout_rates': [0.3, 0.3, 0.3, 0.5, 0.5]
}

# LSTM-specific config  
LSTM_CONFIG = {
    'lstm_units': [128, 64, 32],
    'dense_units': [256, 128, 64],
    'dropout_rates': [0.3, 0.3, 0.3, 0.5, 0.5, 0.3]
}

# BiLSTM-specific config
BILSTM_CONFIG = {
    'lstm_units': [128, 64, 32],
    'dense_units': [256, 128, 64],
    'dropout_rates': [0.3, 0.3, 0.3, 0.5, 0.5, 0.3],
    'merge_mode': 'concat'
}

# Hybrid-specific config
HYBRID_CONFIG = {
    'conv_filters': [64, 128, 256],
    'kernel_sizes': [7, 5, 3],
    'lstm_units': [128, 64],
    'dense_units': [512, 256, 128],
    'dropout_rates': [0.2, 0.2, 0.3, 0.3, 0.5, 0.5, 0.3]
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    'save_plots': True,
    'plot_dpi': 300,
    'generate_report': True,
    'save_predictions': True
}

# Path Configuration
PATH_CONFIG = {
    # paths should be relative to the repository root (not include ../.. segments)
    'data_path': 'data/processed_data/',
    'results_path': 'results/model_performance/',
    # model_save_path is a filename or relative path under results or models folder
    # Use the model path inside the model folder as requested
    'model_save_path': 'models/01_cnn/models/CNN_Model/best_model.h5'
}