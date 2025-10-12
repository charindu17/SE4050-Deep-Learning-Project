import os
import numpy as np
import joblib
from sklearn.utils.class_weight import compute_class_weight

class DataLoader:
    def __init__(self, data_path='data/processed_data/'):
        # Normalize data_path. If a relative path is provided, resolve it
        # relative to the repository root (parent directory of utils).
        if not os.path.isabs(data_path):
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            self.data_path = os.path.normpath(os.path.join(repo_root, data_path))
        else:
            self.data_path = data_path
        self.X_train, self.X_val, self.X_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None
        self.class_weights = None
        
    def load_data(self):
        """Load preprocessed data. Builds file paths using os.path.join and
        provides clearer error messages when files are missing.
        """
        files = {
            'X_train': os.path.join(self.data_path, 'X_train.npy'),
            'X_val': os.path.join(self.data_path, 'X_val.npy'),
            'X_test': os.path.join(self.data_path, 'X_test.npy'),
            'y_train': os.path.join(self.data_path, 'y_train.npy'),
            'y_val': os.path.join(self.data_path, 'y_val.npy'),
            'y_test': os.path.join(self.data_path, 'y_test.npy'),
        }

        # Check files exist before attempting to load
        for name, path in files.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required data file not found: {path}\n"
                                        f"Ensure you run preprocessing or point DataLoader to the correct data_path.")

        self.X_train = np.load(files['X_train'])
        self.X_val = np.load(files['X_val'])
        self.X_test = np.load(files['X_test'])
        self.y_train = np.load(files['y_train'])
        self.y_val = np.load(files['y_val'])
        self.y_test = np.load(files['y_test'])
        
        print(f"Data loaded successfully:")
        print(f"Train: {self.X_train.shape}, {self.y_train.shape}")
        print(f"Val:   {self.X_val.shape}, {self.y_val.shape}")
        print(f"Test:  {self.X_test.shape}, {self.y_test.shape}")
        
        return self
    
    def get_class_weights(self):
        """Compute class weights for imbalance and return a mapping
        from class label to its weight (works for any label values).
        """
        classes = np.unique(self.y_train)
        weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=self.y_train
        )
        class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
        print(f"Class weights: {class_weight_dict}")
        self.class_weights = class_weight_dict
        return class_weight_dict
    
    def get_data_shapes(self):
        """Return data shapes for model configuration"""
        return {
            'timesteps': self.X_train.shape[1],
            'features': self.X_train.shape[2],
            'num_classes': len(np.unique(self.y_train))
        }