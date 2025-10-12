import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os

def create_callbacks(model_name, model_save_path=None, patience=15, monitor='val_loss'):
    """Create common callbacks for all models.

    model_save_path: optional path (absolute or repo-root-relative) where the
    ModelCheckpoint will save the best model. If not provided, defaults to
    models/{model_name}/best_model.h5 (relative to repo root).
    """
    # Resolve default save path
    if model_save_path is None:
        model_save_path = f'models/{model_name}/best_model.h5'

    # If path is not absolute, make it repo-root relative
    if not os.path.isabs(model_save_path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_save_path = os.path.normpath(os.path.join(repo_root, model_save_path))

    # Ensure directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            model_save_path,
            monitor=monitor,
            save_best_only=True,
            verbose=1
        )
    ]
    return callbacks

def setup_gpu():
    """Setup GPU configuration for consistent training"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU setup complete: {len(gpus)} GPU(s) available")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU available, using CPU")