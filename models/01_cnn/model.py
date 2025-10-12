"""1D CNN model builder for models/01_cnn."""

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization

# Try to load a local CNN_CONFIG from the same package; fall back to top-level config import or None
try:
    from .config import CNN_CONFIG  # package-relative import when used as module
except Exception:
    try:
        from config import CNN_CONFIG  # when run from this folder directly
    except Exception:
        CNN_CONFIG = None


def create_cnn_model(input_shape, num_classes=2, config=None):
    """Create a configurable 1D CNN model.

    Args:
        input_shape (tuple): (timesteps, features)
        num_classes (int): number of output classes
        config (dict, optional): keys: conv_filters, kernel_sizes, pool_sizes,
            dense_units, dropout_rates

    Returns:
        tf.keras.Model: uncompiled model
    """
    cfg = config or CNN_CONFIG or {
        'conv_filters': [64, 128, 256],
        'kernel_sizes': [7, 5, 3],
        'pool_sizes': [2, 2, 2],
        'dense_units': [512, 256],
        # dropout_rates length should be at least len(conv_filters) + len(dense_units)
        'dropout_rates': [0.3, 0.3, 0.3, 0.5, 0.5]
    }

    conv_filters = cfg.get('conv_filters', [64, 128, 256])
    kernel_sizes = cfg.get('kernel_sizes', [7, 5, 3])
    pool_sizes = cfg.get('pool_sizes', [2 for _ in conv_filters])
    dropout_rates = cfg.get('dropout_rates', [0.3 for _ in conv_filters] + [0.5, 0.5])
    dense_units = cfg.get('dense_units', [512, 256])

    model = Sequential()
    model.add(Input(shape=input_shape))

    # Convolutional blocks
    for i, filters in enumerate(conv_filters):
        k = kernel_sizes[i] if i < len(kernel_sizes) else kernel_sizes[-1]
        p = pool_sizes[i] if i < len(pool_sizes) else pool_sizes[-1]
        d = dropout_rates[i] if i < len(dropout_rates) else 0.3

        model.add(Conv1D(filters, kernel_size=k, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=p))
        model.add(Dropout(d))

    # Classification head
    model.add(Flatten())
    for j, units in enumerate(dense_units):
        model.add(Dense(units, activation='relu'))
        dr_index = len(conv_filters) + j
        dr = dropout_rates[dr_index] if dr_index < len(dropout_rates) else 0.5
        model.add(Dropout(dr))

    model.add(Dense(num_classes, activation='softmax'))

    return model


def compile_model(model, learning_rate=0.001):
    """Compile the model using Adam and sparse categorical crossentropy."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == '__main__':
    # Quick smoke test: build and show summary for a synthetic input shape
    m = create_cnn_model(input_shape=(100, 13), num_classes=2)
    compile_model(m)
    m.summary()