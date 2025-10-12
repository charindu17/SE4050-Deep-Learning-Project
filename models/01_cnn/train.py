import sys
import os
sys.path.append('../../')

from utils.data_loader import DataLoader
from utils.common import create_callbacks, setup_gpu
from utils.metrics import ModelEvaluator
from model import create_cnn_model, compile_model
from config import TRAINING_CONFIG, PATH_CONFIG

# QUICK_TEST will run a short 1-epoch training to smoke-test plotting/saving
QUICK_TEST = os.environ.get('QUICK_TEST', '0') == '1'

def train_cnn_model():
    """Training script - COMMON structure for all models"""
    # Setup
    setup_gpu()
    model_name = "CNN_Model"
    
    # Load data
    print("Loading data...")
    # Resolve data path relative to repo root when PATH_CONFIG is relative
    data_path = PATH_CONFIG.get('data_path', '../../data/processed_data/')
    data_loader = DataLoader(data_path=data_path)
    data_loader.load_data()
    data_shapes = data_loader.get_data_shapes()
    class_weights = data_loader.get_class_weights()
    
    # Create model
    print("Creating model...")
    model = create_cnn_model(
        input_shape=(data_shapes['timesteps'], data_shapes['features'])
    )
    model = compile_model(model)
    
    # Display model architecture
    model.summary()
    
    # Train model
    print("Training model...")
    # Ensure model directory exists so ModelCheckpoint can save the file
    model_save_path = PATH_CONFIG.get('model_save_path', f'models/{model_name}/best_model.h5')
    # Resolve model_save_path to absolute if necessary
    if not os.path.isabs(model_save_path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        model_save_path_abs = os.path.normpath(os.path.join(repo_root, model_save_path))
    else:
        model_save_path_abs = model_save_path
    os.makedirs(os.path.dirname(model_save_path_abs), exist_ok=True)

    callbacks = create_callbacks(model_name, model_save_path=model_save_path_abs)
    
    epochs = TRAINING_CONFIG.get('epochs', 100)
    batch_size = TRAINING_CONFIG.get('batch_size', 32)
    if QUICK_TEST:
        epochs = 1
        batch_size = min(32, data_loader.X_train.shape[0])

    history = model.fit(
        data_loader.X_train, data_loader.y_train,
        validation_data=(data_loader.X_val, data_loader.y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("Evaluating model...")
    evaluator = ModelEvaluator(model_name)
    evaluator.set_history(history)
    
    # Make predictions
    y_pred_proba = model.predict(data_loader.X_test)
    y_pred = y_pred_proba.argmax(axis=1)
    
    evaluator.set_predictions(data_loader.y_test, y_pred, y_pred_proba[:, 1])
    
    # Generate and save results
    report, cm = evaluator.generate_classification_report()
    
    # Plot results
    # Resolve results path
    results_path = PATH_CONFIG.get('results_path', '../../results/model_performance/')
    # Ensure results_path is absolute (relative to repo root)
    if not os.path.isabs(results_path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        results_path = os.path.normpath(os.path.join(repo_root, results_path))

    os.makedirs(results_path, exist_ok=True)

    evaluator.plot_training_history(os.path.join(results_path, f'{model_name}_training.png'))
    evaluator.plot_confusion_matrix(cm, os.path.join(results_path, f'{model_name}_cm.png'))
    evaluator.plot_roc_curve(os.path.join(results_path, f'{model_name}_roc.png'))
    evaluator.save_results(report, cm, save_dir=results_path)
    
    print(f"\n=== {model_name} Results ===")
    print(f"Test Accuracy: {report['accuracy']:.3f}")
    print(f"Sensitivity: {report['sensitivity']:.3f}")
    print(f"Specificity: {report['specificity']:.3f}")
    if 'roc_auc' in report:
        print(f"ROC AUC: {report['roc_auc']:.3f}")
    
    return model, history, report

if __name__ == "__main__":
    train_cnn_model()