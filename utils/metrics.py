import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import json

class ModelEvaluator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.history = None
        self.predictions = None
        
    def set_history(self, history):
        self.history = history
        
    def set_predictions(self, y_true, y_pred, y_pred_proba=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        
    def plot_training_history(self, save_path=None):
        """Plot training and validation loss/accuracy"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title(f'{self.model_name} - Training & Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title(f'{self.model_name} - Training & Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        if save_path:
            # Resolve relative paths against repository root (parent of utils/)
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            full_save_path = save_path if os.path.isabs(save_path) else os.path.join(repo_root, save_path)
            os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
            plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_classification_report(self):
        """Generate comprehensive classification report"""
        report = classification_report(self.y_true, self.y_pred, output_dict=True)
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        # Calculate additional metrics
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        
        # Add to report
        report['sensitivity'] = sensitivity
        report['specificity'] = specificity
        
        if self.y_pred_proba is not None:
            report['roc_auc'] = roc_auc_score(self.y_true, self.y_pred_proba)
        
        return report, cm
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Control', 'Condition'],
                   yticklabels=['Control', 'Condition'])
        plt.title(f'{self.model_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            full_save_path = save_path if os.path.isabs(save_path) else os.path.join(repo_root, save_path)
            os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
            plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_roc_curve(self, save_path=None):
        """Plot ROC curve if probability predictions available"""
        if self.y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_proba)
            auc_score = roc_auc_score(self.y_true, self.y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {auc_score:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{self.model_name} - ROC Curve')
            plt.legend(loc="lower right")
            plt.grid(True)
            
            if save_path:
                repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                full_save_path = save_path if os.path.isabs(save_path) else os.path.join(repo_root, save_path)
                os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
                plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            return auc_score
        return None
    
    def save_results(self, report, cm, save_dir='results/model_performance/'):
        """Save all results to files"""
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        full_save_dir = save_dir if os.path.isabs(save_dir) else os.path.join(repo_root, save_dir)
        os.makedirs(full_save_dir, exist_ok=True)

        # Save classification report
        with open(os.path.join(full_save_dir, f'{self.model_name}_report.json'), 'w') as f:
            json.dump(report, f, indent=2)

        # Save confusion matrix
        np.save(os.path.join(full_save_dir, f'{self.model_name}_confusion_matrix.npy'), cm)

        # Save training history
        if self.history:
            with open(os.path.join(full_save_dir, f'{self.model_name}_history.json'), 'w') as f:
                json.dump(self.history.history, f, indent=2)