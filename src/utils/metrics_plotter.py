# src/utils/metrics_plotter.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize

def plot_roc_curve(y_true, y_scores, model_name, phase, out_dir):
    """Genera y guarda la curva ROC."""
    os.makedirs(out_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name} ({phase})')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(out_dir, f"{phase}_{model_name}_ROC.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name, phase, out_dir, class_names=None):
    """Genera y guarda la Matriz de Confusión estilo Heatmap."""
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names if class_names else "auto", 
                yticklabels=class_names if class_names else "auto")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name} ({phase})')
    plt.savefig(os.path.join(out_dir, f"{phase}_{model_name}_CM.png"), dpi=300, bbox_inches='tight')
    plt.close()