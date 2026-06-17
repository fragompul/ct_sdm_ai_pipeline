# src/utils/metrics_plotter.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize

def plot_roc_curve(y_true, y_scores, model_name, phase, out_dir, n_classes=None):
    """Genera y guarda la curva ROC (Binaria o Multiclase)."""
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    
    if n_classes is not None and n_classes > 2:
        # Multiclase (Macro-Average ROC)
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        if y_true_bin.shape[1] == 1:
            pass # Prevenir errores si solo hay 2 clases presentes
        else:
            fpr_grid = np.linspace(0.0, 1.0, 1000)
            mean_tpr = np.zeros_like(fpr_grid)
            for i in range(n_classes):
                # Calcular la curva ROC por cada clase
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
                mean_tpr += np.interp(fpr_grid, fpr, tpr)
            mean_tpr /= n_classes
            roc_auc = auc(fpr_grid, mean_tpr)
            plt.plot(fpr_grid, mean_tpr, color='darkorange', lw=2, label=f'Macro-average ROC (AUC = {roc_auc:.4f})')
    else:
        # Binaria (Fase 1 OOD)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
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