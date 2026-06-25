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

def plot_training_curves(train_losses, val_losses, model_name, phase, out_dir):
    """Genera curvas de Loss para las redes generativas y surrogadas."""
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    if len(val_losses) > 0:
        plt.plot(epochs, val_losses, 'r--', label='Validation Loss', linewidth=2)
        
    plt.title(f'Training & Validation Loss - {model_name} ({phase})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, f"{phase}_{model_name}_LearningCurve.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_parity(y_true, y_pred, metric_names, model_name, phase, out_dir):
    """Genera Parity Plots (True vs Predicted) para evaluar regresores multivariables."""
    os.makedirs(out_dir, exist_ok=True)
    n_metrics = y_true.shape[1]
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    
    if n_metrics == 1: 
        axes = [axes]
        
    for i in range(n_metrics):
        ax = axes[i]
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.4, color='teal')
        
        # Línea ideal (x=y)
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal')
        
        ax.set_title(f'{metric_names[i]}')
        ax.set_xlabel('True Scaled Value')
        ax.set_ylabel('Predicted Scaled Value')
        ax.legend()
        ax.grid(alpha=0.3)
        
    plt.suptitle(f'Parity Plot - {model_name} ({phase})', y=1.05, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{phase}_{model_name}_Parity.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_roc_curves(y_true, dict_y_scores, phase, out_dir):
    """Genera la FIGURA 1 del paper: Curvas ROC combinadas de varios modelos."""
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    
    # Paleta de colores para que se distingan bien en el paper
    colors = plt.cm.tab10(np.linspace(0, 1, len(dict_y_scores)))
    
    for (model_name, y_scores), color in zip(dict_y_scores.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2.5, label=f'{model_name} (AUC = {roc_auc:.4f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'Combined ROC Curves - {phase}', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(out_dir, f"{phase}_Combined_ROC_Paper.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_training_curves(dict_val_losses, phase, out_dir):
    """Genera la FIGURA 2 del paper: Curvas de Loss combinadas (ideal para ver la explosión del MDN)."""
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(dict_val_losses)))
    
    for (model_name, val_losses), color in zip(dict_val_losses.items(), colors):
        if len(val_losses) > 0:
            epochs = range(1, len(val_losses) + 1)
            # Líneas más gruesas para DDPM o cVAE si se quiere resaltar, estándar para todos aquí
            plt.plot(epochs, val_losses, color=color, lw=2.5, label=f'{model_name} (Val Loss)')
            
    # CRÍTICO: Escala Logarítmica Simétrica para que la explosión del MDN (10^14) no oculte los modelos buenos (0.4)
    plt.yscale('symlog', linthresh=1.0)
    
    plt.title(f'Combined Validation Loss Trajectories - {phase}', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (SymLog Scale)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, which="both", ls="--")
    plt.savefig(os.path.join(out_dir, f"{phase}_Combined_LossCurves_Paper.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_metric_bars(metrics_dict, metric_name, phase, out_dir, maximize=True):
    """Bonus para el Paper: Gráfico de barras comparando el rendimiento global (AUC, LogLoss, R2)."""
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    
    models = list(metrics_dict.keys())
    values = [metrics_dict[m] for m in models]

    # Ordenar de mejor a peor
    sorted_indices = np.argsort(values)
    if maximize: 
        sorted_indices = sorted_indices[::-1]

    models = [models[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]

    sns.barplot(x=values, y=models, palette="viridis")
    plt.title(f'{metric_name} Comparison - {phase}', fontsize=14)
    plt.xlabel(metric_name, fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    
    # Añadir los valores en texto sobre las barras
    for index, value in enumerate(values):
        plt.text(value, index, f' {value:.4f}', va='center')
        
    plt.savefig(os.path.join(out_dir, f"{phase}_Comparison_{metric_name}_Paper.png"), dpi=300, bbox_inches='tight')
    plt.close()
