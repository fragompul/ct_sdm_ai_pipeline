# src/utils/xai_shap.py

import shap
import numpy as np
import matplotlib.pyplot as plt
import os

class ShapleyExplainer:
    def __init__(self, trained_model, X_train_sample, feature_names):
        self.model = trained_model
        self.feature_names = feature_names
        
        try:
            self.explainer = shap.TreeExplainer(self.model)
            print("XAI: Usando TreeExplainer.")
        except Exception:
            predict_fn = lambda x: self.model.predict_proba(x)
            self.explainer = shap.KernelExplainer(predict_fn, shap.sample(X_train_sample, 100))
            print("XAI: Usando KernelExplainer.")

    def generate_explanation(self, input_specs, predicted_topology_index, final_adj_prob=None, out_dir="logs/plots/xai/"):
        """Inferencia única: Genera Force Plot y texto natural."""
        os.makedirs(out_dir, exist_ok=True)
        shap_values = self.explainer.shap_values(input_specs)
        
        if isinstance(shap_values, list):
            class_shap_values = shap_values[predicted_topology_index]
            expected_value = self.explainer.expected_value[predicted_topology_index]
        else:
            class_shap_values = shap_values[0, :, predicted_topology_index]
            expected_value = self.explainer.expected_value[predicted_topology_index]

        plt.figure(figsize=(10, 3))
        shap.force_plot(
            base_value=expected_value, 
            shap_values=class_shap_values, 
            features=input_specs, 
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        plt.title(f"Lógica de Decisión para Topología {predicted_topology_index + 1}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"shap_explanation_top_{predicted_topology_index + 1}.png"))
        plt.close()

        base_prob = expected_value * 100
        impacts = class_shap_values[0] if len(class_shap_values.shape) > 1 else class_shap_values
        
        text_report = f"--- INFORME DE INTELIGENCIA EXPLICABLE (XAI) ---\n"
        text_report += f"La probabilidad base (sesgo del dataset) para esta arquitectura es del {base_prob:.1f}%.\n"
        if final_adj_prob: text_report += f"Sin embargo, la confianza final se ajustó al {final_adj_prob*100:.1f}%. Esto se debe a:\n"
        
        for j, feature in enumerate(self.feature_names):
            impact_percent = impacts[j] * 100
            sign = "aumentó" if impact_percent > 0 else "penalizó"
            text_report += f"  • El requisito de {feature} ({input_specs[0][j]:.2e}) {sign} la probabilidad en un {abs(impact_percent):.2f}%.\n"
            
        return impacts, text_report

    def generate_global_explanation(self, X_sample, out_dir="logs/plots/benchmark/"):
        """Benchmark: Genera un Global Feature Importance Plot a prueba de fallos."""
        os.makedirs(out_dir, exist_ok=True)
        shap_values = self.explainer.shap_values(X_sample)
        
        # 1. Extracción matemática segura (Bypass del bug de la librería SHAP)
        if isinstance(shap_values, list):
            # shap_values es una lista de 12 clases, cada una con un array (samples, features)
            mean_abs_impact = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            mean_abs_impact = np.abs(shap_values).mean(axis=(0, 2)) if len(shap_values.shape) > 2 else np.abs(shap_values).mean(axis=0)
            
        importance_dict = {feat: float(imp) for feat, imp in zip(self.feature_names, mean_abs_impact)}

        # 2. Generación del gráfico de barras horizontal (Global Importance)
        sorted_indices = np.argsort(mean_abs_impact)
        sorted_features = [self.feature_names[i] for i in sorted_indices]
        sorted_impacts = [mean_abs_impact[i] for i in sorted_indices]

        plt.figure(figsize=(10, 6))
        bars = plt.barh(sorted_features, sorted_impacts, color='teal', edgecolor='black', alpha=0.8)
        
        # Añadir las etiquetas de valor al lado de cada barra
        for bar in bars:
            plt.text(bar.get_width() + (max(sorted_impacts)*0.01), bar.get_y() + bar.get_height()/2, 
                     f'{bar.get_width():.4f}', va='center', ha='left', fontsize=10)

        plt.xlabel("Mean Absolute SHAP Value (Global Impact on Routing)", fontsize=12)
        plt.title("Global Feature Importance for Topological Routing", fontsize=14, pad=15)
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        
        # Ampliar un poco el eje X para que quepa el texto
        plt.xlim(0, max(sorted_impacts) * 1.15) 
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "Benchmark_SHAP_Global_Summary.png"), dpi=300)
        plt.close()
        
        return importance_dict