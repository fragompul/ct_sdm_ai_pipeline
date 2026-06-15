# src/utils/xai_shap.py

import shap
import numpy as np
import matplotlib.pyplot as plt

class ShapleyExplainer:
    def __init__(self, trained_model, X_train_sample, feature_names):
        """
        trained_model: El modelo final de la Fase 2 (Idealmente el meta-learner del ensemble).
        X_train_sample: Una muestra representativa de los datos de entrenamiento para el baseline.
        """
        self.model = trained_model
        self.feature_names = feature_names
        
        # Dependiendo del modelo, usamos TreeExplainer (para XGBoost/RF) o KernelExplainer (para Ensembles complejos o MLP)
        try:
            # Intento de usar el explainer más rápido si es basado en árboles
            self.explainer = shap.TreeExplainer(self.model)
            print("Usando TreeExplainer.")
        except Exception:
            # Fallback seguro para StackingClassifier o MLP
            # Usamos el método predict_proba para explicar las probabilidades
            predict_fn = lambda x: self.model.predict_proba(x)
            self.explainer = shap.KernelExplainer(predict_fn, shap.sample(X_train_sample, 100))
            print("Usando KernelExplainer (modelo complejo detectado).")

    def generate_explanation(self, input_specs, predicted_topology_index):
        """
        Genera el Force Plot para una inferencia concreta.
        input_specs: Las especificaciones del usuario (SNDR, Bw, Power).
        predicted_topology_index: El índice de la clase que queremos explicar (ej. la top 1).
        """
        # Calcular los valores SHAP
        shap_values = self.explainer.shap_values(input_specs)
        
        # KernelExplainer devuelve una lista (una matriz por clase). Seleccionamos la clase predicha.
        if isinstance(shap_values, list):
            class_shap_values = shap_values[predicted_topology_index]
            expected_value = self.explainer.expected_value[predicted_topology_index]
        else:
            class_shap_values = shap_values[0, :, predicted_topology_index]
            expected_value = self.explainer.expected_value[predicted_topology_index]

        # Configurar figura para visualización (se podría guardar en un HTML en producción)
        plt.figure(figsize=(10, 3))
        
        # Usamos force_plot_html para web o matplotlib para local
        force_plot = shap.force_plot(
            base_value=expected_value, 
            shap_values=class_shap_values, 
            features=input_specs, 
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        
        plt.title(f"Lógica de Decisión para Topología {predicted_topology_index + 1}")
        plt.tight_layout()
        plt.savefig(f"shap_explanation_top_{predicted_topology_index + 1}.png")
        plt.close()
        
        print(f"Force plot generado para la clase {predicted_topology_index + 1}.")
        return class_shap_values