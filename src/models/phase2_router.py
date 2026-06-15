# src/models/phase2_router.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import xgboost as xgb
# Importar LightGBM y CatBoost según config, aquí simplificamos el pipeline base

class TopologicalRouter:
    def __init__(self, random_state=42, lambda_penalty=0.5):
        self.random_state = random_state
        self.lambda_penalty = lambda_penalty
        
        # Matriz estática de costes C en R^12.
        # Ejemplo figurado: Índices 0-11 corresponden a Topologías 1-12.
        # Orden 2 = Coste bajo, Orden 4 = Coste alto. Active-RC > Gm-C.
        self.cost_matrix = np.array([
            1.0, 1.2, # Top 1,2 (Orden 2)
            2.0, 2.2, # Top 3,4 (Orden 3)
            3.0, 3.5, # Top 5,6 (Orden 4)
            1.1, 1.3, # Top 7,8... etc
            2.1, 2.3,
            3.2, 3.6
        ])
        
        self.model = self._build_stacking_ensemble()

    def _build_stacking_ensemble(self):
        """Construye el Stacking Meta-Ensemble optimizando Log-Loss."""
        base_learners = [
            ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=self.random_state)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=self.random_state)),
            ('svm', SVC(kernel='rbf', probability=True, random_state=self.random_state)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=self.random_state))
        ]
        
        # Meta-learner: Logistic Regression para calibrar
        meta_learner = LogisticRegression(multi_class='multinomial', random_state=self.random_state)
        
        return StackingClassifier(estimators=base_learners, final_estimator=meta_learner, cv=5)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print("Stacking Ensemble entrenado correctamente.")

    def predict_top3_with_heuristic(self, X_infer):
        """
        Calcula Softmax raw, aplica penalización heurística y devuelve Top-3.
        Ecuación: P_adj(Ti) = (P(Ti) * e^(-lambda * Ci)) / Sumatoria(...).
        """
        raw_probs = self.model.predict_proba(X_infer) # [N_samples, 12]
        
        # Calcular factor de decaimiento exponencial: e^(-lambda * C)
        decay_factors = np.exp(-self.lambda_penalty * self.cost_matrix)
        
        adjusted_probs = raw_probs * decay_factors
        # Normalizar para que sumen 1
        row_sums = adjusted_probs.sum(axis=1)[:, np.newaxis]
        adjusted_probs = adjusted_probs / row_sums
        
        # Obtener los índices de las Top-3 topologías (orden descendente)
        top3_indices = np.argsort(adjusted_probs, axis=1)[:, -3:][:, ::-1]
        
        # Recuperar las probabilidades ajustadas de esas Top-3
        top3_probs = np.take_along_axis(adjusted_probs, top3_indices, axis=1)
        
        return top3_indices + 1, top3_probs # +1 asumiendo clases 1-12