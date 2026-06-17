# src/models/phase2_router.py

import numpy as np
import time
import optuna
import glob
import os
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss, precision_recall_fscore_support

class TopologicalRouterBenchmark:
    def __init__(self, random_state=42, lambda_penalty=0.25):
        self.random_state = random_state
        self.timings = {}
        self.best_params = {}
        self.trained_models = {}
        self.mapping_info = {}
        
        # Lambda=0.25 garantiza que un 70/30 sobrevive a favor del clasificador.
        self.lambda_penalty = lambda_penalty
        self.cost_matrix = self._build_dynamic_cost_matrix()

    def _build_dynamic_cost_matrix(self):
        """Lee los archivos raw exactamente igual que el dataset_builder original para mapear los costes."""
        files = glob.glob("data/raw/*.csv")
        cost_list = []
        
        print("\n" + "="*60)
        print("[ROUTER] MAPEANDO TOPOLOGÍAS Y PENALIZACIONES")
        print("="*60)
        
        for idx, file in enumerate(files):
            top_id = idx + 1
            name = os.path.basename(file).replace(".csv", "").lower()
            
            # Priorizar Orden (Bajo = 0.0, Medio = 1.0, Alto = 2.0)
            if "_2_" in name: order_cost = 0.0
            elif "_3_" in name: order_cost = 1.0
            elif "_4_" in name: order_cost = 2.0
            else: order_cost = 1.0 # Fallback
            
            # Priorizar Active-RC (0.0) frente a Gm-C (0.5)
            if "active_rc" in name: imp_cost = 0.0
            elif "gm_c" in name: imp_cost = 0.5
            else: imp_cost = 0.5 # Fallback
            
            total_cost = order_cost + imp_cost
            cost_list.append(total_cost)
            self.mapping_info[top_id] = {"name": name, "cost": total_cost}
            
            print(f"ID {top_id:02d} -> {name.ljust(22)} | Coste: {total_cost:.1f} (Ord:{order_cost}+Imp:{imp_cost})")
        print("="*60 + "\n")
        
        return np.array(cost_list)

        """
        ID 01 -> fb_2_active_rc         | Coste: 0.0 (Ord:0.0+Imp:0.0)
        ID 02 -> fb_4_gm_c              | Coste: 2.5 (Ord:2.0+Imp:0.5)
        ID 03 -> fb_4_active_rc         | Coste: 2.0 (Ord:2.0+Imp:0.0)
        ID 04 -> ff_3_active_rc         | Coste: 1.0 (Ord:1.0+Imp:0.0)
        ID 05 -> fb_3_gm_c              | Coste: 1.5 (Ord:1.0+Imp:0.5)
        ID 06 -> ff_2_active_rc         | Coste: 0.0 (Ord:0.0+Imp:0.0)
        ID 07 -> ff_3_gm_c              | Coste: 1.5 (Ord:1.0+Imp:0.5)
        ID 08 -> fb_3_active_rc         | Coste: 1.0 (Ord:1.0+Imp:0.0)
        ID 09 -> ff_4_active_rc         | Coste: 2.0 (Ord:2.0+Imp:0.0)
        ID 10 -> ff_2_gm_c              | Coste: 0.5 (Ord:0.0+Imp:0.5)
        ID 11 -> ff_4_gm_c              | Coste: 2.5 (Ord:2.0+Imp:0.5)
        ID 12 -> fb_2_gm_c              | Coste: 0.5 (Ord:0.0+Imp:0.5)
        """

    def _get_base_model(self, name, params=None):
        """Instancia condicionalmente el modelo solicitado (evitando instanciar todos a la vez)."""
        if params is None: params = {}
        
        if name == "LogisticRegression":
            return LogisticRegression(max_iter=1000, random_state=self.random_state, **params)
        elif name == "NaiveBayes":
            return GaussianNB(**params)
        elif name == "KNN":
            return KNeighborsClassifier(**params)
        elif name == "SVM_RBF":
            return SVC(kernel='rbf', probability=True, max_iter=2000, random_state=self.random_state, **params)
        elif name == "DecisionTree":
            return DecisionTreeClassifier(random_state=self.random_state, **params)
        elif name == "RandomForest":
            return RandomForestClassifier(random_state=self.random_state, **params)
        elif name == "ExtraTrees":
            return ExtraTreesClassifier(random_state=self.random_state, **params)
        elif name == "AdaBoost":
            return AdaBoostClassifier(random_state=self.random_state, **params)
        elif name == "XGBoost":
            return XGBClassifier(eval_metric='mlogloss', random_state=self.random_state, **params)
        elif name == "LightGBM":
            return LGBMClassifier(verbose=-1, random_state=self.random_state, **params)
        elif name == "CatBoost":
            return CatBoostClassifier(verbose=0, random_state=self.random_state, **params)
        elif name == "MLP":
            return MLPClassifier(max_iter=500, random_state=self.random_state, **params)
        else:
            raise ValueError(f"Modelo no reconocido: {name}")

    def optimize_and_train_all(self, X_train, y_train, X_val, y_val, n_trials=5):
        model_names = ["LogisticRegression", "NaiveBayes", "KNN", "DecisionTree", 
                       "RandomForest", "ExtraTrees", "AdaBoost", "XGBoost", 
                       "LightGBM", "CatBoost", "MLP", "SVM_RBF"]
        
        # 1. Búsqueda de Hiperparámetros (HPO)
        for name in model_names:
            start_hpo = time.time()
            
            # Usamos current_name para evitar problemas de late-binding en funciones internas
            def objective(trial, current_name=name):
                params = {}
                if current_name == "LogisticRegression":
                    params["C"] = trial.suggest_float("C", 1e-4, 10.0, log=True)
                elif current_name == "NaiveBayes":
                    params["var_smoothing"] = trial.suggest_float("var_smoothing", 1e-10, 1e-2, log=True)
                elif current_name == "KNN":
                    params["n_neighbors"] = trial.suggest_int("n_neighbors", 3, 20)
                elif current_name == "SVM_RBF":
                    params["C"] = trial.suggest_float("C", 0.1, 10.0, log=True)
                    params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
                elif current_name == "DecisionTree":
                    params["max_depth"] = trial.suggest_int("max_depth", 5, 30)
                elif current_name in ["RandomForest", "ExtraTrees"]:
                    params["n_estimators"] = trial.suggest_int("n_estimators", 50, 200)
                    params["max_depth"] = trial.suggest_int("max_depth", 5, 30)
                elif current_name in ["XGBoost", "LightGBM", "CatBoost"]:
                    params["n_estimators"] = trial.suggest_int("n_estimators", 50, 200)
                    params["learning_rate"] = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
                elif current_name == "AdaBoost":
                    params["n_estimators"] = trial.suggest_int("n_estimators", 50, 200)
                    params["learning_rate"] = trial.suggest_float("learning_rate", 1e-3, 1.0, log=True)
                elif current_name == "MLP":
                    params["hidden_layer_sizes"] = trial.suggest_categorical("hidden_layer_sizes", [(64,), (128, 64)])
                    params["alpha"] = trial.suggest_float("alpha", 1e-4, 1e-1, log=True)
                
                model = self._get_base_model(current_name, params)
                model.fit(X_train, y_train)
                preds = model.predict_proba(X_val)
                # Optimizar Log-Loss para calibrar probabilidades
                return log_loss(y_val, preds, labels=np.arange(12))

            study = optuna.create_study(direction="minimize")
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study.optimize(objective, n_trials=n_trials)
            
            self.timings[f"{name}_hpo_time"] = time.time() - start_hpo
            self.best_params[name] = study.best_params
            
            # Entrenamiento final
            start_train = time.time()
            best_model = self._get_base_model(name, self.best_params[name])
            best_model.fit(X_train, y_train)
            self.trained_models[name] = best_model
            self.timings[f"{name}_train_time"] = time.time() - start_train

        # 2. Creación Dinámica de Ensambles
        val_accs = {n: accuracy_score(y_val, m.predict(X_val)) for n, m in self.trained_models.items()}
            
        top3_names = sorted(val_accs, key=val_accs.get, reverse=True)[:3]
        above90_names = [n for n, acc in val_accs.items() if acc >= 0.90]
        if len(above90_names) < 2: above90_names = top3_names 
            
        groups = {
            "All": list(self.trained_models.keys()),
            "Top3": top3_names,
            "Above90": above90_names
        }
        
        for group_name, names in groups.items():
            estimators = [(n, self.trained_models[n]) for n in names]
            
            # Voting Classifier
            start_vote = time.time()
            voting = VotingClassifier(estimators=estimators, voting='soft')
            voting.fit(X_train, y_train)
            self.trained_models[f"Voting_{group_name}"] = voting
            self.timings[f"Voting_{group_name}_train_time"] = time.time() - start_vote
            
            # Stacking Classifier
            start_stack = time.time()
            meta = LogisticRegression(max_iter=1000, random_state=self.random_state)
            stacking = StackingClassifier(estimators=estimators, final_estimator=meta, cv=3)
            stacking.fit(X_train, y_train)
            self.trained_models[f"Stacking_{group_name}"] = stacking
            self.timings[f"Stacking_{group_name}_train_time"] = time.time() - start_stack

    def evaluate_all(self, X_test, y_test):
        results = {}
        for name, model in self.trained_models.items():
            start_inf = time.time()
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            self.timings[f"{name}_inference_time"] = time.time() - start_inf
            
            # Métricas Multiclase detalladas
            acc = accuracy_score(y_test, y_pred)
            lloss = log_loss(y_test, y_proba, labels=np.arange(12))
            prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
            
            top3_preds = np.argsort(y_proba, axis=1)[:, -3:]
            top3_acc = np.mean([1 if y_test[i] in top3_preds[i] else 0 for i in range(len(y_test))])
            
            results[name] = {
                "y_pred": y_pred, "y_proba": y_proba,
                "metrics": {"Accuracy": acc, "Top3_Accuracy": top3_acc, "Precision": prec, "Recall": rec, "F1": f1, "LogLoss": lloss}
            }
        return results

    def predict_with_heuristic(self, best_model_name, X_infer):
        model = self.trained_models[best_model_name]
        raw_probs = model.predict_proba(X_infer)
        
        decay_factors = np.exp(-self.lambda_penalty * self.cost_matrix)
        adj_probs = raw_probs * decay_factors
        adj_probs /= adj_probs.sum(axis=1)[:, np.newaxis]
        
        top3_indices = np.argsort(adj_probs, axis=1)[:, -3:][:, ::-1]
        top3_probs = np.take_along_axis(adj_probs, top3_indices, axis=1)
        
        return top3_indices + 1, top3_probs