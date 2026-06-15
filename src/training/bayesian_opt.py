# src/optimization/bayesian_opt.py

from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real
import numpy as np

class BayesianSearch:
    def __init__(self, non_diff_surrogate, topology_onehot, all_design_vars, mask):
        self.surrogate = non_diff_surrogate
        self.topology_onehot = topology_onehot
        self.mask = mask
        
        # Definir el espacio de búsqueda solo para las variables pertinentes (donde mask == 1)
        self.search_space = []
        self.active_indices = []
        
        for idx, is_active in enumerate(self.mask):
            if is_active == 1:
                # Límites genéricos, en la práctica vendrían de los dominios físicos de cada variable
                self.search_space.append(Real(1e-15, 1e-3, name=f'var_{idx}'))
                self.active_indices.append(idx)

    def optimize(self, n_calls=50):
        """Búsqueda activa dentro del espacio acotado."""
        
        def objective_function(active_vars):
            # Reconstruir el súper-vector con zero-padding
            y_design = np.zeros(len(self.mask))
            for i, val in zip(self.active_indices, active_vars):
                y_design[i] = val
                
            x_surrogate = np.concatenate([self.topology_onehot, y_design]).reshape(1, -1)
            
            # Predecir métricas [SNDR, Bw, Power]
            metrics = self.surrogate.predict(x_surrogate)[0]
            
            # Maximizar FoM es minimizar -FoM
            sndr, bw, power = metrics[0], metrics[1], metrics[2]
            fom = sndr + 10 * np.log10(bw / (power + 1e-12))
            
            return -fom # skopt minimiza por defecto

        # Ejecutar Gaussian Process Optimization
        res = gp_minimize(
            func=objective_function,
            dimensions=self.search_space,
            acq_func="EI", # Expected Improvement
            n_calls=n_calls,
            random_state=42
        )
        
        # Reconstruir el vector ganador
        best_y = np.zeros(len(self.mask))
        for i, val in zip(self.active_indices, res.x):
            best_y[i] = val
            
        return best_y, -res.fun