from json import load
import os
import csv
import numpy as np
from gc import callbacks
from typing import Any, Callable, Dict, Union, List, Optional, Tuple

from scipy_optimizer import SciPyOptimizer
from optimizer import Optimizer, OptimizerSupportLevel, OptimizerResult, POINT

class SuffixAveragingOptimizer(SciPyOptimizer):
    def __init__(
        self,
        optimizer: SciPyOptimizer,
        optimizer_name: str, 
        alpha: float = 0.3,
        suffix_dir: str = None,
        suffix_filename: str = None,
        save_params: bool = False,
        save_averaged_params: bool = False
    ) -> None:
        """
        Args:
            optimizer: 
            alpha: The hyperparameter to determine the ratio of ansatz parameters to take the average.
            suffix_dir: The directory for storing ansatz parameters.
            suffix_filename: The filename for storing ansatz parameters.
            save_params: If True, intermediate ansatz parameters are stored.
            save_averaged_params: If True, intermediate ansatz parameters over which the suffix average is taken are stored.

        References:
            [1] S. Tamiya and H. Yamasaki. 2021.
            Stochastic Gradient Line Bayesian Optimization: Reducing Measurement Shots in Optimizing Parameterized Quantum Circuits.
            arXiv preprint arXiv:2111.07952.
        """

        self._alpha = alpha
        self._suffix_dir = suffix_dir
        self._suffix_filename = suffix_filename
        self._save_params = save_params
        self._save_averaged_params = save_averaged_params
        self._optimizer = optimizer

        self._circ_params = []
        
        if optimizer_name == "SPSA":
            def load_params(nfev, x_next, fx_next, update_step, bool):
                self._circ_params.append(x_next)
        elif optimizer_name == "QNSPSA":
            def load_params(nfev, x_next, fx_next, update_step, bool):
                self._circ_params.append(x_next)
        elif optimizer_name == "GradientDescent":
            def load_params(nfevs, x_next, fx_next, stepsize):
                self._circ_params.append(x_next)
        else:
            def load_params(x):
                self._circ_params.append(x)

        #super().__init__(method=optimizer._method, options=optimizer._options, callback=load_params)
        self._optimizer.__init__(callback = load_params)

    def _save_circ_params(self, circ_params: List[float], csv_dir: str, csv_filename: str) -> None:
        with open(os.path.join(csv_dir, csv_filename+".csv"), mode="a") as csv_file:
            writer = csv.writer(csv_file, lineterminator='\n')
            writer.writerows(circ_params)

    def read_circ_params(self, csv_dir: str, csv_filename: str) -> List[float]:
        with open(os.path.join(csv_dir, csv_filename+".csv")) as csv_file:
            reader = csv.reader(csv_file)
            circ_params = [list(map(float, row)) for row in reader]
        return circ_params

    def _return_suffix_average(self) -> List[float]:
        if self._save_params:
            self._save_circ_params(self._circ_params, self._suffix_dir, self._suffix_filename)
        
        if self._save_averaged_params:
            n_iterates = len(self._circ_params)
            averaged_params = np.zeros_like(self._circ_params)
            for i in range(n_iterates):
                averaged_param = np.zeros_like(self._circ_params[0])
                for j in range(int(np.ceil(i*self._alpha))):
                    averaged_param += self._circ_params[i-j]
                averaged_param /= np.ceil(i*self._alpha)
                averaged_params[i] = averaged_param
            self._save_circ_params(averaged_params, self._suffix_dir, self._suffix_filename+"suffix")
            return averaged_params[-1]
                
        else:
            n_iterates = len(self._circ_params)
            averaged_param = np.zeros_like(self._circ_params[0])
            for j in range(int(np.ceil(n_iterates*self._alpha))):
                averaged_param += self._circ_params[n_iterates-j-1]
            averaged_param /= np.ceil(n_iterates*self._alpha)

            return averaged_param
        
    def minimize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Optional[Callable[[POINT], POINT]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> OptimizerResult:

        result = self._optimizer.minimize(fun, x0, jac=jac, bounds=bounds)
        result.x = self._return_suffix_average()
        result.fun = fun(np.copy(result.x))

        return result