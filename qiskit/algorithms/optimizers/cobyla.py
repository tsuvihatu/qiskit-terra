# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Constrained Optimization By Linear Approximation optimizer."""

from typing import Optional

from .optimizer import OptimizerCallback, OptimizerSupportLevel
from .scipy_optimizer import SciPyOptimizer


class COBYLA(SciPyOptimizer):
    """
    Constrained Optimization By Linear Approximation optimizer.

    COBYLA is a numerical optimization method for constrained problems
    where the derivative of the objective function is not known.

    Uses scipy.optimize.minimize COBYLA.
    For further detail, please refer to
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """

    _OPTIONS = ["maxiter", "disp", "rhobeg"]
    _callback_suppoert_level = OptimizerSupportLevel.supported

    # pylint: disable=unused-argument
    def __init__(
        self,
        maxiter: int = 1000,
        disp: bool = False,
        rhobeg: float = 1.0,
        tol: Optional[float] = None,
        options: Optional[dict] = None,
        callback: Optional[OptimizerCallback] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            maxiter: Maximum number of function evaluations.
            disp: Set to True to print convergence messages.
            rhobeg: Reasonable initial changes to the variables.
            tol: Final accuracy in the optimization (not precisely guaranteed).
                 This is a lower bound on the size of the trust region.
            options: A dictionary of solver options.
            callback: A callback function passed parameters in each iteration step.
            kwargs: additional kwargs for scipy.optimize.minimize.
        """
        if options is None:
            options = {}
        for k, v in list(locals().items()):
            if k in self._OPTIONS:
                options[k] = v
        super().__init__(method="COBYLA", options=options, tol=tol, callback=callback, **kwargs)
