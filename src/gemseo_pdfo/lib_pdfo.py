# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Jean-Christophe Giret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""PDFO optimization library wrapper, see the [PDFO website](https://www.pdfo.net)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Optional
from typing import Union

from gemseo.algos.design_space_utils import get_value_and_bounds
from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary
from gemseo.algos.opt.base_optimization_library import OptimizationAlgorithmDescription
from numpy import inf
from numpy import isfinite
from numpy import ndarray
from numpy import real
from pdfo import pdfo

if TYPE_CHECKING:
    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.algos.optimization_result import OptimizationResult

OptionType = Optional[Union[str, int, float, bool, ndarray]]


@dataclass
class PDFOAlgorithmDescription(OptimizationAlgorithmDescription):
    """The description of an optimization algorithm from the PDFO library."""

    library_name: str = "PDFO"
    website: str = "https://www.pdfo.net/"


class PDFOOpt(BaseOptimizationLibrary):
    """PDFO optimization library interface.

    See OptimizationLibrary.
    """

    LIB_COMPUTE_GRAD = False
    LIBRARY_NAME = "PDFO"

    ALGORITHM_INFOS: ClassVar[dict[str, Any]] = {
        "PDFO_COBYLA": PDFOAlgorithmDescription(
            algorithm_name="COBYLA",
            description="Constrained Optimization By Linear Approximations ",
            handle_equality_constraints=True,
            handle_inequality_constraints=True,
            internal_algorithm_name="cobyla",
            positive_constraints=True,
        ),
        "PDFO_BOBYQA": PDFOAlgorithmDescription(
            algorithm_name="BOBYQA",
            description="Bound Optimization By Quadratic Approximation",
            internal_algorithm_name="bobyqa",
        ),
        "PDFO_NEWUOA": PDFOAlgorithmDescription(
            algorithm_name="NEWUOA",
            description="NEWUOA",
            internal_algorithm_name="newuoa",
        ),
    }

    def __init__(self, algo_name: str) -> None:  # noqa: D107
        super().__init__(algo_name)
        self.name = "PDFO"

    def _get_options(
        self,
        ftol_rel: float = 1e-12,
        ftol_abs: float = 1e-12,
        xtol_rel: float = 1e-12,
        xtol_abs: float = 1e-12,
        max_time: float = 0,
        rhobeg: float = 0.5,
        rhoend: float = 1e-6,
        max_iter: int = 500,
        ftarget: float = -inf,
        scale: bool = False,
        quiet: bool = True,
        classical: bool = False,
        debug: bool = False,
        chkfunval: bool = False,
        ensure_bounds: bool = True,
        normalize_design_space: bool = True,
        **kwargs: OptionType,
    ) -> dict[str, Any]:
        r"""Set the options default values.

        To get the best and up-to-date information about algorithms options,
        go to pdfo documentation on the [PDFO website](https://www.pdfo.net/).

        Args:
            ftol_rel: A stop criteria, relative tolerance on the
               objective function,
               if abs(f(xk)-f(xk+1))/abs(f(xk))<= ftol_rel: stop.
            ftol_abs: A stop criteria, absolute tolerance on the objective
               function, if abs(f(xk)-f(xk+1))<= ftol_rel: stop.
            xtol_rel: A stop criteria, relative tolerance on the
               design variables,
               if norm(xk-xk+1)/norm(xk)<= xtol_rel: stop.
            xtol_abs: A stop criteria, absolute tolerance on the
               design variables,
               if norm(xk-xk+1)<= xtol_abs: stop.
            max_time: The maximum runtime in seconds,
                disabled if 0.
            rhobeg: The initial value of the trust region radius.
            max_iter: The maximum number of iterations.
            rhoend: The final value of the trust region radius. Indicates
                the accuracy required in the final values of the variables.
            maxfev:  The upper bound of the number of calls of the objective function
                `fun`.
            ftarget: The target value of the objective function. If a feasible
                iterate achieves an objective function value lower or equal to
                `options['ftarget']`, the algorithm stops immediately.
            scale: The flag indicating whether to scale the problem according to
                the bound constraints.
            quiet: The flag of quietness of the interface. If True,
                the output message will not be printed.
            classical: The flag indicating whether to call the classical Powell code
                or not.
            debug: The debugging flag.
            chkfunval: A flag used when debugging. If both `options['debug']`
                and `options['chkfunval']` are True, an extra function/constraint
                evaluation would be performed to check whether the returned values of
                the objective function and constraint match the returned x.
            ensure_bounds: Whether to project the design vector
                onto the design space before execution.
            normalize_design_space: If True, normalize the design space.
            **kwargs: The other algorithm's options.
        """
        return self._process_options(
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            xtol_abs=xtol_abs,
            max_time=max_time,
            rhobeg=rhobeg,
            rhoend=rhoend,
            max_iter=max_iter,
            ftarget=ftarget,
            scale=scale,
            quiet=quiet,
            classical=classical,
            debug=debug,
            chkfunval=chkfunval,
            ensure_bounds=ensure_bounds,
            normalize_design_space=normalize_design_space,
            **kwargs,
        )

    def _run(
        self, problem: OptimizationProblem, **options: OptionType
    ) -> OptimizationResult:
        """Run the algorithm, to be overloaded by subclasses.

        Args:
            **options: The options of the algorithm.

        Returns:
            The optimization result.
        """
        # Remove GEMSEO options to avoid passing them to the optimizer.
        del options[self._X_TOL_ABS]
        del options[self._X_TOL_REL]
        del options[self._F_TOL_ABS]
        del options[self._F_TOL_REL]
        del options[self._MAX_TIME]
        normalize_ds = options.pop(self._NORMALIZE_DESIGN_SPACE_OPTION, True)
        # Get the normalized bounds:
        x_0, l_b, u_b = get_value_and_bounds(problem.design_space, normalize_ds)

        # Replace infinite values with None:
        l_b = [val if isfinite(val) else None for val in l_b]
        u_b = [val if isfinite(val) else None for val in u_b]

        ensure_bounds = options.pop("ensure_bounds")

        cstr_pdfo = []
        for cstr in self._get_right_sign_constraints(problem):
            c_pdfo = {"type": cstr.f_type}
            if ensure_bounds:
                c_pdfo["fun"] = self.__ensure_bounds(cstr.func, normalize_ds)
            else:
                c_pdfo["fun"] = cstr.func

            cstr_pdfo.append(c_pdfo)

        # |g| is in charge of ensuring max iterations, since it may
        # have a different definition of iterations, such as for SLSQP
        # for instance which counts duplicate calls to x as a new iteration
        options["maxfev"] = int(options.pop(self._MAX_ITER) * 1.2)

        def real_part_fun(
            x: ndarray,
        ) -> int | float:
            """Wrap the objective function and keep the real part.

            Args:
                x: The values to be given to the function.

            Returns:
                The real part of the evaluation of the function.
            """
            return real(problem.objective.evaluate(x))

        def ensure_bounds_fun(x_vect):
            return real_part_fun(
                self.problem.design_space.project_into_bounds(x_vect, normalize_ds)
            )

        opt_result = pdfo(
            fun=ensure_bounds_fun if ensure_bounds else real_part_fun,
            x0=x_0,
            method=self.ALGORITHM_INFOS[self._algo_name].internal_algorithm_name,
            bounds=list(zip(l_b, u_b)),
            constraints=cstr_pdfo,
            options=options,
        )

        return self._get_optimum_from_database(
            problem, opt_result.message, opt_result.status
        )

    def __ensure_bounds(
        self, orig_func: Callable[[ndarray], ndarray], normalize: bool = True
    ) -> Callable[[ndarray], ndarray]:
        """Project the design vector onto the design space before execution.

        Args:
            orig_func: The original function.
            normalize: Whether to use the normalized design space.

        Returns:
            A function calling the original function
            with the input data projected onto the design space.
        """

        def wrapped_func(x_vect):
            return orig_func(
                self.problem.design_space.project_into_bounds(x_vect, normalize)
            )

        return wrapped_func
