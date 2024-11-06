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
from typing import Final
from typing import Optional
from typing import Union

from gemseo.algos.design_space_utils import get_value_and_bounds
from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary
from gemseo.algos.opt.base_optimization_library import OptimizationAlgorithmDescription
from gemseo.algos.opt.base_optimizer_settings import BaseOptimizerSettings
from numpy import isfinite
from numpy import ndarray
from numpy import real
from pdfo import pdfo

from gemseo_pdfo.settings.base_pdfo_settings import BasePDFOSettings
from gemseo_pdfo.settings.pdfo_bobyqa_settings import PDFO_BOBYQA_Settings
from gemseo_pdfo.settings.pdfo_cobyla_settings import PDFO_COBYLA_Settings
from gemseo_pdfo.settings.pdfo_newuoa_settings import PDFO_NEWUOA_Settings

if TYPE_CHECKING:
    from gemseo.algos.optimization_problem import OptimizationProblem

OptionType = Optional[Union[str, int, float, bool, ndarray]]


@dataclass
class PDFOAlgorithmDescription(OptimizationAlgorithmDescription):
    """The description of an optimization algorithm from the PDFO library."""

    library_name: str = "PDFO"

    Settings: type[BasePDFOSettings] = BasePDFOSettings
    """The option valiation model for  PDFO optimization library."""


class PDFOOpt(BaseOptimizationLibrary):
    """PDFO optimization library interface.

    See OptimizationLibrary.
    """

    __DOC: Final[str] = "https://www.pdfo.net/"

    LIBRARY_NAME = PDFOAlgorithmDescription.library_name

    LIB_COMPUTE_GRAD = False

    ALGORITHM_INFOS: ClassVar[dict[str, Any]] = {
        "PDFO_COBYLA": PDFOAlgorithmDescription(
            algorithm_name="COBYLA",
            description="Constrained Optimization By Linear Approximations ",
            handle_equality_constraints=True,
            handle_inequality_constraints=True,
            internal_algorithm_name="cobyla",
            positive_constraints=True,
            website=f"{__DOC}",
            Settings=PDFO_COBYLA_Settings,
        ),
        "PDFO_BOBYQA": PDFOAlgorithmDescription(
            algorithm_name="BOBYQA",
            description="Bound Optimization By Quadratic Approximation",
            internal_algorithm_name="bobyqa",
            website=f"{__DOC}",
            Settings=PDFO_BOBYQA_Settings,
        ),
        "PDFO_NEWUOA": PDFOAlgorithmDescription(
            algorithm_name="NEWUOA",
            description="NEWUOA",
            internal_algorithm_name="newuoa",
            website=f"{__DOC}",
            Settings=PDFO_NEWUOA_Settings,
        ),
    }

    def __init__(self, algo_name: str) -> None:  # noqa: D107
        super().__init__(algo_name)
        self.name = "PDFO"

    def _run(self, problem: OptimizationProblem, **settings: Any) -> tuple[str, Any]:
        # Filter settings to get only the ones of the PDFO optimizer
        settings_ = self._filter_settings(settings, BaseOptimizerSettings)

        # Get the normalized bounds:
        x_0, l_b, u_b = get_value_and_bounds(problem.design_space, self._normalize_ds)

        # Replace infinite values with None:
        l_b = [val if isfinite(val) else None for val in l_b]
        u_b = [val if isfinite(val) else None for val in u_b]

        ensure_bounds = settings_["ensure_bounds"]

        cstr_pdfo = []
        for cstr in self._get_right_sign_constraints(problem):
            c_pdfo = {"type": cstr.f_type}
            if ensure_bounds:
                c_pdfo["fun"] = self.__ensure_bounds(cstr.func)
            else:
                c_pdfo["fun"] = cstr.func

            cstr_pdfo.append(c_pdfo)

        # |g| is in charge of ensuring max iterations, since it may
        # have a different definition of iterations, such as for SLSQP
        # for instance which counts duplicate calls to x as a new iteration
        settings_["maxfev"] = int(settings["max_iter"] * 1.2)

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
                self._problem.design_space.project_into_bounds(
                    x_vect, self._normalize_ds
                )
            )

        opt_result = pdfo(
            fun=ensure_bounds_fun if ensure_bounds else real_part_fun,
            x0=x_0,
            method=self.ALGORITHM_INFOS[self._algo_name].internal_algorithm_name,
            bounds=list(zip(l_b, u_b)),
            constraints=cstr_pdfo,
            options=settings_,
        )

        return opt_result.message, opt_result.status

    def __ensure_bounds(
        self, orig_func: Callable[[ndarray], ndarray]
    ) -> Callable[[ndarray], ndarray]:
        """Project the design vector onto the design space before execution.

        Args:
            orig_func: The original function.

        Returns:
            A function calling the original function
            with the input data projected onto the design space.
        """

        def wrapped_func(x_vect):
            return orig_func(
                self._problem.design_space.project_into_bounds(
                    x_vect, self._normalize_ds
                )
            )

        return wrapped_func
