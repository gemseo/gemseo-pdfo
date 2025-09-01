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
"""Settings for the optimization algorithms from PDFO."""

from __future__ import annotations

from functools import partial

from gemseo.algos.opt.base_optimizer_settings import BaseOptimizerSettings
from gemseo.utils.pydantic import copy_field
from numpy import inf
from pydantic import Field
from pydantic import NonNegativeFloat  # noqa:TC002
from pydantic import PositiveInt  # noqa:TC002

copy_field_opt = partial(copy_field, model=BaseOptimizerSettings)


class BasePDFOSettings(BaseOptimizerSettings):
    """The PDFO optimization library settings."""

    ftol_rel: NonNegativeFloat = copy_field_opt("ftol_rel", default=1e-12)

    ftol_abs: NonNegativeFloat = copy_field_opt("ftol_abs", default=1e-12)

    xtol_abs: NonNegativeFloat = copy_field_opt("xtol_abs", default=1e-12)

    xtol_rel: NonNegativeFloat = copy_field_opt("xtol_rel", default=1e-12)

    max_time: NonNegativeFloat = copy_field_opt("max_time", default=0)

    max_iter: PositiveInt = copy_field_opt("max_iter", default=500)

    rhobeg: NonNegativeFloat = Field(
        default=0.5, description="The initial value of the trust region radius."
    )

    rhoend: NonNegativeFloat = Field(
        default=1e-6,
        description=(
            "The final value of the trust region radius. "
            "Indicates the accuracy required in the final values of the variables."
        ),
    )

    ftarget: float = Field(
        default=-inf,
        description=(
            "The target value of the objective function. "
            "If a feasible iterate achieves an objective function value "
            "lower or equal to ``ftarget``, the algorithm stops immediately."
        ),
    )

    scale: bool = Field(
        default=False,
        description=(
            "The flag indicating whether to scale the problem "
            "according to the bound constraints."
        ),
    )

    quiet: bool = Field(
        default=True,
        description=(
            "The flag of quietness of the interface. "
            "If ``True``, the output message will not be printed."
        ),
    )

    classical: bool = Field(
        default=False,
        description=(
            "The flag indicating whether to call the classical Powell code or not."
        ),
    )

    debug: bool = Field(default=False, description="The debugging flag.")

    chkfunval: bool = Field(
        default=False,
        description=(
            "A flag used when debugging. "
            "If both ``debug`` and ``chkfunval`` are ``True``, "
            "an extra function/constraint evaluation is performed "
            "to check whether the returned values of the objective function "
            "and constraint match the returned x."
        ),
    )

    ensure_bounds: bool = Field(
        default=True,
        description=(
            "Whether to project the design vector onto the design space "
            "before execution."
        ),
    )

    normalize_design_space: bool = copy_field_opt(
        "normalize_design_space", default=True
    )
