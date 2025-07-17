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
#    INITIAL AUTHORS - API and implementation and/or documentation
#      :author: Jean-Christophe Giret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from copy import copy
from math import sqrt
from unittest import TestCase

import pytest
from gemseo import execute_algo
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary
from gemseo.algos.opt.factory import OptimizationLibraryFactory
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.problems.optimization.rosenbrock import Rosenbrock
from gemseo.utils.testing.opt_lib_test_base import OptLibraryTestBase
from numpy import isnan
from numpy import nan
from scipy.optimize import rosen
from scipy.optimize import rosen_der

from gemseo_pdfo.pdfo import PDFOOpt

pytest.importorskip("pdfo", reason="pdfo is not available")


class TestPDFO(TestCase):
    OPT_LIB_NAME = "PDFOOpt"

    def test_failed(self):
        """"""
        algo_name = "PDFO_COBYLA"
        self.assertRaises(
            ValueError,
            OptLibraryTestBase.generate_error_test,
            "PDFOAlgorithms",
            algo_name=algo_name,
            max_iter=10,
            ensure_bounds=False,
        )

    def test_nan_handling(self):
        """Test that an occurrence of NaN value in the objective function does not stop
        the optimizer.

        In this case, a NaN "bubble" is put at the beginning of the optimizer path. In
        this test, it is expected that the optimizer will encounter and by-pass the NaN
        bubble.
        """
        opt_problem = Rosenbrock()

        fun = copy(opt_problem.objective)

        def wrapped_fun(x_vec):
            x = x_vec[0]
            y = x_vec[1]

            d = sqrt((x - 0.1) ** 2 + (y - 0.1) ** 2)

            if d < 0.05:
                return nan
            return fun.func(x_vec)

        opt_problem.objective._func = wrapped_fun
        opt_problem.stop_if_nan = False

        settings = {"max_iter": 10000, "rhobeg": 0.1, "rhoend": 1e-6}

        opt_result = execute_algo(
            opt_problem, algo_name="PDFO_COBYLA", algo_type="opt", **settings
        )

        obj_history = opt_problem.database.get_function_history("rosen")

        is_nan = any(isnan(obj_history))
        assert is_nan
        assert opt_result.f_opt < 1e-3

    def test_nan_handling_2(self):
        """Test that an occurence of NaN value in the objective function does not stop
        the optimizer.

        In this test, all the values of x>0.7 are not realizable. The optimum is then
        expected for x[0] ~= 0.7
        """
        opt_problem = Rosenbrock()

        fun = copy(opt_problem.objective)

        def wrapped_fun(x_vec):
            x = x_vec[0]

            if x > 0.7:
                return nan
            return fun.func(x_vec)

        opt_problem.objective._func = wrapped_fun
        opt_problem.stop_if_nan = False

        settings = {"max_iter": 10000, "rhobeg": 0.1, "rhoend": 1e-6}

        opt_result = execute_algo(
            opt_problem, algo_name="PDFO_COBYLA", algo_type="opt", **settings
        )

        obj_history = opt_problem.database.get_function_history("rosen")

        is_nan = any(isnan(obj_history))
        assert is_nan
        assert pytest.approx(opt_result.x_opt[0], rel=1e-3) == 0.7

    def test_xtol_ftol_activation(self):
        def run_pb(settings):
            design_space = DesignSpace()
            design_space.add_variable(
                "x1", 2, DesignSpace.DesignVariableType.FLOAT, -1.0, 1.0, 0.0
            )
            problem = OptimizationProblem(design_space)
            problem.objective = MDOFunction(rosen, "Rosenbrock", "obj", rosen_der)
            res = OptimizationLibraryFactory().execute(
                problem, algo_name="PDFO_COBYLA", **settings
            )
            return res, problem

        for tol_name in (
            BaseOptimizationLibrary._F_TOL_ABS,
            BaseOptimizationLibrary._F_TOL_REL,
            BaseOptimizationLibrary._X_TOL_ABS,
            BaseOptimizationLibrary._X_TOL_REL,
        ):
            res, pb = run_pb({tol_name: 1e10})
            assert tol_name in res.message
            # Check that the criteria is activated as ap
            assert len(pb.database) == 3


def get_options(algo_name):
    """
    :param algo_name:
    """
    return {"max_iter": 10000, "rhobeg": 0.3, "rhoend": 1e-6}


def get_pb_options(pb_name):
    """
    :param algo_name:
    """
    if pb_name == "Power2":
        return {"initial_value": 0.0}
    return {}


suite_tests = OptLibraryTestBase()
for test_method in suite_tests.generate_test("PDFOOpt", get_options, get_pb_options):
    setattr(TestPDFO, test_method.__name__, test_method)


def test_library_name():
    """Check the library name."""
    assert PDFOOpt.LIBRARY_NAME == "PDFO"
