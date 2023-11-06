from __future__ import annotations, division

from typing import TYPE_CHECKING
import math
import logging

from pyomo.environ import log as pyo_log, sqrt as pyo_sqrt
from pyomo.environ import LogicalConstraint, lor as pyo_or, land as pyo_and, Constraint, implies as pyo_implies

if TYPE_CHECKING:
    from .expectation import Expectation, ExpectationTerm
    from .boundable import BoundableValue
    from .specification import SpecificationThreshold
from .numerical import NumericalOperator


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
_CONS_EPSILON = 1e-6



def _generate_eps_bound_adaptive_hoeffding_pyo(delta: float, n: int):
    return pyo_sqrt((0.6 * pyo_log(pyo_log(n)/math.log(1.1) + 1) + (5 / 9) * (math.log(24) - pyo_log(delta))) / n)


def _generate_eps_bound_adaptive_hoeffding(delta: float, n: int):
    return math.sqrt((0.6 * math.log(math.log(n, 1.1) + 1) + (5 / 9) * (math.log(24/delta))) / n)

def _generate_delta_bound_adaptive_hoeffding(eps: float, n: int):
    return 24 * math.exp((9 / 5) * (0.6 * math.log(math.log(n, 1.1) + 1) - n * eps * eps))


generate_eps = _generate_eps_bound_adaptive_hoeffding
generate_delta = _generate_delta_bound_adaptive_hoeffding
generate_eps_pyo = _generate_eps_bound_adaptive_hoeffding_pyo


def eps_sum(left_val: BoundableValue, right_val: BoundableValue):
    eps_tot = left_val.bound_epsilon + right_val.bound_epsilon
    del_tot = left_val.bound_delta + right_val.bound_delta
    return eps_tot, del_tot


eps_sub = eps_sum


def eps_mul(left_val: BoundableValue, right_val: BoundableValue):
    leval = left_val.bound_val
    rval = right_val.bound_val
    leps = left_val.bound_epsilon
    reps = right_val.bound_epsilon
    eps_tot = leps * rval + reps * leval + leps * reps
    del_tot = left_val.bound_delta + right_val.bound_delta
    return eps_tot, del_tot


def eps_div_comp(left_val: BoundableValue, right_val: BoundableValue):
    del_tot = left_val.bound_delta + right_val.bound_delta
    leval = left_val.bound_val
    rval = right_val.bound_val
    leps = left_val.bound_epsilon
    reps = right_val.bound_epsilon
    if rval > reps:
        eps_inv = reps/(rval * (rval - reps))
        rval_inv = 1.0/rval
        eps_tot = leps * rval_inv + eps_inv * leval + leps * eps_inv
        return eps_tot, del_tot
    else:
        return math.inf, 1.0


def compute_combined_eps(expectation_term: ExpectationTerm):
    op = expectation_term.op
    eps, delta = {
        NumericalOperator.add: eps_sum,
        NumericalOperator.subtract: eps_sub,
        NumericalOperator.multiply: eps_mul,
        NumericalOperator.divide: eps_div_comp
    }[op](expectation_term.left_child, expectation_term.right_child)

    delta = min(1.0, delta)
    expectation_term._bound_epsilon = eps
    expectation_term._bound_delta = delta
    return expectation_term


def or_constraints(const_list1, const_list2):
    return LogicalConstraint(expr=pyo_or(const_list1, const_list2))


def and_constraint(const_list1, const_list2):
    return LogicalConstraint(expr=pyo_and(const_list1, const_list2))


def equality_constraint(var):
    return Constraint(expr=(var == 0))


def implies_constraint(a, b):
    return LogicalConstraint(expr=pyo_implies(a, b))


def gt_constraint(a, b):
    return Constraint(expr=(a >= b + _CONS_EPSILON))


def lt_constraint(a, b):
    return Constraint(expr=(a <= b - _CONS_EPSILON))