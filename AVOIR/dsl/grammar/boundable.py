from typing import List
import logging

import pyomo.environ as pyo


from .bound_utils import generate_eps, generate_delta

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(message)s')


class OptimizationProblem:
    def __init__(self):
        self._m = None
        self._solver = pyo.SolverFactory("mindtpy")
        self._var_suffixes_E: List[str] = []
        self._suffix_index_map_E = {}
        self._var_suffixes_ET: List[str] = []
        self._suffix_index_map_ET = {}
        self._is_problem_ready = False
        self._instance = None
        self._bindings_E = {}
        self._bindings_n = {}
        self._bindings_ET = {}
        self.prev_constraints_len = 0
        self._can_optimize_at_step = True

    def get_num_E(self):
        return len(self._var_suffixes_E)

    def get_E_var_index(self, suffix):
        return self._suffix_index_map_E.get(suffix)

    def create_base_opt_model(self):
        self._m = pyo.AbstractModel()
        m = self._m

        m.I = pyo.RangeSet(len(self._var_suffixes_E))
        m.J = pyo.RangeSet(len(self._var_suffixes_ET))
        m.delta = pyo.Var(m.I , within=pyo.NonNegativeReals, bounds=(0.0, 1.0))
        m.epsilon = pyo.Var(m.I, within=pyo.NonNegativeReals)
        m.n = pyo.Param(m.I)
        m.E = pyo.Param(m.I)
        m.ET = pyo.Param(m.J)
        self.add_opt_objective()

    def add_var(self, var_suffix: str):
        self._var_suffixes_E.append(var_suffix)
        self._suffix_index_map_E[var_suffix] = len(self._var_suffixes_E)

    def add_opt_objective(self):
        def obj_expression(m):
            return pyo.summation(m.delta)
        model = self._m
        model.OBJ = pyo.Objective(rule=obj_expression, sense=pyo.minimize)

    def add_constraints(self, constraints):
        m = self._instance
        for ind, c in enumerate(constraints):
            c_name = "c_{}".format(ind)
            setattr(m, c_name, c)
        self.prev_constraints_len = len(constraints)

    def add_eterm_param(self, var_suffix: str):
        self._var_suffixes_ET.append(var_suffix)
        self._suffix_index_map_ET[var_suffix] = len(self._var_suffixes_ET)

    def getE(self, id_suffix):
        index = self._suffix_index_map_E[id_suffix]
        return self._instance.E[index]

    def getET(self, id_suffix):
        index = self._suffix_index_map_ET[id_suffix]
        return self._instance.ET[index]

    def get_delta(self, id_suffix):
        index = self._suffix_index_map_E[id_suffix]
        return self._instance.delta[index]

    def get_epsilon(self, id_suffix):
        index = self._suffix_index_map_E[id_suffix]
        return self._instance.epsilon[index]

    def get_n(self, id_suffix):
        index = self._suffix_index_map_E[id_suffix]
        return self._instance.n[index]

    def init_instance(self, data):
        if self._instance is not None:
            for ind in range(self.prev_constraints_len):
                self._instance.del_component(getattr(self._instance, "c_{}".format(ind)))
                # need to cleanup because of random index issues
                # https://github.com/Pyomo/pyomo/issues/45
        self._instance = self._m.create_instance(data)

    def bind_E(self, var_suffix, val):
        index = self._suffix_index_map_E[var_suffix]
        self._bindings_E[index] = val
        logger.log(level=logging.INFO, msg="assigned E[{}] at index {} to {}".format(var_suffix, index, val))

    def bind_n(self, var_suffix, val):
        index = self._suffix_index_map_E[var_suffix]
        self._bindings_n[index] = val
        logger.log(level=logging.DEBUG, msg="assigned n[{}] at index {} to {}".format(var_suffix, index, val))

    def bind_ET(self, var_suffix, val):
        index = self._suffix_index_map_ET[var_suffix]
        self._bindings_ET[index] = val
        logger.log(level=logging.INFO, msg="assigned ET[{}] at index {} to {}".format(var_suffix, index, val))

    def generate_bindings_dict(self):
        b_E, b_ET, b_n = self._bindings_E, self._bindings_ET, self._bindings_n
        bindings_dict = {
                             "E": b_E,
                             "ET": b_ET,
                             "n": b_n
                         }
        return {None: bindings_dict}

    @property
    def is_problem_ready(self):
        return self._is_problem_ready

    def solve(self):
        self._solver.solve(self._instance)

    @property
    def var_suffixes_E(self):
        return self._var_suffixes_E

    @property
    def instance(self):
        return self._instance


class ConstrainedValue:
    def __init__(self):
        self._is_identifier_created = False
        self._identifier = None
        self._id_suffix = None
        self._is_top_level = False
        self._opt_prob: OptimizationProblem = None
        self.opt_constraints = []
        self.opt_delta = None
        self.opt_epsilon = None
        self.opt_E = None
        self.opt_n = None

    def assign_identifier(self, identifer, suffix):
        # TODO: use Numerical expresssion representation to ensure multiple occurences of same
        # term map to same expression
        self._identifier = identifer
        self._id_suffix = suffix

    def set_problem(self, opt: OptimizationProblem):
        self._opt_prob = opt

    def get_E(self):
        return self._opt_prob.getE(self._id_suffix)

    def get_ET(self):
        return self._opt_prob.getET(self._id_suffix)

    def get_delta(self):
        return self._opt_prob.get_delta(self._id_suffix)

    def get_epsilon(self):
        return self._opt_prob.get_epsilon(self._id_suffix)

    def get_n(self):
        return self._opt_prob.get_n(self._id_suffix)

    def solve(self):
        self._opt_prob.solve()

    @property
    def num_E(self):
        return self._opt_prob.get_num_E()

class ObservedProbabilisticBoolean:
    def __init__(self, val, undetermined, delta):
        self.val = val
        self.undetermined = undetermined
        self.delta = delta


class ObservedBoundedValue:
    def __init__(self, val: float, epsilon: float, delta: float):
        self.val: float = val
        self.epsilon: float = epsilon
        self.delta: float = delta


class ProbabilisticBoolean:
    def __init__(self):
        self._bool_val = None
        self._fail_prob = 1.0
        self._prob_val = None
        self._undetermined = True
        self.bounded_observations = {} # TODO: Merge with observations, and allow both computations in one pass


    @property
    def bound_val(self):
        return self._prob_val

    @property
    def bound_delta(self):
        return self._fail_prob

    @property
    def undetermined(self):
        return self._undetermined

    def record_value(self, str_id):
        # TODO: this does not take call id etc into account
        self.bounded_observations[str_id] = ObservedProbabilisticBoolean(self._bool_val, self._undetermined,
                                                                         self._fail_prob)

ERROR_MSG = "Child bounds must be computed before bound {} on parent called"


class RangeBoundedValue:
    def __init__(self):
        self.bound_epsilon = None
        self.bound_delta = None
        self.bound_val = None
        self.bound_n = 0  # num obs
        self.bounded_observations = {} # TODO: Merge with observations, and allow both computations in one pass

    def compute_eps_for_known_delta(self, delta: float, n: int) -> float:
        return generate_eps(delta, n)
    
    def compute_delta_for_known_eps(self, eps: float, n: int) -> float:
        return generate_delta(eps, n)

    def record_value(self, idx):
        self.bounded_observations[idx] = ObservedBoundedValue(self.bound_val, self.bound_epsilon, self.bound_delta)
    
    def get_value_at_idx(self, idx) -> ObservedBoundedValue:
        return self.bounded_observations[idx]
