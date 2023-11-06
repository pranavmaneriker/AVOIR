from enum import Enum
import math
from typing import Dict, Union, NamedTuple, List
import logging
from .history import HistoryLoggingExpression

from .numerical import NumericalExpression, create_constant, NumericalOperator, NumericalExpressionType
from .errors import NoObservedOutcomesError
from .specification import create_base_spec
from .representation import JSONTableEncodableTreeExpression, JSONTableRow, TableRowType
from .boundable import BoundableValue, ConstrainedValue
from .bound_utils import compute_combined_eps, generate_eps_pyo, Constraint
from .type_utils import ObservationType

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(message)s')

"""
This module implements the following grammar from Fairness Aware Programming:
TODO: Add given syntax for 'given'
Spec := ETerm > c
  | Spec & Spec | Spec '|' Spec
ETerm := E[E]
  | E[H]
  | c <in> R
  | ETerm {+, -, /, x} ETerm
"""

def promote_item_to_et(item):
    if isinstance(item, ExpectationTerm):
        return item
    elif isinstance(item, Expectation):
        return ExpectationTerm(repr(item), ExpectationTermType.expectation, item, None, None)
    elif type(item) in [NumericalExpression, float, int]:
        return ExpectationTerm.from_numerical(item)
    else:
        raise ValueError("Cannot promote type ", type(item))
        
def create_numexpr_from_constant(term):
    if isinstance(term, NumericalExpression):
        if term.expression_type != NumericalExpressionType.constant:
            raise ValueError("Using non-constant numerical expressions not allowed")
        return term
    elif isinstance(term, float) or isinstance(term, int):
        return create_constant(term)
    else:
        raise NotImplementedError("Unsupported type for numerical expression", type(term))

def promote_before_op(op_func):
    def wrapper(*args):
        promoted_args = [promote_item_to_et(arg) for arg in args]
        return op_func(*promoted_args)
    return wrapper

class ExpectationTermOps:
    """ A class to capture promotion to ETerm and then ETerm operations
    """
    def __init__(self):
        pass
    
    @promote_before_op
    def __add__(self, other):
        return create_binary_expectation_term(self, other, op=NumericalOperator.add)

    @promote_before_op
    def __radd__(self, other):
        return self.__add__(other)

    @promote_before_op
    def __sub__(self, other):
        return create_binary_expectation_term(self, other, op=NumericalOperator.subtract)

    @promote_before_op
    def __rsub__(self, other):
        return (self * -1) + other

    @promote_before_op
    def __mul__(self, other):
        return create_binary_expectation_term(self, other, op=NumericalOperator.multiply)

    @promote_before_op
    def __rmul__(self, other):
        return self.__mul__(other)

    @promote_before_op
    def __truediv__(self, other):
        return create_binary_expectation_term(self, other, op=NumericalOperator.divide)

    @promote_before_op
    def __rtruediv__(self, other):
        return other / self

    def __gt__(self, other):
        return create_base_spec(promote_item_to_et(self),create_numexpr_from_constant(other), NumericalOperator.greater_than)
    
    def __lt__(self, other):
        return create_base_spec(promote_item_to_et(self), create_numexpr_from_constant(other), NumericalOperator.less_than)

    def __eq__(self, other) -> bool:
        return create_base_spec(promote_item_to_et(self), create_numexpr_from_constant(other), NumericalOperator.equality)


class Expectation(ExpectationTermOps, BoundableValue):
    """
    An Expectation keeps track of the expected value of a NumericalExpression.
    Through the observe(observations) function, an Expectation collects observations
    and updates the evaluations history with the computed value of :expression: given
    updates the expected value of the expression it contains.
    :param expression: NumericalExpression that Expectation calculates the expected value for
    :param condition: 
    """
    expression: NumericalExpression
    condition: NumericalExpression

    def __init__(self, expression: NumericalExpression, given: NumericalExpression = None):
        self.expression = expression
        self.condition = given
        if not given:
            self.condition = create_constant(True)
        # Note we keep track of n observations manually in Expectation, because
        # update_values needs to eval the expecation to add it to values, and
        # the eval depends on n_observations being set, so we cannot rely on
        # the default implementation in ExpectationTerm that uses the length
        # of values
        self.aggregate_value = 0.0
        self.n_observations = 0
        self.observations = []
        #JSONTableEncodableTreeExpression.__init__(self)
        BoundableValue.__init__(self)
        #ConstrainedValue.__init__(self)

    def _observe_with_children(self, observations: ObservationType):
        """
        The expected value of the expression that an Expectation contains is based
        on the evaluation of the expression on observed values.
        observe(observations:) takes variable bindings that are used to evaluate
        the contained expression and update the aggregate value of evaluations.
        """
        # unbind so we don't inherit values from previous calls
        self.condition.unbind_variables()
        self.condition.bind_variables(observations)
        # only observe values that meet the given condition
        if not self.condition.eval():
            return
        self.expression.unbind_variables()
        self.expression.bind_variables(observations) #, call_id, observation_key)
        value = self.expression.eval()
        self.aggregate_value += value
        self.n_observations += 1
        # keep track for hyper expressions TODO
        self.observations.append(observations)
        logger.debug(f"Observations {observations} used to update for {str(self)},"
                     f" leading to aggregate: {self.aggregate_value} at n: {self.n_observations}")
        #observation_key = self.update_values(call_id, value_key=observation_key)
        #return observation_key, None, None

    def observe(self, observations: ObservationType, call_id=None):
        self._observe_with_children(observations)

    def __repr__(self):
        return f"E[{str(self.expression)} | {self.condition}]"
    
    def eval(self, call_id=None) -> float:
        if self.n_observations == 0:
            raise NoObservedOutcomesError(
                ("Cannot provided an expected value with no outcomes "
                 f"for expression: {self.expression.symbolic_rep}"
                 f" under condition: {self.condition.symbolic_rep}")
            )
        logger.debug(
            f"EXPECTATION OF {self} is {self.aggregate_value / self.n_observations}")

        return self.aggregate_value / self.n_observations

    def eval_bounded_at_delta(self, delta, call_id=None):
        try:
            my_delta = delta
            if not isinstance(delta, float):
                # deltas must be captured from delta dict
                my_suffix = self._id_suffix
                my_delta = delta.get(my_suffix, 1.0)

            self._bound_val = self.eval(call_id)
            self._bound_n = self.n_observations
            eps = self.compute_eps_for_known_delta(my_delta, self.n_observations)
            self._bound_epsilon = eps
            self._bound_delta = my_delta
        except NoObservedOutcomesError:
            delta = 1.0
            eps = math.inf
            self._bound_delta = delta
            self._bound_epsilon = eps
        self.record_value(call_id)
        return self
    
    def eval_bounded_at_eps(self, eps, call_id=None):
        raise NotImplementedError

    ######

    def ensure_identifier_created(self):
        self._is_identifier_created = True

    def ensure_base_model_created(self):
        opt_prob = self._opt_prob
        if opt_prob is None:
            raise ValueError("Base Expecatation did not have opt prob set when model has to be created")
        self._opt_prob.add_var(self._id_suffix)

    def construct_opt_problem(self, parent=None):
        # TODO: simplify/refractor
        self.opt_delta = self.get_delta()
        self.opt_epsilon = self.get_epsilon()
        self.opt_E = self.get_E()
        self.opt_n = self.get_n()
        self.opt_constraints = []
        self.opt_constraints.append(Constraint(expr=(self.opt_epsilon == generate_eps_pyo(self.opt_delta, self.opt_n))))

    ## overrides(ExpectationTerm:JSONTableEncodableTreeExpression)
    #@property
    #def children(self) -> List[JSONTableEncodableTreeExpression]:
    #    return [self.expression] + [
    #        self.condition
    #    ] if self.condition.expression_type != NumericalExpressionType.constant else []

    ## overrides(ExpectationTerm:JSONTableEncodableTreeExpression)
    #@property
    #def row_representation(self) -> JSONTableRow:
    #    return JSONTableRow(
    #        row_type=TableRowType.expectation
    #    )


    # overrides(ExpectationTerm)
    # TODO refactor observation to be a higher level function that calls a protocol method. Not sure

class ExpectationTermType(Enum):
    binary = 1
    expectation = 2
    constant = 3

class ExpectationTerm(ExpectationTermOps, BoundableValue):#(JSONTableEncodableTreeExpression, , ConstrainedValue):
    """
    An ExpectationTerm is a binary expression that captures binary operations of 
    probabilistic expressions. A base expectation term is either an Expectation or 
    a constant numerical expression.
    :param symbolic_rep: string representation of the ExpectationTerm
    :param left_child: Could be an Expectation, Expectation, or constant NumericalExpression
    :param right_child: Either an ExpectationTerm or None
    """

    def __init__(self, symbolic_rep: str, 
                 term_type: ExpectationTermType,
                 left_child: 'ExpectationTerm',
                 right_child: 'ExpectationTerm', op: NumericalOperator):
        self.term_type = term_type
        self.symbolic_rep = symbolic_rep
        self.left_child = left_child
        self.right_child = right_child
        self.op = op
        #JSONTableEncodableTreeExpression.__init__(self)
        #BoundableValue.__init__(self)
        #ConstrainedValue.__init__(self)

    def observe(self, observations: ObservationType, call_id=None): #, call_id=None, observation_key=None, with_bounds=False, delta=0.05):
        if self.term_type in [ExpectationTermType.expectation, ExpectationTermType.binary]:
            obs = self.left_child.observe(observations, call_id=call_id) #, call_id, observation_key, with_bounds, delta)
        if self.term_type == ExpectationTermType.binary:
            obs = self.right_child.observe(observations, call_id=call_id)
        #return obs[0] if obs is not None else None

    def eval(self, call_id=None) -> float:
        left_val = self.left_child.eval(call_id)
        val = left_val
        if self.term_type == ExpectationTermType.binary:
            right_val = self.right_child.eval(call_id)
            val = NumericalOperator.functions()[self.op](left_val, right_val)
        # self.logger.debug(f"EXPECTATION OF {self} is {val}")
        return val
    

    def __repr__(self):
        return self.symbolic_rep

    def eval_bounded_at_delta(self, delta, call_id=None):
        self._bound_val = self.eval(call_id)
        self.left_child.eval_bounded_at_delta(delta, call_id)
        self.right_child.eval_bounded_at_delta(delta, call_id)
        combined_eps = compute_combined_eps(self)
        self.record_value(call_id)
        return combined_eps

    @classmethod
    def from_numerical(cls, term: Union[NumericalExpression, float, int]) -> NumericalExpression:
        term2 = create_numexpr_from_constant(term)
        return cls(repr(term2), ExpectationTermType.constant, term2, None, None)
    



def create_binary_expectation_term(left: ExpectationTerm, right: ExpectationTerm, op: NumericalOperator):
    op_symbol = NumericalOperator.symbols()[op]
    return ExpectationTerm(
        symbolic_rep=f"{left} {op_symbol} {right}",
        term_type=ExpectationTermType.binary,
        left_child=left,
        right_child=right,
        op=op
    )

    #def add_child_bounds(self, left, right, l_value_key, r_value_key):
    #    self.left_child.add_bounds(left, value_key=l_value_key)
    #    self.right_child.add_bounds(right, value_key=r_value_key)


    #@set_params_in_opt
    #@HistoryLoggingExpression.cached_eval


    # @HistoryLoggingExpression.cached_eval_bounded
    # def eval_bounded(self, epsilon, call_id=None) -> ProbabilisticEvaluation:
    #    left_child = self.left_child
    #    right_child = self.right_child
    #    op = self.op
    #    return eval_base_expectation_term_bound(left_child, right_child, op, epsilon)

    # Note that the `ETerm {+, −, ÷, ×} ETerm` type is defined through
    # operator overloading

    # overrides(JSONTableEncodableTreeExpression)
    #@property
    #def children(self) -> List[JSONTableEncodableTreeExpression]:
    #    return [self.left_child, self.right_child]

    ## overrides(JSONTableEncodableTreeExpression)
    #@property
    #def row_representation(self) -> JSONTableRow:
    #    return JSONTableRow(
    #        row_type=TableRowType.expectation_term
    #    )

    #def ensure_identifier_created(self):
    #    l = self.left_child
    #    r = self.right_child
    #    l.assign_identifier("{}_{}".format(self._identifier, "1"),
    #                        "{}_{}".format(self._id_suffix, "1"))
    #    l.ensure_identifier_created()

    #    r.assign_identifier("{}_{}".format(self._identifier, "2"),
    #                        "{}_{}".format(self._id_suffix, "2"))
    #    r.ensure_identifier_created()
    #    self._is_identifier_created = True

    #def ensure_base_model_created(self):
    #    opt_prob = self._opt_prob
    #    if opt_prob is None:
    #        raise ValueError("Base Expecatation did not have opt prob set when model has to be created")
    #    self._opt_prob.add_eterm_param(self._id_suffix)
    #    l = self.left_child
    #    r = self.right_child
    #    l.set_problem(self._opt_prob)
    #    r.set_problem(self._opt_prob)
    #    l.ensure_base_model_created()
    #    r.ensure_base_model_created()

    #def construct_opt_problem(self, parent=None):
    #    l = self.left_child
    #    r = self.right_child
    #    l.construct_opt_problem(parent=self)
    #    r.construct_opt_problem(parent=self)
    #    self.opt_delta = l.opt_delta + r.opt_delta
    #    self.opt_constraints = list(l.opt_constraints)
    #    self.opt_constraints.extend(list(r.opt_constraints))
    #    self.opt_E = self.get_ET()
    #    # TODO cleanup
    #    if self.op == NumericalOperator.add or self.op == NumericalOperator.subtract:
    #        self.opt_epsilon = l.opt_epsilon + r.opt_epsilon
    #    else:
    #        eps_l = l.opt_epsilon
    #        E_l = l.opt_E
    #        eps_r = r.opt_epsilon
    #        E_r = r.opt_E
    #        if self.op == NumericalOperator.divide:
    #            eps_inv = eps_r/(E_r * (E_r - eps_r))
    #            E_inv = 1/E_r

    #            E_r = E_inv
    #            eps_r = eps_inv
    #        self.opt_epsilon = E_l * eps_r + E_r * eps_l + eps_r * eps_l

    #def record_value(self, str_id):
    #    super().record_value(str_id)
    #    if not isinstance(self, Expectation):
    #        self.bounded_observations.update(self.left_child.bounded_observations)
    #        self.bounded_observations.update(self.right_child.bounded_observations)

#########################################

def set_params_in_opt(instance_func):
    def wrapper(self, *args, **kwargs):
        val = instance_func(self, *args, **kwargs)
        if self._opt_prob is not None:
            if isinstance(self, Expectation):
                self._opt_prob.bind_E(self._id_suffix, val)
                self._opt_prob.bind_n(self._id_suffix, self.n_observations)
            else:
                self._opt_prob.bind_ET(self._id_suffix, val)
        else:
            if self._opt_prob is not None:
                self._opt_prob.can_optimize_at_step = False
        return val
    return wrapper



