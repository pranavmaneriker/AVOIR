from __future__ import annotations

import math
import operator
import pdb
from enum import Enum
from typing import Union, Dict, List, TYPE_CHECKING

from .representation import JSONTableEncodableTreeExpression, HistoryLoggingExpression, JSONTableRow, TableRowType
from .numerical import NumericalExpression, NumericalOperator, NumericalExpressionType
from .boundable import ProbabilisticBoolean, ConstrainedValue, OptimizationProblem
from .bound_utils import or_constraints, and_constraint, equality_constraint, \
    gt_constraint, lt_constraint, implies_constraint

if TYPE_CHECKING:
    from expectation import ExpectationTerm

    

class SpecificationThreshold:#(JSONTableEncodableTreeExpression, ProbabilisticBoolean, ConstrainedValue):
    # Implements Eterm > c

    def __init__(self, expectation_term, threshold, op,
                 threshold_prob: float = 0.9):
        """
        Base term that results from promoting
        :param expectation_term:
        :param threshold:
        """
        self.threshold: NumericalExpression = threshold
        self.expectation_term: ExpectationTerm = expectation_term
        self.operator: NumericalOperator = op
        self.threshold_prob: float = threshold_prob
        assert self.threshold.expression_type == NumericalExpressionType.constant, "Threshold must be constant"

        #JSONTableEncodableTreeExpression.__init__(self)
        #ProbabilisticBoolean.__init__(self)
        #ConstrainedValue.__init__(self)

    #@HistoryLoggingExpression.cached_eval
    def eval(self, call_id=None):
        eval_val = self.expectation_term.eval(call_id)
        threshold = self.threshold.eval(call_id)
        self.threshold.unbind_variables()  # unbind after eval
        val = NumericalOperator.functions()[self.operator](eval_val, threshold)
        # required since comparison for np.array returns bool_ which is not serializable
        val = bool(val)
        return val

    def observe(self, vals: Dict[str, any], call_id=None, observation_key=None, with_bounds=False, delta=0.05):
        #self.undo_previous_observations(with_key=observation_key)
        self.expectation_term.observe(vals, call_id=call_id)
        self.threshold.unbind_variables()
        self.threshold.bind_variables(vals, call_id=call_id)
        #observation_key = self.update_values(call_id, value_key=observation_key)
        #return observation_key
    #def eval_bounded_at_delta(self, delta, call_id=None):
    #    eval_obs = self.expectation_term.eval_bounded_at_delta(delta, call_id)
    #    threshold = self.threshold.eval(call_id)
    #    eval_obs_upper = eval_obs.bound_val + eval_obs.bound_epsilon
    #    eval_obs_lower = eval_obs.bound_val - eval_obs.bound_epsilon
    #    comp_op_func = NumericalOperator.functions()[self.operator]
    #    opposite_op_func = NumericalOperator.opposite_function()[self.operator]
    #    perp = True
    #    if comp_op_func(eval_obs.bound_val, threshold):
    #        self._bool_val = True
    #        if comp_op_func(eval_obs_upper, threshold) and comp_op_func(eval_obs_lower, threshold):
    #            self._prob_val = True
    #            self._fail_prob = eval_obs.bound_delta
    #            perp = False
    #            # the mean value satisfied the inequality
    #    else:
    #        self._bool_val = False
    #        if opposite_op_func(eval_obs_upper, threshold) and opposite_op_func(eval_obs_lower, threshold):
    #            self._prob_val = False
    #            self._fail_prob = eval_obs.bound_delta
    #            perp = False

    #    if perp:
    #        self._undetermined = True
    #        self._fail_prob = 1.0
    #    else:
    #        self._undetermined = False

    #    self.record_value(self.row_id)  # NOTE: Move to observe, fix with uuid from JSONTable
    #    return self


    def __repr__(self):
        return f"({str(self.expectation_term)} {NumericalOperator.symbols()[self.operator]} {self.threshold})"

    # overrides(JSONTableEncodableTreeExpression)
    #@property
    #def children(self) -> List[JSONTableEncodableTreeExpression]:
    #    return [self.expectation_term, self.threshold]

    ## overrides(JSONTableEncodableTreeExpression)
    #@property
    #def row_representation(self) -> JSONTableRow:
    #    return JSONTableRow(
    #        row_type=TableRowType.specification_threshold
    #    )

    #def ensure_identifiers_created(self, is_top_level=False):
    #    self.expectation_term.assign_identifier("E_{}".format(self._id_suffix),
    #                                            self._id_suffix)

    #    self.expectation_term.ensure_identifier_created()
    #    self._is_identifier_created = True

    #def ensure_base_model_created(self):
    #    self.expectation_term.set_problem(self._opt_prob)
    #    self.expectation_term.ensure_base_model_created()

    #def construct_opt_problem(self, parent=None):
    #    self.expectation_term.construct_opt_problem(parent=self)
    #    self.opt_delta = self.expectation_term.opt_delta
    #    self.opt_epsilon = self.expectation_term.opt_epsilon
    #    self.opt_constraints = list(self.expectation_term.opt_constraints)
    #    if self.operator == NumericalOperator.equality:
    #        self.opt_constraints.append(equality_constraint(self.opt_epsilon))
    #    else:
    #        eterm = self.expectation_term.opt_E
    #        thresh_val = self.threshold.eval()
    #        # E > c => E - \eps > c
    #        if eterm > thresh_val:
    #            self.opt_constraints.append(gt_constraint(eterm - self.opt_epsilon, thresh_val))
    #        elif eterm < thresh_val:
    #            self.opt_constraints.append(lt_constraint(eterm + self.opt_epsilon, thresh_val))

    #def record_value(self, str_id):
    #    super().record_value(str_id)
    #    self.bounded_observations.update(self.expectation_term.bounded_observations)


class SpecificationOperator(Enum):
    op_or = 1
    op_and = 2

    @classmethod
    def functions(cls):
        return {
            SpecificationOperator.op_and: operator.and_,
            SpecificationOperator.op_or: operator.or_
        }


class SpecificationType(Enum):
    base_spec = 1  # ETerm > c
    binary_spec = 2  # Spec &/| Spec


class Specification:#(JSONTableEncodableTreeExpression, ProbabilisticBoolean, ConstrainedValue):

    def __init__(self, spec_type: SpecificationType = None,
                 left_child: Union['Specification',
                                   SpecificationThreshold] = None,
                 right_child: Union['Specification',
                                    SpecificationThreshold] = None,
                 op: SpecificationOperator = None):
        """
        Create a specification
        :param spec_type:
        :param left_child:
        :param right_child:
        :param op:
        """
        self.spec_type = spec_type
        self.left_child = left_child
        self.right_child = right_child
        self.spec_op = op
        #JSONTableEncodableTreeExpression.__init__(self)
        #ProbabilisticBoolean.__init__(self)
        #ConstrainedValue.__init__(self)

    def observe(self, vals: Dict[str, any], call_id=None, observation_key=None, with_bounds=False, delta=0.05):
        # TODO possibly observe, and then observe bounds?
        #self.undo_previous_observations(with_key=observation_key)
        if self.left_child:
            self.left_child.observe(vals, call_id, observation_key, with_bounds, delta)
        if self.right_child:
            self.right_child.observe(vals, call_id, observation_key, with_bounds, delta)
        #observation_key = self.update_values(call_id, value_key=observation_key)
        #return observation_key

    #def unobserve(self, call_id=None, observation_key=None):
    #    self.undo_previous_observations(with_key=observation_key)
    #    if self.spec_type == SpecificationType.binary_spec:
    #        self.left_child.unobserve(call_id, observation_key)
    #        self.right_child.unobserve(call_id, observation_key)
    #    self.update_values(call_id, value_key=observation_key)

    #def undo_previous_observations(self, with_key):
    #    self.retire_values(with_key)

    def __and__(self, other: "Specification") -> "Specification":
        # operator: &
        return create_specification(self, other, SpecificationOperator.op_and)

    def __or__(self, other: "Specification") -> "Specification":
        # operator |
        return create_specification(self, other, SpecificationOperator.op_or)

    #@HistoryLoggingExpression.cached_eval
    def eval(self, call_id=None):
        # Note: __bool__ not used because it gets invoked by `if self.left_child:` when recursive binding with observe
        l = self.left_child
        r = self.right_child
        if self.spec_type == SpecificationType.base_spec:
            val = l.eval(call_id)
        else:
            val = SpecificationOperator.functions()[self.spec_op](
                l.eval(call_id), r.eval(call_id))
        val = bool(val)
        return val

    # @HistoryLoggingExpression.cached_eval_bounded
    #def eval_bounded_at_delta(self, delta, call_id=None) -> ProbabilisticBoolean:
    #    # Return max probability with which eval == self.eval()
    #    l = self.left_child
    #    r = self.right_child
    #    if self.spec_type == SpecificationType.base_spec:
    #        # if base spec, return the min prob that SpecificationThreshold eval fails
    #        rec_val = l.eval_bounded_at_delta(delta, call_id)
    #        self._undetermined = rec_val.undetermined
    #        self._bool_val = rec_val._bool_val
    #        self._prob_val = rec_val.bound_val
    #        self._fail_prob = rec_val.bound_delta
    #    else:
    #        l_eb = l.eval_bounded_at_delta(delta, call_id)
    #        r_eb = r.eval_bounded_at_delta(delta, call_id)
    #        # f_prob = l_eb.f_prob + r_eb.f_prob
    #        val = SpecificationOperator.functions()[self.spec_op](
    #            l_eb._bool_val, r_eb._bool_val)
    #        self._bool_val = val
    #        self._fail_prob = min(1.0, r_eb._fail_prob + l_eb._fail_prob)
    #        # move this to spec op class for cleanup TODO
    #        if self.spec_op == SpecificationOperator.op_or:
    #            if l_eb._undetermined and r_eb._undetermined:
    #                self._fail_prob = 1.0
    #                self._undetermined = True
    #            else:
    #                self._undetermined = False
    #                self._prob_val = val
    #        else:
    #            if l_eb._undetermined or r_eb._undetermined:
    #                self._fail_prob = 1.0
    #                self._undetermined = True
    #            else:
    #                self._undetermined = False
    #                self._prob_val = val
    #    self.record_value(self.row_id)
    #    return self

    #def ensure_identifiers_created(self, is_top_level=False):
    #    # creates the model and identifiers associated with each term in optimizer
    #    if not self._is_identifier_created:  # only called once
    #        l = self.left_child
    #        r = self.right_child
    #        # identifiers are generated with the model, so this can only happen if
    #        # model has not been created
    #        # top level identifier is psi
    #        # each child identifier is {parent}_1 or {parent}_2
    #        # specification threshold identifier is {parent}_T
    #        if is_top_level:
    #            self.assign_identifier("psi_1", "1")
    #            self._is_top_level = True

    #        if self.spec_type == SpecificationType.base_spec:
    #            l.assign_identifier("{}_T".format(self._identifier), self._id_suffix)
    #            l.ensure_identifiers_created()
    #        else:
    #            l.assign_identifier("{}_{}".format(self._identifier, "1"),
    #                                "{}_1".format(self._id_suffix))
    #            l.ensure_identifiers_created()

    #            r.assign_identifier("{}_{}".format(self._identifier, "2"),
    #                                "{}_2".format(self._id_suffix))
    #            r.ensure_identifiers_created()
    #        self._is_identifier_created = True

    #def ensure_base_model_created(self, is_top_level=False):
    #    opt_prob = self._opt_prob
    #    if opt_prob is None:
    #        opt_prob = OptimizationProblem()
    #        self.set_problem(opt_prob)

    #    l = self.left_child
    #    r = self.right_child
    #    l.set_problem(opt_prob)
    #    l.ensure_base_model_created()

    #    if self.spec_type != SpecificationType.base_spec:
    #        r.set_problem(opt_prob)
    #        r.ensure_base_model_created()

    #    if is_top_level:
    #        self._opt_prob.create_base_opt_model()

    #def construct_opt_problem(self, parent=None):
    #    l = self.left_child
    #    l.construct_opt_problem(parent=self)
    #    # TODO: Cleanup this
    #    if self.spec_type == SpecificationType.base_spec:
    #        self.opt_constraints = l.opt_constraints
    #        self.opt_delta = l.opt_delta
    #        self.opt_epsilon = l.opt_epsilon
    #    else:
    #        r = self.right_child
    #        r.construct_opt_problem(parent=self)
    #        if self.spec_op == SpecificationOperator.op_or:
    #            self.opt_constraints = [or_constraints(l.opt_constraints, r.opt_constraints)]
    #        else:
    #            self.opt_constraints = list(l.opt_constraints)
    #            self.opt_constraints.extend(r.opt_constraints)
    #    if parent is None:
    #        self._opt_prob.add_constraints(self.opt_constraints)
    #        # set the constraints of problem

    #    self._opt_prob._is_problem_ready = True

    #def prepare_for_opt(self, is_top_level=False):
    #    self.ensure_identifiers_created(is_top_level)
    #    self.ensure_base_model_created(is_top_level)

    #def eval_bounded_with_constraints_at_delta(self, delta, call_id=None):
    #    self.eval(call_id)  # TODO call_id support
    #    opt_prob = self._opt_prob
    #    opt_prob.init_instance(opt_prob.generate_bindings_dict())
    #    self.construct_opt_problem()

    #    opt_prob.solve()
    #    return self
    #    #self.ensure_opt_created
    #    # l = self.left_child
    #    # r = self.right_child

    #def __repr__(self):
    #    if self.spec_type == SpecificationType.base_spec:
    #        return str(self.left_child)
    #    else:
    #        op_rep = {
    #            SpecificationOperator.op_or: "|",
    #            SpecificationOperator.op_and: "&"
    #        }[self.spec_op]
    #        return f"({str(self.left_child)} {op_rep} {str(self.right_child)})"

    ## overrides(JSONTableEncodableTreeExpression)
    #@property
    #def children(self) -> List[JSONTableEncodableTreeExpression]:
    #    if self.spec_type == SpecificationType.base_spec:
    #        return self.left_child.children
    #    else:
    #        return [self.left_child, self.right_child]

    ## overrides(JSONTableEncodableTreeExpression)
    #@property
    #def row_representation(self) -> JSONTableRow:
    #    return JSONTableRow(
    #        row_type=TableRowType.specification
    #    )

    #@property
    #def opt_prob(self):
    #    return self._opt_prob

    #def get_delta_bindings(self):
    #    if not self._is_top_level:
    #        raise ValueError("Illegal to get bindings from a non-top level spec")
    #    else:
    #        delta_bindings = {label: delta.value for label, delta in
    #                          zip(self.opt_prob.var_suffixes_E, self.opt_prob.instance.delta.values())}
    #        return delta_bindings

    #def get_E_bindings(self):
    #    if not self._is_top_level: # TODO: Roll up these errors into a decorator
    #        raise ValueError("Illegal to get bindings from a non-top level spec")
    #    else:
    #        E_bindings = {label: E.value for label, E in
    #                        zip(self.opt_prob.var_suffixes_E, self.opt_prob.instance.E.values())}
    #        return E_bindings

    #def get_epsilon_bindings(self):
    #    if not self._is_top_level: # TODO: Roll up these errors into a decorator
    #        raise ValueError("Illegal to get bindings from a non-top level spec")
    #    else:
    #        epsilon_bindings = {label: E.value for label, E in
    #                      zip(self.opt_prob.var_suffixes_E, self.opt_prob.instance.epsilon.values())}
    #        return epsilon_bindings

    #def record_value(self, str_id):
    #    # first record own value
    #    super().record_value(str_id)
    #    self.bounded_observations.update(self.left_child.bounded_observations)
    #    if self.spec_type != SpecificationType.base_spec:
    #        self.bounded_observations.update(self.right_child.bounded_observations)



def create_specification(spec_1: Specification, spec_2: Specification, op: SpecificationOperator) -> Specification:
    """Semantics similar to create_expression"""
    return Specification(SpecificationType.binary_spec, spec_1, spec_2, op)


def create_base_spec(term_1: ExpectationTerm, term_2: NumericalExpression, op: NumericalOperator):
    return Specification(SpecificationType.base_spec, SpecificationThreshold(term_1, term_2, op))
