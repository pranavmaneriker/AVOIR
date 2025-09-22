import unittest

from AVOIR.dsl.grammar.numerical import *
from AVOIR.dsl.grammar.expectation import *
from AVOIR.dsl.grammar.specification import SpecificationThreshold, Specification

_DELTA_BOUND = 0.02


class TestSpecificationThreshold(unittest.TestCase):
    def test_specification_threshold(self):
        expression = create_constant(2) * create_variable("x")
        expression = Expectation(expression)
        threshold = create_constant(1)
        spec_threshold = SpecificationThreshold(expression, threshold, NumericalOperator.greater_than)
        spec_threshold.observe({
            "x": 1
        })
        spec_threshold.observe({
            "x": 2
        })
        self.assertTrue(spec_threshold.eval(), "E[2x] > 1, with observation x=1, 2 is true")

    def test_less_than_threshold(self):
        expression = create_constant(2) * create_variable("x")
        expression = Expectation(expression)
        threshold = create_constant(1)
        spec_threshold = SpecificationThreshold(expression, threshold, NumericalOperator.less_than)
        spec_threshold.observe({
            "x": 1
        })
        spec_threshold.observe({
            "x": 2
        })
        self.assertFalse(spec_threshold.eval(), "E[2x] < 1, with observation x=1, 2 is false")

    def test_equality_threshold(self):
        expression = create_constant(2) * create_variable("x")
        expression = Expectation(expression)
        threshold = create_constant(3)
        spec_threshold = SpecificationThreshold(expression, threshold, NumericalOperator.equality)
        spec_threshold.observe({
            "x": 1
        })
        spec_threshold.observe({
            "x": 2
        })
        self.assertTrue(spec_threshold.eval(), "E[2x] == 3, with observation x=1, 2 is true")


class TestSpecification(unittest.TestCase):
    def test_specification(self):
        spec_1 = (Expectation(create_constant(1) / create_variable("x")) > 1)
        spec_2 = (Expectation(create_constant(1) / create_variable("x")) > 0)
        spec = (spec_2 & spec_1)
        spec.observe({
            "x": 2
        })
        self.assertFalse(spec.eval(), "E[1/x] > 1 & E[1/x] > 0 is false with observation x=2")

    def test_specification_update(self):
        spec_1 = (Expectation(create_constant(1) / create_variable("x")) > 1)
        spec_2 = (Expectation(create_constant(1) / create_variable("x")) > 0)
        spec = (spec_2 & spec_1)
        spec.observe({
            "x": 2
        }, call_id=0, observation_key="key")
        self.assertFalse(spec.eval(), "E[1/x] > 1 & E[1/x] > 0 is false with observation x=2")
        spec.observe({
            "x": 0.5
        }, call_id=1, observation_key="key")
        self.assertTrue(spec.eval(), "E[1/x] > 1 & E[1/x] > 0 is true with observation x=0.5")

    def test_boolean_operator_specification(self):
        expectation_term_1 = Expectation(create_variable('x') + 3)
        expectation_term_2 = Expectation(create_variable('y') * create_variable('x') - 3)
        binary_spec: Specification = (expectation_term_1 > 0) & (expectation_term_2 > 0)
        vals = [{'x': 2, 'y': 2},
                {'x': 2, 'y': 2},
                {'x': 1, 'y': 3},
                {'x': 2, 'y': 3}]
        for idx, val in enumerate(vals):
            binary_spec.observe(val, call_id=idx)
        bounded_eval = binary_spec.eval_bounded_at_delta(0.1)
        self.assertLessEqual(bounded_eval.bound_delta, 0.9,
                             msg="(E[X + 2] > 0) & (E[XY - 3 > 0]) with (X, Y) = [(2, 2), (2, 2), (1, 3), (2, 3) "
                                 " should be true with some probability")

def suite():
    mysuite = unittest.TestSuite()
    mysuite.addTests([
        TestSpecificationThreshold("test_specification_threshold"),
        TestSpecification("test_specification"),
        TestSpecification("test_specification_update"),
        TestSpecificationThreshold("test_less_than_threshold"),
        TestSpecificationThreshold("test_equality_threshold"),
        TestSpecification('test_boolean_operator_specification')
    ])
    return mysuite


def test():
    test_suite = suite()
    runner = unittest.TextTestRunner()
    runner.run(test_suite)