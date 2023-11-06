import unittest
import math
from typing import List, Dict

from AVOIR.dsl.grammar import numerical as nu, expectation as ex

_DELTA_BOUND = 0.05
_CHECK_DELTA = 1e-5
_PLACES = 5

class TestExpectation(unittest.TestCase):
    def test_expectation(self):
        expression = nu.create_variable("x")
        expectation = ex.Expectation(expression)
        expectation.observe({
            "x": 1
        })
        expectation.observe({
            "x": 3
        })
        self.assertEqual(expectation.eval(), 2, "expectation of x = 1,3 should be 2")

    #def test_expectation_unobserve(self):
    #    expression = nu.create_variable("x")
    #    expectation = ex.Expectation(expression)
    #    expectation.observe({
    #        "x": 1
    #    }, call_id=0, observation_key="key")
    #    expectation.observe({
    #        "x": 3
    #    }, call_id=1)
    #    self.assertEqual(expectation.eval(), 2, "expectation of x = 1,3 should be 2")
    #    expectation.unobserve(call_id=2, observation_key="key")
    #    self.assertEqual(expectation.eval(), 3, "expectation of x = 3 should be 3")

    def test_conditional_expectation(self):
        expression = nu.create_variable("x")
        condition = nu.create_variable("x") > 10
        expectation = ex.Expectation(expression, given=condition)
        expectation.observe({
            "x": 1
        })
        expectation.observe({
            "x": 3
        })
        expectation.observe({
            "x": 41
        })
        expectation.observe({
            "x": 43
        })
        self.assertEqual(expectation.eval(), 42, "expectation of x | x > 10, after seeing 1,3,41,43 should be 42")

    #def test_expectation_bound_at_delta_constant(self):
    #    expression = nu.create_variable("x")
    #    condition = nu.create_variable("x") > 0.8
    #    expectation = ex.Expectation(expression, given=condition)
    #    values = [0.9 for _ in range(1000)]
    #    for v in values:
    #        expectation.observe({
    #            "x": v
    #        })

    #    self.assertLessEqual(expectation.eval_bounded_at_delta(_DELTA_BOUND).bound_epsilon, 0.1,
    #                         "large number of constant values in Expectation do not have a small \\epsilon")

    #def test_expectation_bound_at_delta_monotonic_epsilon(self):
    #    expression = nu.create_variable("x")
    #    condition = nu.create_variable("x") > 0.8
    #    expectation = ex.Expectation(expression, given=condition)
    #    values = [0.9 for _ in range(100)]
    #    bound_epsilon_prev = math.inf
    #    for v in values:
    #        expectation.observe({
    #            "x": v
    #        })
    #        computed_expectation = expectation.eval_bounded_at_delta(_DELTA_BOUND)
    #        self.assertLessEqual(computed_expectation.bound_epsilon, bound_epsilon_prev,
    #                             "\\epsilon does not decrease monotonically for Expectation")

    #        bound_epsilon_prev = computed_expectation.bound_epsilon


class TestExpectationTerm(unittest.TestCase):

    @property
    def _x_vals_test_list(self):
        vals = [1, 3, 41, 43]
        return [{"x": val} for val in vals]

    @property
    def _x_vals_bern_test_list(self):
        vals = [0.8 for _ in range(100)]
        vals.extend([0.4 for _ in range(100)])
        return [{"x": val} for val in vals]

    def _create_basic_terms(self):
        condition_lt = nu.create_variable("x") < 10
        condition_gt = nu.create_variable("x") > 10
        expectation_lt = ex.Expectation(nu.create_variable("x"), given=condition_lt)
        expectation_gt = ex.Expectation(nu.create_variable("x"), given=condition_gt)
        return expectation_lt, expectation_gt

    def _create_basic_terms_bern(self):
        condition_lt = nu.create_variable("x") < 0.5
        condition_gt = nu.create_variable("x") > 0.5
        expectation_lt = ex.Expectation(nu.create_variable("x"), given=condition_lt)
        expectation_gt = ex.Expectation(nu.create_variable("x"), given=condition_gt)
        return expectation_lt, expectation_gt

    def _expression_observe_values(self, expression: ex.ExpectationTerm, vals: List[Dict]):
        for idx, val in enumerate(vals):
            expression.observe(val, call_id=idx)

    def test_binary_expectation_term(self):
        expectation_lt, expectation_gt = self._create_basic_terms()
        binary_expectation_term = expectation_gt / expectation_lt
        self._expression_observe_values(binary_expectation_term, self._x_vals_test_list)
        self.assertEqual(binary_expectation_term.eval(), 21,
                         "expectation of x when < 10 / expectation of x when > 10 should be 21")

    #def test_binary_expectation_term_unobserve(self):
    #    expectation_lt, expectation_gt = self._create_basic_terms()
    #    binary_expectation_term = expectation_gt / expectation_lt
    #    binary_expectation_term.observe({
    #        "x": 1
    #    }, call_id=0)
    #    binary_expectation_term.observe({
    #        "x": 3
    #    }, call_id=1, observation_key="key")
    #    binary_expectation_term.observe({
    #        "x": 41
    #    }, call_id=2)
    #    binary_expectation_term.observe({
    #        "x": 43
    #    }, call_id=3)
    #    self.assertEqual(binary_expectation_term.eval(), 21,
    #                     "expectation of x when < 10 / expectation of x when > 10 should be 21")
    #    binary_expectation_term.unobserve(call_id=4, observation_key="key")
    #    self.assertEqual(binary_expectation_term.eval(), 42,
    #                     "expectation of x when < 10 / expectation of x when > 10 should be 21")

    def test_binary_expectation_term_update(self):
        expectation_lt, expectation_gt = self._create_basic_terms()
        binary_expectation_term = expectation_gt / expectation_lt
        binary_expectation_term.observe({
            "x": 1
        }, call_id=0)
        binary_expectation_term.observe({
            "x": 3
        }, call_id=1)
        binary_expectation_term.observe({
            "x": 41
        }, call_id=2)
        binary_expectation_term.observe({
            "x": 43
        }, call_id=3)
        self.assertEqual(binary_expectation_term.eval(), 21,
                         "expectation of x when < 10 / expectation of x when > 10 should be 21")
        binary_expectation_term.observe({
            "x": 8
        }, call_id=4)
        self.assertEqual(binary_expectation_term.eval(), 10.5,
                         "expectation of x when < 10 / expectation of x when > 10 should be 10.5")

    def test_binary_expectation_term_with_constant(self):
        expression = nu.create_variable("x")
        binary_expectation_term = ex.Expectation(expression) * 5
        binary_expectation_term.observe({
            "x": 1
        })
        binary_expectation_term.observe({
            "x": 3
        })
        self.assertEqual(binary_expectation_term.eval(), 10,
                         "(expectation of x = 1,3) * 5 should be 10")

    def test_binary_expectation_term_with_constant_update(self):
        expression = nu.create_variable("x")
        binary_expectation_term = ex.Expectation(expression) * 5
        binary_expectation_term.observe({
            "x": 1
        }, call_id=0)
        binary_expectation_term.observe({
            "x": 3
        }, call_id=1)
        self.assertEqual(binary_expectation_term.eval(), 10,
                         "(expectation of x = 1,3) * 5 should be 10")
        binary_expectation_term.observe({
            "x": 8
        }, call_id=2)

        self.assertEqual(binary_expectation_term.eval(), 20,
                         "(expectation of x = 1,3,8) * 5 should be 20")

    def test_binary_expectation_term_with_r_constant(self):
        expression = nu.create_variable("x")
        binary_expectation_term_add = 5 + ex.Expectation(expression)
        binary_expectation_term_mul = 5 * ex.Expectation(expression)
        binary_expectation_term_sub = 5 - ex.Expectation(expression)
        binary_expectation_term_div = 5 / ex.Expectation(expression)

        for eterm in [binary_expectation_term_add,
                      binary_expectation_term_mul,
                      binary_expectation_term_sub,
                      binary_expectation_term_div]:
            eterm.observe({
                "x": 1
            })
            eterm.observe({
                "x": 3
            })
        self.assertEqual(binary_expectation_term_add.eval(), 7,
                         "5 + (expectation of x = 1,3) should be 10")
        self.assertEqual(binary_expectation_term_mul.eval(), 10,
                         "5 * (expectation of x = 1,3) should be 10")
        self.assertEqual(binary_expectation_term_sub.eval(), 3,
                         "5 - (expectation of x = 1,3) should be 10")
        self.assertEqual(binary_expectation_term_div.eval(), 2.5,
                         "5 / (expectation of x = 1,3) should be 2.5")

    #def test_eval_prob_bound_additive_given_delta(self):
    #    expectation_lt, expectation_gt = self._create_basic_terms_bern()
    #    binary_expectation_term = expectation_gt + expectation_lt
    #    self._expression_observe_values(binary_expectation_term, self._x_vals_bern_test_list)
    #    # computing E[X | X != 0.5] -
    #    # also: https://www.wolframcloud.com/obj/a2a1833f-84cf-4a2f-9103-adf232d09fe0
    #    prob_val = binary_expectation_term.eval_bounded_at_delta(delta=_DELTA_BOUND)
    #    self.assertAlmostEqual(prob_val.bound_val, 1.2, delta=_CHECK_DELTA,
    #                           msg="(expectation of x = 0.8) + (expectation of x = 0.4) should be 0.6",
    #                        )
    #    self.assertLess(prob_val.bound_epsilon, 0.5,
    #                    msg="epsilon for (expectation of x = 0.8) + (expectation of x = 0.5)"
    #                        " with delta={} should be less than 0.5 after 100 observations of each".format(_DELTA_BOUND)
    #                    )


    #def test_eval_prob_bound_subtractive(self):
    #    expectation_lt, expectation_gt = self._create_basic_terms()
    #    binary_expectation_term = expectation_gt - expectation_lt
    #    self._expression_observe_values(binary_expectation_term, self._x_vals_test_list)
    #    prob_val = binary_expectation_term.eval_bounded(epsilon=3)
    #    self.assertEqual(prob_val.val, 40,
    #                     " (expectation of x = 41,43) - (expectation of x = 1,3) should be 40")
    #    self.assertAlmostEqual(prob_val.f_prob, 0.88,
    #                           msg="delta for (expectation of x = 41,43) - (expectation of x = 1,3)"
    #                               " with epsilon=3 should be 0.88",
    #                           delta=_DELTA_BOUND)

    #def test_eval_prob_bound_multiplicative(self):
    #    expectation_lt, expectation_gt = self._create_basic_terms()
    #    binary_expectation_term = expectation_gt * expectation_lt
    #    self._expression_observe_values(binary_expectation_term, self._x_vals_test_list)
    #    # see https://www.wolframcloud.com/obj/a2a1833f-84cf-4a2f-9103-adf232d09fe0
    #    self._expression_observe_values(binary_expectation_term,
    #                                    [{"x": 42}, {"x": 42}, {"x": 2}, {"x": 2}])
    #    prob_val = binary_expectation_term.eval_bounded(epsilon=40)
    #    self.assertEqual(prob_val.val, 84,
    #                     " (expectation of x = 41,43, 42, 42) * (expectation of x = 1, 3, 2, 2) should be 84")
    #    self.assertAlmostEqual(prob_val.f_prob, 0.290585,
    #                           msg="delta for (expectation of x = 41,43, 42, 42) * (expectation of x = 1, 3, 2, 2)"
    #                               " with epsilon=3 should be 0.290585",
    #                           delta=_DELTA_BOUND)

    #def test_eval_prob_bound_additive_unequal(self):
    #    expectation_lt, expectation_gt = self._create_basic_terms()
    #    binary_expectation_term = expectation_gt + expectation_lt

    #def test_eval_prob_bound_additive_update(self):
    #    condition_lt = create_variable("x") < 10
    #    condition_gt = create_variable("x") > 10
    #    expectation_lt = Expectation(create_variable("x"), given=condition_lt)
    #    expectation_gt = Expectation(create_variable("x"), given=condition_gt)
    #    binary_expectation_term = expectation_gt + expectation_lt
    #    binary_expectation_term.observe({
    #        "x": 1
    #    }, call_id=0)
    #    binary_expectation_term.observe({
    #        "x": 3
    #    }, call_id=1, observation_key="key")
    #    binary_expectation_term.observe({
    #        "x": 41
    #    }, call_id=2)
    #    binary_expectation_term.observe({
    #        "x": 43
    #    }, call_id=3)
    #    # variance([1,3]) == variance([41,43]) == 2
    #    # For plot of example see: https://www.wolframalpha.com/input/?i=plot+%28%28%282+%2F+%282*d1%29%29%5E0.5%29+%2B+%28%282+%2F+%282*d2%29%29%5E0.5%29+%3D+3%29%2C+0+%3C%3D+d1%3C%3D+1%2C+0+%3C%3D+d2+%3C%3D+1+and+plot+d1+%2B+d2+%3D+0.88888888
    #    prob_val = binary_expectation_term.eval_bounded(epsilon=3)
    #    self.assertEqual(prob_val.val, 44,
    #                     "(expectation of x = 1,3) + (expectation of x = 41,43) should be 44")
    #    self.assertAlmostEqual(prob_val.f_prob, 0.88,
    #                           msg="delta for (expectation of x = 1,3) + (expectation of x = 41,43) with epsilon=3 should be 0.88",
    #                           delta=_DELTA_BOUND)
    #    binary_expectation_term.observe({
    #        "x": 2
    #    }, call_id=4, observation_key="key")
    #    # variance([1,2]) = 0.5
    #    # For plot of example see: wolframalpha.com/input/?i=plot+%28%28%280.5+%2F+%282*d1%29%29%5E0.5%29+%2B+%28%282+%2F+%282*d2%29%29%5E0.5%29+%3D+3%29%2C+0+<%3D+d1<%3D+1%2C+0+<%3D+d2+<%3D+1+and+plot+d1+%2B+d2+%3D+0.48
    #    prob_val_updated = binary_expectation_term.eval_bounded(epsilon=3)
    #    self.assertEqual(prob_val_updated.val, 43.5,
    #                     "(expectation of x = 1,2) + (expectation of x = 41,43) should be 43.5")
    #    self.assertAlmostEqual(prob_val_updated.f_prob, 0.48,
    #                           msg="delta for (expectation of x = 1,2) + (expectation of x = 41,43) with epsilon=3 should be 0.88",
    #                           delta=_DELTA_BOUND)

    #def test_eval_prob_bound_additive_unobserve(self):
    #    condition_lt = create_variable("x") < 10
    #    condition_gt = create_variable("x") > 10
    #    expectation_lt = Expectation(create_variable("x"), given=condition_lt)
    #    expectation_gt = Expectation(create_variable("x"), given=condition_gt)
    #    binary_expectation_term = expectation_gt + expectation_lt
    #    binary_expectation_term.observe({
    #        "x": 1
    #    }, call_id=0)
    #    binary_expectation_term.observe({
    #        "x": 3
    #    }, call_id=1)
    #    binary_expectation_term.observe({
    #        "x": 9
    #    }, call_id=2, observation_key="key")
    #    binary_expectation_term.observe({
    #        "x": 41
    #    }, call_id=3)
    #    binary_expectation_term.observe({
    #        "x": 43
    #    }, call_id=4)
    #    binary_expectation_term.unobserve(call_id=5, observation_key="key")
    #    # variance([1,3]) == variance([41,43]) == 2
    #    # For plot of example see: https://www.wolframalpha.com/input/?i=plot+%28%28%282+%2F+%282*d1%29%29%5E0.5%29+%2B+%28%282+%2F+%282*d2%29%29%5E0.5%29+%3D+3%29%2C+0+%3C%3D+d1%3C%3D+1%2C+0+%3C%3D+d2+%3C%3D+1+and+plot+d1+%2B+d2+%3D+0.88888888
    #    prob_val = binary_expectation_term.eval_bounded(epsilon=3)
    #    self.assertEqual(prob_val.val, 44,
    #                     "(expectation of x = 1,3) + (expectation of x = 41,43) should be 44")
    #    self.assertAlmostEqual(prob_val.f_prob, 0.88,
    #                           msg="delta for (expectation of x = 1,3) + (expectation of x = 41,43)"
    #                               " with epsilon=3 should be 0.88",
    #                           delta=_DELTA_BOUND)


def suite():
    mysuite = unittest.TestSuite()
    mysuite.addTests([
        TestExpectation("test_expectation"),
        #TestExpectation("test_expectation_unobserve"),
        TestExpectation("test_conditional_expectation"),
        #TestExpectation("test_expectation_bound_at_delta"),
        #TestExpectation("test_expectation_bound_at_delta_monotonic_epsilon"),
        TestExpectationTerm("test_binary_expectation_term"),
        #TestExpectationTerm("test_binary_expectation_term_unobserve"),
        TestExpectationTerm("test_binary_expectation_term_update"),
        TestExpectationTerm("test_binary_expectation_term_with_constant"),
        TestExpectationTerm("test_binary_expectation_term_with_constant"),
        TestExpectationTerm("test_binary_expectation_term_with_r_constant")#,
        #TestExpectationTerm("test_epsilon_propagation"),
        #TestExpectationTerm("test_eval_prob_bound_additive"),
        #TestExpectationTerm("test_eval_prob_bound_subtractive"),
        #TestExpectationTerm("test_eval_prob_bound_multiplicative"),
        #TestExpectationTerm("test_eval_prob_bound_additive_update"),
        #TestExpectationTerm("test_eval_prob_bound_additive_unobserve"),
    ])
    return mysuite


def test():
    test_suite = suite()
    runner = unittest.TextTestRunner()
    runner.run(test_suite)