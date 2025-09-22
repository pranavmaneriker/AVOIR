import unittest

from AVOIR.dsl.grammar import numerical as nu


class TestNumericalExpression(unittest.TestCase):
    def test_constant(self):
        e = nu.create_constant(2)
        self.assertEqual(e.eval(), 2, "Should be 2")

    def test_expression_constants(self):
        c1 = nu.create_constant(2)
        c2 = nu.create_constant(3)
        e = nu.create_expression(c1, c2, nu.NumericalOperator.add)
        self.assertEqual(e.eval(), 5, "2 + 3 Should be 5")

    def test_expression_constant_overloading(self):
        c1 = nu.create_constant(2)
        c2 = nu.create_constant(3)
        e = c1 + c2
        self.assertEqual(e.eval(), 5, "2 + 3 should be 5")

    def test_expression_variables(self):
        v1 = nu.create_variable("x")
        c1 = nu.create_constant(10)
        e = c1 * v1
        self.assertRaises(UnboundLocalError, e.eval)
        v1.bind_variables({"x": 10})
        self.assertEqual(e.eval(), 100, "10 * 10 should be 100")

    def test_expression_equality(self):
        c1 = nu.create_constant(2)
        c2 = nu.create_constant(3)
        e_eq = c1 == c1
        self.assertTrue(e_eq.eval(), "2 should be equivalent to 2")
        e_uneq = c1 == c2
        self.assertFalse(e_uneq.eval(), "2 should not be equivalent to 3")
        v1 = nu.create_variable("x")
        e_neq_var = c1 != v1
        self.assertRaises(UnboundLocalError, e_neq_var.eval)
        v1.bind(3)
        self.assertTrue(e_neq_var.eval(), "2 should not be equivalent to 3")

    def test_auto_overloading(self):
        v1 = nu.create_variable("x")
        e = 2/v1
        v1.bind_variables({"x": 5})
        self.assertEqual(e.eval(), 0.4, "2/x = 0.4 for x = 5")
    
    def test_recursive_binding(self):
        v1 = nu.create_variable("x")
        v2 = nu.create_variable("y")
        e = v1 + v2
        e.bind_variables({"x": 1, "y": 2})
        self.assertEqual(e.eval(), 3, "1 + 2 = 3")
        
        e.unbind_variables()
        e.bind_variables({"x": 3, "y": 4})
        self.assertEqual(e.eval(), 7, "3 + 4 = 7 after unbinding")


def suite():
    suite = unittest.TestSuite()
    suite.addTests([
        TestNumericalExpression("test_constant"),
        TestNumericalExpression("test_expression_constants"),
        TestNumericalExpression("test_expression_constant_overloading"),
        TestNumericalExpression("test_expression_variables"),
        TestNumericalExpression("test_expression_equality"),
        TestNumericalExpression("test_auto_overloading"),
        TestNumericalExpression("test_recursive_binding")
    ])
    return suite


def test():
    test_suite = suite()
    runner = unittest.TextTestRunner()
    runner.run(test_suite)