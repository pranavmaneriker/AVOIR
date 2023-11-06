from enum import Enum
from typing import Dict, Union, List
import operator
import math

from .representation import JSONTableEncodableTreeExpression, JSONTableRow, TableRowType
from .type_utils import ObservationType


class NumericalExpressionType(Enum):
    variable = 1
    constant = 2
    expr = 3


def _custom_divide(left, right):
    if math.isclose(right, 0.0):
        return float('nan')
    else:
        return left/right


class NumericalOperator(Enum):
    add = 1
    subtract = 2
    multiply = 3
    divide = 4
    equality = 5
    non_equality = 6
    greater_than = 7
    less_than = 8
    op_and = 9
    op_or = 10


    @classmethod
    def functions(cls):
        return {
            cls.add: operator.add,
            cls.multiply: operator.mul,
            cls.divide: _custom_divide,  # operator.truediv,
            cls.subtract: operator.sub,
            cls.equality: operator.eq,
            cls.non_equality: operator.ne,
            cls.greater_than: operator.gt,
            cls.less_than: operator.lt,
            cls.op_and: operator.and_,
            cls.op_or: operator.or_
        }

    @classmethod
    def opposite_function(cls):
        return {
            cls.equality: operator.ne,
            cls.non_equality: operator.eq,
            cls.greater_than: operator.le,
            cls.less_than: operator.ge,
        }

    @classmethod
    def symbols(cls):
        return {
            cls.add: "+",
            cls.multiply: "*",
            cls.divide: "/",
            cls.subtract: "-",
            cls.equality: "=",
            cls.non_equality: "!=",
            cls.greater_than: ">",
            cls.less_than: "<",
            cls.op_and: "&",
            cls.op_or: "|"
        }

    def is_arithmetic(self):
        if self.name in ["add", "subtract", "multiply", "divide"]:
            return True
        return False


class NumericalExpression:#(JSONTableEncodableTreeExpression):
    RESERVE_WORD_RETVAL = "NumericalExpression.RESERVE_WORD_RETVAL"
    _RETVAL_SYMBOLIC_REP = "return"

    def __init__(self, symbolic_rep: str = None,
                 expression_type: NumericalExpressionType = None,
                 left_child: 'NumericalExpression' = None,
                 right_child: 'NumericalExpression' = None,
                 op: NumericalOperator = None):
        """
        Numerical Expressions are implemented using operator overloading
        The expression keeps track of the operations to be performed on operands
        eval() evaluates the expression, assuming all variables are bound to values
        The assumption is that the underlying variable value type must support binary operations
        Note: Eval does NOT currently support hyperexpressions (binding must be to formal param, not func)
        :param symbolic_rep: string representation for expression
        from .utils import ObservationType
        :param expression_type: variable (binding), constant, or expr
        :param left_child: the value, or the left child if binary op
        :param right_child:
        :param op: binary operator
        """
        self.left_child = left_child
        self.right_child = right_child
        self.op = op
        self.expression_type = expression_type
        self.symbolic_rep = symbolic_rep
        self.val = None
        self.is_return_value = False
        if symbolic_rep == self.RESERVE_WORD_RETVAL:
            self.is_return_value = True
        self.is_bound = False
        #JSONTableEncodableTreeExpression.__init__(self)

    def bind(self, val):
        """
        Before eval, we must bind all variable Numerical Expressions to values
        For constant types, the binding must be completed when creating the expression
        :param val: scalar, vector, or matrix from arbitrary domains
        """
        if self.expression_type == NumericalExpressionType.expr:
            raise TypeError("Binding only allowed for variables and constants")
        self.val = val
        self.is_bound = True

    def unbind(self):
        if self.expression_type == NumericalExpressionType.expr:
            raise TypeError("Unbinding only allowed for variables and constants")
        self.val = None
        self.is_bound = False

    def bind_variables(self, vals: ObservationType, call_id=None): #, call_id=0):
        """
        Given a mapping from variable names to values, bind_variables recursively
        binds values in the mapping to variables contained within its expression
        :param vals: mapping from variable names to a value (scalar, vector, or matrix)
        """

        if self.expression_type == NumericalExpressionType.constant:
            return
        elif self.expression_type == NumericalExpressionType.variable and self.symbolic_rep in vals.keys():
            val = vals.get(self.symbolic_rep)
            self.bind(val) 
        elif self.expression_type == NumericalExpressionType.expr:
            self.left_child.bind_variables(vals) #, call_id)
            self.right_child.bind_variables(vals) #, call_id)

        #self.update_values(call_id, value_key=observation_key)

    def unbind_variables(self):
        if self.expression_type == NumericalExpressionType.constant:
            return
        elif self.expression_type == NumericalExpressionType.variable:
            self.unbind() 
        elif self.expression_type == NumericalExpressionType.expr:
            self.left_child.unbind_variables()
            self.right_child.unbind_variables()

    def eval(self, call_id=None):
        if self.expression_type != NumericalExpressionType.expr and not self.is_bound:
            raise UnboundLocalError(f"Trying to evaluate an unbound expression: {self.symbolic_rep}")

        if self.expression_type == NumericalExpressionType.expr:
            left_val = self.left_child.eval(call_id)
            right_val = self.right_child.eval(call_id)
            return NumericalOperator.functions()[self.op](left_val, right_val)
        else:
            return self.val

    def __repr__(self):
        if self.is_return_value:
            return self._RETVAL_SYMBOLIC_REP
        return self.symbolic_rep

    # reverse operators overloading added to make it easier to write things like 1/x
    def __add__(self, other: Union['NumericalExpression', int, float]):
        return create_expression(self, other, NumericalOperator.add)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return create_expression(self, other, NumericalOperator.subtract)

    def __rsub__(self, other):
        return (self * -1) + other

    def __mul__(self, other):
        return create_expression(self, other, NumericalOperator.multiply)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return create_expression(self, other, NumericalOperator.divide)

    def __rtruediv__(self, other):
        if isinstance(other, NumericalExpression):
            return other.__truediv__(self)
        elif isinstance(other, int) or isinstance(other, float):
            return create_constant(other).__truediv__(self)

    def __eq__(self, other):
        return create_expression(self, other, NumericalOperator.equality)

    def __ne__(self, other):
        return create_expression(self, other, NumericalOperator.non_equality)

    def __gt__(self, other):
        return create_expression(self, other, NumericalOperator.greater_than)
    
    def __lt__(self, other):
        return create_expression(self, other, NumericalOperator.less_than)

    def __and__(self, other):
        return create_expression(self, other, NumericalOperator.op_and)

    def __or__(self, other):
        return create_expression(self, other, NumericalOperator.op_or)

    # overrides(JSONTableEncodableTreeExpression)
    #@property
    #def children(self) -> List[JSONTableEncodableTreeExpression]:
    #    return []

    ## overrides(JSONTableEncodableTreeExpression)
    #@property
    #def row_representation(self) -> JSONTableRow:
    #    return JSONTableRow(
    #        row_type=TableRowType.numerical_expression
    #    )


def create_expression(left: NumericalExpression, right: Union[NumericalExpression, int, float],
                      op: NumericalOperator) -> NumericalExpression:
    op_symbol = NumericalOperator.symbols()[op]
    if isinstance(right, float) or isinstance(right, int) or isinstance(right, str):
        right = create_constant(right)
    return NumericalExpression(
        symbolic_rep=f"{left} {op_symbol} {right}",
        expression_type=NumericalExpressionType.expr,
        left_child=left,
        right_child=right,
        op=op
    )


def create_constant(val) -> NumericalExpression:
    const_expr = NumericalExpression(
        symbolic_rep=str(val),
        expression_type=NumericalExpressionType.constant
    )
    const_expr.bind(val)
    return const_expr


def create_variable(var_name) -> NumericalExpression:
    """Note: This is the ONLY mechanism that should be used to create variables to avoid repeated binding"""
    var_expr = NumericalExpression(
        symbolic_rep=var_name,
        expression_type=NumericalExpressionType.variable
    )
    return var_expr


RETURN_VARIABLE = create_variable(NumericalExpression.RESERVE_WORD_RETVAL)
