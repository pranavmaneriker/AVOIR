from typing import Union, List, Dict
from enum import Enum
import json
from uuid import uuid4, UUID
from .history import HistoryLoggingExpression


class TableRowType(Enum):
    specification = "Specification"
    specification_threshold = "Specification Threshold"
    expectation = "Expectation"
    constant_expectation_term = "Constant"
    expectation_term = "ETerm"
    numerical_expression = "Numerical Expression"


class JSONTableRow:
    def __init__(self, row_type: TableRowType):
        self.row_type = row_type

    @property
    def dict(self):
        return {
            "type": self.row_type.value
        }


class JSONTableEncodableTreeExpression(HistoryLoggingExpression):
    """
    Defines a tree type that can be enoded in to a tabular json format.
    Implementing classes should atleast override children and row_representation
    """
    def __init__(self):
        super().__init__()
        self.id = None

    @property
    def children(self) -> List['JSONTableEncodableTreeExpression']:
        raise NotImplementedError()

    @property
    def row_representation(self) -> JSONTableRow:
        raise NotImplementedError()

    @property
    def row_id(self) -> str:
        if not self.id:
            self.id = str(uuid4())
        return self.id

    @property
    def bound_values(self):
        if not self.history:
            return []
        return self.history

    def __repr__(self):
        raise NotImplementedError()

    def tabular_representation(self, with_idx=0, with_parent_id=None):
        this_row = {
            **self.row_representation.dict,
            **{"id": self.row_id, "vals": self.bound_values,
               "parent_id": with_parent_id, "repr": f"{self}", "idx": with_idx}
        }
        return [this_row] + [
            row
            for child in self.children
            for row in child.tabular_representation(with_idx=with_idx, with_parent_id=self.row_id)
        ]