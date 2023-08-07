from sklearn import linear_model
import numpy as np
from .model import ModelBasedTest
from ..datasets import Dataset


_NUMERIC_KINDS = set('buif')
def _valid_array_type(array):
    return np.asarray(array).dtype.kind in _NUMERIC_KINDS

class LinearRegressionTest(ModelBasedTest):
    # overrides(ModelBasedTest)
    def init_data_and_model(self, dataset: Dataset):
        X_train, y_train = dataset.train_Xy
        self.X_test, self.y_test = dataset.test_Xy

        self.model = linear_model.LinearRegression()
        self.model.fit(X_train, y_train)

    # overrides(ModelBasedTest)
    def is_valid_target(self, values: np.ndarray) -> bool:
        return _valid_array_type(values)