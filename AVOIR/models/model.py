from typing import List
from ...dsl import SpecTracker
from ..datasets import Dataset
import numpy as np
import pdb

class InvalidTargetError(Exception):
    pass

class InvalidTrainError(Exception):
    pass

class ModelBasedTest:
    model = None
    f_x = None

    def __init__(self, dataset: Dataset):
        valid_targets = np.all([
            self.is_valid_target(y.to_numpy())
            for y in [dataset.train_Xy[1], dataset.test_Xy[1]]
        ])
        if not valid_targets:
            raise InvalidTargetError("Invalid target provided to ModelBasedTest")

        valid_trains = np.all([
            self.is_valid_train(x)
            for x in [dataset.train_Xy[0], dataset.test_Xy[0]]
        ])
        if not valid_trains:
            raise InvalidTrainError("Invalid training data provided to ModelBasedTest")

        self.init_data_and_model(dataset)

    def init_data_and_model(self, dataset: Dataset):
        raise NotImplementedError(
            ("Test must be initialized with: "
            "model, X_test, y_test")
        )

    @property
    def data(self):
        raise NotImplementedError("Must define data to be used during simulation")

    def is_valid_target(cls, values: np.ndarray) -> bool:
        """
        Determines if a given value can be used as a target value by this model
        """
        raise NotImplementedError("Must define what a valid target looks like in a ModelBasedTest")

    def is_valid_train(cls, values) -> bool:
        """
        Determines if the given training data can be utiltlized by this model test
        """
        return True

    def update_spec(self, new_spec):
        raise NotImplementedError("update_spec should be implmented and should set f_x")

    def get_tabular_rep(self):
        spec_obj = SpecTracker.get_spec(self.f_x)
        tabular_rep = spec_obj.tabular_representation()
        return tabular_rep

    def eval_spec(self):
        SpecTracker.get_spec(self.f_x).eval()

    def predict(self, x):
        return self.model.predict(x)
    
    def predict_list(self, input_list):
        outputs = []
        for item in input_list:
            outputs.append(self.predict(item))
        return outputs
                         
    
    def run_eval_loop(self, progress_bar=None):
        iteration_data = self.data
        if progress_bar:
            iteration_data = progress_bar(iteration_data)
        for datum in iteration_data:
            self.f_x(**datum)
            
