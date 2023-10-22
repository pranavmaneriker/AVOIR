from simpletransformers.classification import ClassificationModel, ClassificationArgs
import os
import torch
import numpy as np
import pandas as pd
from .model import ModelBasedTest
from ..datasets import Dataset



_NUMERIC_KINDS = set('buif')
def _valid_array_type(array):
    return np.asarray(array).dtype.kind in _NUMERIC_KINDS

class TransformerRegressionTest(ModelBasedTest):
    @property
    def untrained_model(self):
        if os.path.exists("./outputs/"):
            self.is_trained = True
            model = ClassificationModel("bert", "outputs", use_cuda=False)
        else:
            self.is_trained = False
            model_args = ClassificationArgs()
            model_args.num_train_epochs = 10
            model_args.regression = True
            model_args.overwrite_output_dir = False
            model_args.use_multiprocessing = False
            model_args.use_multiprocessing_for_eval = False

            # Create a ClassificationModel
            #use_cuda = torch.cuda.is_available()
            use_cuda = False
            model = ClassificationModel(
                "bert",
                "bert-base-uncased",
                num_labels=1,
                args=model_args,
                use_cuda=use_cuda,
            )
        return model
    
    @classmethod
    def xy_to_transformer_input(cls, X, y):
        X_list = X[X.columns[0]].to_list()
        y_list = y.to_list()

        input_df = pd.DataFrame({"text": X_list, "labels": y_list});
        return input_df

    # overrides(ModelBasedTest)
    def init_data_and_model(self, dataset: Dataset):
        X_train, y_train = dataset.train_Xy
        transformer_train = self.xy_to_transformer_input(X_train, y_train)
        self.X_test, self.y_test = dataset.test_Xy
        self.transformer_test = self.xy_to_transformer_input(self.X_test, self.y_test)

        self.model = self.untrained_model
        if not self.is_trained:
            self.model.train_model(transformer_train)

    # overrides(ModelBasedTest)
    def is_valid_target(self, values: np.ndarray) -> bool:
        return _valid_array_type(values)

    # overrides(ModelBasedTest)
    def is_valid_train(self, values: pd.DataFrame) -> bool:
        if len(values.columns) != 1:
            return False

        train_series = values[values.columns[0]]

        return train_series.apply(lambda s: isinstance(s, str)).all()

    # overrides(ModelBasedTest)
    def predict(self, x):
        silent_orig = self.model.args.silent
        self.model.args.silent = True
        model_input = list(x[0])
        predictions, raw_outputs = self.model.predict(model_input)
        self.model.args.silent = silent_orig
        return [predictions]

    def predict_list(self, x):
        predictions, raw_outputs = self.model.predict(x)
        return predictions.tolist()