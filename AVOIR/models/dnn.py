from sklearn.neural_network import MLPClassifier
from sklearn.utils.multiclass import check_classification_targets
from .model import ModelBasedTest
from ..datasets import Dataset


class ThreeLayerMLPClassificationTest(ModelBasedTest):
    # overrides(ModelBasedTest)
    def init_data_and_model(self, dataset: Dataset):
        X_train, y_train = dataset.train_Xy
        self.X_test, self.y_test = dataset.test_Xy

        self.model = MLPClassifier(hidden_layer_sizes=(20, 10, 5))
        self.model.fit(X_train, y_train)

    # overrides(ModelBasedTest)
    def is_valid_target(self, values) -> bool:
        try:
            check_classification_targets(values)
            return True
        except ValueError:
            return False




