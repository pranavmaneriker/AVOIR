from sklearn import tree
from sklearn.utils.multiclass import check_classification_targets
from .model import ModelBasedTest
from ..datasets import Dataset

class DecisionTreeTest(ModelBasedTest):
    # overrides(ModelBasedTest)
    def init_data_and_model(self, dataset: Dataset):
        X_train, y_train = dataset.train_Xy
        self.X_test, self.y_test = dataset.test_Xy

        self.model = tree.DecisionTreeClassifier()
        self.model.fit(X_train, y_train)

    # overrides(ModelBasedTest)
    def is_valid_target(self, values) -> bool:
        try:
            check_classification_targets(values)
            return True
        except ValueError:
            return False