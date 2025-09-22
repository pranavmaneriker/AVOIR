from .model import ModelBasedTest


class ViewMaintenance(ModelBasedTest):
    # overrides(ModelBasedTest)
    def init_data_and_model(self, _):
        pass

    # overrides(ModelBasedTest)
    def is_valid_target(self, _) -> bool:
        return True

    def predict(self, x):
        return [1.0]