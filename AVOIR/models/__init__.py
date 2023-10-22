import logging
from .decision_tree import DecisionTreeTest as decision_tree
from .logistic_regression import LogisticRegressionTest as logistic
from .linear_regression import LinearRegressionTest as lr
from .rbf_svc import SVCTest as svc
from .dnn import ThreeLayerMLPClassificationTest as mlp
from .transformer_regression import TransformerRegressionTest as transformer_regression
from .view_maintenance import ViewMaintenance as view_maintenance
from .model import ModelBasedTest, InvalidTargetError, InvalidTrainError

_decision = "Decision Tree"
_lr = "Linear Regression"
_logistic = "Logistic Regression"
_rbf_svc = "RBF-SVM Classification"
_mlp = "Three Layer MLP"
_transformer = "Bert Text Regression"
_view_maintenance = "View Maintenance"

_models = [_lr, _decision, _logistic, _rbf_svc, _mlp, _transformer, _view_maintenance]


def check_model_exists(m: str):
    if m not in _models:
        raise ValueError("Model Does not exist")


def get_models():
    return _models


def get_model(m: str):
    check_model_exists(m)
    return {
        _decision: decision_tree,
        _lr: lr,
        _logistic: logistic,
        _rbf_svc: svc,
        _mlp: mlp,
        _transformer: transformer_regression,
        _view_maintenance: view_maintenance
    }[m]
