from sklearn import datasets
from .dataset import Dataset
import numpy as np
import pandas as pd

attributes = [
    "crim", # per capita crime rate by town
    "zn", # proportion of residential land zoned for lots over 25,000 sq.ft.
    "indus", # proportion of non-retail business acres per town
    "chas", # Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    "nox", # nitric oxides concentration (parts per 10 million)
    "rm", # average number of rooms per dwelling
    "age", # proportion of owner-occupied units built prior to 1940
    "dis", # weighted distances to five Boston employment centres
    "rad", # index of accessibility to radial highways
    "tax", # full-value property-tax rate per $10,000
    "ptratio", # pupil-teacher ratio by town
    "b", # 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    "lstat", # % lower status of the population
    "medv" # Median value of owner-occupied homes in $1000’s
]

attributes_dict = dict(crim=" per capita crime rate by town",
                       zn=" proportion of residential land zoned for lots over 25000 sq.ft.",
                       indus=" proportion of non-retail business acres per town",
                       chas=" Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)",
                       nox=" nitric oxides concentration (parts per 10 million)",
                       rm=" average number of rooms per dwelling",
                       age=" proportion of owner-occupied units built prior to 1940",
                       dis=" weighted distances to five Boston employment centres",
                       rad=" index of accessibility to radial highways",
                       tax=" full-value property-tax rate per $10000",
                       ptratio=" pupil-teacher ratio by town",
                       b=" 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town",
                       lstat=" % lower status of the population",
                       medv=" Median value of owner-occupied homes in $1000’s")

def load_data() -> Dataset:
    X, y = datasets.load_boston(return_X_y=True)

    np_data = np.column_stack((X, y))

    data = pd.DataFrame(data=np_data, columns=attributes)

    train = data[:-20]
    test = data[:120]

    return Dataset(
        train=train,
        test=test,
        attributes_dict=attributes_dict,
        target="medv"
    )