from typing import Dict, Tuple
import pandas as pd
import numpy as np
from os import path
from .dataset import Dataset


DATA_DIR = path.join(path.dirname(__file__), "data", "compas")
DATA_FILE = "compas-scores-two-years.csv"

SORT_BY_DATE_COLUMN = "compas_screening_date"

TRAIN_PORTION = 0.5

original_attributes = {
    'id': '',
    'age': '',
    'c_charge_degree': '',
    'race': '',
    'age_cat': '',
    'score_text': '',
    'sex': '',
    'priors_count': '',
    'days_b_screening_arrest': '',
    'decile_score': '',
    'is_recid': '',
    'two_year_recid': ''
}

attributes = ['id', 'age', 'priors_count', 'days_b_screening_arrest', 'decile_score',
              'is_recid', 'two_year_recid', 'c_charge_degree_F', 'c_charge_degree_M',
              'race_African-American', 'race_Asian', 'race_Caucasian',
              'race_Hispanic', 'race_Native American', 'race_Other',
              'age_cat_25 - 45', 'age_cat_Greater than 45', 'age_cat_Less than 25',
              'score_text_High', 'score_text_Low', 'score_text_Medium', 'sex_Female',
              'sex_Male']

attributes_dict = dict((attr, "") for attr in attributes)

for attr, description in original_attributes.items():
    attributes_dict.update({
        matched_attr:description
        for matched_attr in attributes
        if matched_attr.startswith(attr)
    })

def _read_data():
    data_fp = path.join(DATA_DIR, DATA_FILE)
    data = pd.read_csv(data_fp)

    data[SORT_BY_DATE_COLUMN] = pd.to_datetime(data[SORT_BY_DATE_COLUMN]).dt.date
    data = data.sort_values(by=[SORT_BY_DATE_COLUMN], ascending=True)

    view = data[original_attributes.keys()].copy()
    view = view[(view.days_b_screening_arrest <= 30) &
                (view.days_b_screening_arrest >= -30)]
    view = view[view.is_recid != -1]
    view = view[view.c_charge_degree != "O"]
    view = view[view.score_text != 'N/A']

    view = pd.get_dummies(view)

    train_size = int(len(view) * TRAIN_PORTION)
    train = view.iloc[:train_size]
    test = view.iloc[train_size:]
    test = view

    return train, test


def load_data() -> Dataset:
    train, test = _read_data()

    return Dataset(
        train=train,
        test=test,
        attributes_dict=attributes_dict,
        target="two_year_recid"
    )
