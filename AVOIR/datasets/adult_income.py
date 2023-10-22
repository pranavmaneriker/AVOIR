from typing import Dict, Tuple
import pandas as pd
import numpy as np
from os import path
from .dataset import Dataset

DATA_DIR = path.join(path.dirname(__file__), "data", "adult_income")  #"./dsl/tests/datasets/data/adult_income"
TRAIN_FILE = "adult_train.csv"
TEST_FILE = "adult_test.csv"

"""
Attributes before categorical encoding
"""
original_attributes = {
    "age": "continuous",
    "workclass": "Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.",
    "fnlwgt": "Final Weight (continuous): See https://www.census.gov/programs-surveys/sipp/methodology/weighting.html",
    "education": "Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.",
    "education_num": "School grade level (continuous)",
    "marital_status": "Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.",
    "occupation": "Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces",
    "relationship": "Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.",
    "race": "White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.",
    "sex": "Female, Male.",
    "capital_gain": "continuous",
    "capital_loss": "continuous.",
    "hours_per_week": "continuous.",
    "native_country": "United-States, Cambodia, England, etc.",
    "high_income": ">50k, <=50k"
}

attributes = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "high_income",
    "workclass_Federal-gov",
    "workclass_Local-gov",
    "workclass_Never-worked",
    "workclass_Private",
    "workclass_Self-emp-inc",
    "workclass_Self-emp-not-inc",
    "workclass_State-gov",
    "workclass_Without-pay",
    "education_10th",
    "education_11th",
    "education_12th",
    "education_1st-4th",
    "education_5th-6th",
    "education_7th-8th",
    "education_9th",
    "education_Assoc-acdm",
    "education_Assoc-voc",
    "education_Bachelors",
    "education_Doctorate",
    "education_HS-grad",
    "education_Masters",
    "education_Preschool",
    "education_Prof-school",
    "education_Some-college",
    "marital_status_Divorced",
    "marital_status_Married-AF-spouse",
    "marital_status_Married-civ-spouse",
    "marital_status_Married-spouse-absent",
    "marital_status_Never-married",
    "marital_status_Separated",
    "marital_status_Widowed",
    "occupation_Adm-clerical",
    "occupation_Armed-Forces",
    "occupation_Craft-repair",
    "occupation_Exec-managerial",
    "occupation_Farming-fishing",
    "occupation_Handlers-cleaners",
    "occupation_Machine-op-inspct",
    "occupation_Other-service",
    "occupation_Priv-house-serv",
    "occupation_Prof-specialty",
    "occupation_Protective-serv",
    "occupation_Sales",
    "occupation_Tech-support",
    "occupation_Transport-moving",
    "relationship_Husband",
    "relationship_Not-in-family",
    "relationship_Other-relative",
    "relationship_Own-child",
    "relationship_Unmarried",
    "relationship_Wife",
    "race_Amer-Indian-Eskimo",
    "race_Asian-Pac-Islander",
    "race_Black",
    "race_Other",
    "race_White",
    "sex_Female",
    "sex_Male",
    "native_country_Cambodia",
    "native_country_Canada",
    "native_country_China",
    "native_country_Columbia",
    "native_country_Cuba",
    "native_country_Dominican-Republic",
    "native_country_Ecuador",
    "native_country_El-Salvador",
    "native_country_England",
    "native_country_France",
    "native_country_Germany",
    "native_country_Greece",
    "native_country_Guatemala",
    "native_country_Haiti",
    "native_country_Holand-Netherlands",
    "native_country_Honduras",
    "native_country_Hong",
    "native_country_Hungary",
    "native_country_India",
    "native_country_Iran",
    "native_country_Ireland",
    "native_country_Italy",
    "native_country_Jamaica",
    "native_country_Japan",
    "native_country_Laos",
    "native_country_Mexico",
    "native_country_Nicaragua",
    "native_country_Outlying-US(Guam-USVI-etc)",
    "native_country_Peru",
    "native_country_Philippines",
    "native_country_Poland",
    "native_country_Portugal",
    "native_country_Puerto-Rico",
    "native_country_Scotland",
    "native_country_South",
    "native_country_Taiwan",
    "native_country_Thailand",
    "native_country_Trinadad&Tobago",
    "native_country_United-States",
    "native_country_Vietnam",
    "native_country_Yugoslavia"
]

attributes_dict = dict((attr, "") for attr in attributes)

for attr, description in original_attributes.items():
    attributes_dict.update({
        matched_attr:description
        for matched_attr in attributes
        if matched_attr.startswith(attr)
    })



def _read_data():
    train_fp = path.join(DATA_DIR, TRAIN_FILE)
    test_fp = path.join(DATA_DIR, TEST_FILE)
    train_raw = pd.read_csv(train_fp, names=original_attributes.keys(), header=0)
    test_raw = pd.read_csv(test_fp, names=original_attributes.keys(), header=0)

    replace = {
        ">50K": 1,
        "<=50K": 0,
        ">50K.": 1,
        "<=50K.": 0,
        "?": np.nan
    }

    # Note: concat data together so we get all values when using `get_dummies`
    all_raw = pd.concat([train_raw, test_raw])
    all_data = all_raw.replace(replace)
    all_data = pd.get_dummies(all_data)
    train = all_data.iloc[:len(train_raw)]
    test = all_data.iloc[len(train_raw):]

    return train, test

def _df_to_Xy(df):
    return (
        df.drop("high_income", axis=1).to_numpy(),
        df.high_income.to_numpy()
    )

def load_data(train_size=8000, test_size=1000) -> Dataset:
    train_full, test_full = _read_data()

    train, test = (train_full.sample(n=train_size), test_full.sample(n=test_size))

    return Dataset(
        train=train,
        test=test,
        attributes_dict=attributes_dict,
        target="high_income"
    )




