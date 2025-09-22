from typing import NamedTuple, Dict, Union, List
from pandas import DataFrame

class Dataset:
    train: DataFrame
    test: DataFrame
    attributes_dict: Dict[str, str]
    target: Union[None, str] # target attribute
    inputs: Union[None, List[str]] # attributes to use as input
    
    def __init__(self, train, test, attributes_dict: Union[List[str], Dict[str, str]], target=None, inputs=None):
        self.train = train
        self.test = test

        if not (isinstance(attributes_dict, dict) or isinstance(attributes_dict, list)):
            raise NotImplementedError("Dataset only supports attributes of in a list or dictionary")

        self.attributes_dict = attributes_dict if isinstance(attributes_dict, dict) else {k:"" for k in attributes_dict}
        self.target = target
        self.inputs = inputs if inputs is not None else [attr for attr in self.train.columns.values if attr != self.target]

    @property
    def attributes(self):
        return list(self.attributes_dict.keys())

    def _split_Xy(self, data):
        return (data.loc[:, self.inputs], data[self.target])

    @property 
    def train_Xy(self):
        return self._split_Xy(self.train)

    @property 
    def test_Xy(self):
        return self._split_Xy(self.test)