import pandas as pd
import numpy as np
import os

import time
from contextlib import contextmanager
from sklearn.base import BaseEstimator, TransformerMixin

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.3f} s')

def read_csv(fname, input_path="./input/", **kwargs):
    full_path = os.path.join(input_path,fname)
    return pd.read_csv(full_path,**kwargs)

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

class TextStats(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, texts):
        return [{'num_letter': len(texts)}
                for text in texts]