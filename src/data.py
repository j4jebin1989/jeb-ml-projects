# Data processing functions for ML project
import pandas as pd
from sklearn import datasets

def load_iris_data():
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df
