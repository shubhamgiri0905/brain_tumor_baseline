import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

def load_data():
    X,y = make_classification(n_samples=50, n_features=5, n_informative=3, n_redundant=0, random_state=42)
    columns = ['age', 'tumor_size', 'density', 'symptom_score', 'risk_factor']
    df = pd.DataFrame(X, columns=columns)
    df['has_tumor'] = y
    return df


def preprocessing_data(df):
    X = df.drop('has_tumor', axis=1)
    y = df['has_tumor']
    return train_test_split(X, y, test_size=0.2, random_state=42)

