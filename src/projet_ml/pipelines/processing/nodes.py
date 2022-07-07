import pandas as pd

from typing import Dict, Any

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def encode_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Encode features of data file.
    """
    features = dataset.drop(["TransactionNo", "ProductNo"], axis=1).copy()
    features['Date'] = pd.to_datetime(features['Date'], errors='coerce')
    features['Date_day'] = features.Date.dt.day
    features['Date_month'] = features.Date.dt.month
    features['Date_year'] = features.Date.dt.year
    features = dataset.drop(["Date"], axis=1).copy()
    encoders = []
    for label in ["Date_day", "Date_month", "Date_year", "ProductName", "Country"]:
        features[label] = features[label].astype(str)
        features.loc[features[label] == "nan", label] = "unknown"
        encoder = LabelEncoder()
        features.loc[:, label] = encoder.fit_transform(features.loc[:, label].copy())
        encoders.append((label, encoder))
    return dict(features=features, transform_pipeline=encoders)


def split_dataset(dataset: pd.DataFrame, test_ratio: float) -> Dict[str, Any]:
    """
    Splits dataset into a training set and a test set.
    """
    X = dataset.drop("Price", axis=1)
    y = dataset["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=40
    )

    return dict(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)