from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def load_data():
    """load the data for california housing

    Returns:
        X_train: X train set of the data
        X_val: X val set of the data
        X_test: X test set of the data
        y_train: y train set of the data
        y_val: y val set of the data
        y_test: y test set of the data
    """
    housing = fetch_california_housing()

    X = housing.data
    y = housing.target

    train_ratio = 0.7
    test_ratio = 0.15
    val_ratio = 0.15 / (1 - 0.70)  # = 0.5 du reste soit les 0.15 du total

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_ratio, test_size=test_ratio, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
