import joblib
import numpy as np
import pytest
from pydantic import ValidationError

from src.data import load_data
from src.schemas import HouseFeatures


def test_load_data():
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    assert X_train.shape[1] == 8  # 8 features
    assert len(X_train) > 0


@pytest.fixture
def trained_pipeline():
    return joblib.load("models/pipeline.pkl")


def test_pipeline_output_shape(trained_pipeline):
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    predictions = trained_pipeline.predict(X_test)
    assert predictions.shape == y_test.shape


def test_no_nan_in_predictions(trained_pipeline):
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    predictions = trained_pipeline.predict(X_test)
    assert not np.any(np.isnan(predictions))


def test_invalid_features():
    with pytest.raises(ValidationError):
        HouseFeatures(
            MedInc=-5,
            HouseAge=10,
            AveRooms=5,
            AveBedrms=1,
            Population=-100,
            AveOccup=3,
            Latitude=34,
            Longitude=-118,
        )
