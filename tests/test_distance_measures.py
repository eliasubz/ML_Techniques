import numpy as np
import pandas as pd
import pytest
from distance_measures import euclidean_distance, cosine_distance


@pytest.fixture
def sample_data():
    X = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    instance = pd.Series({'a': 1, 'b': 4})
    return X, instance


def test_euclidean_distance(sample_data):
    X, instance = sample_data
    result = euclidean_distance(X, instance)
    assert np.isclose(result.loc[0, 'Distance'], 0.0)
    assert np.isclose(result.loc[1, 'Distance'], np.sqrt(2))
    assert np.isclose(result.loc[2, 'Distance'], np.sqrt(8))


def test_cosine_distance(sample_data):
    X, instance = sample_data
    result = cosine_distance(X, instance)
    assert np.isclose(result.loc[0, 'Distance'], 0.0)
    assert (result['Distance'] >= 0).all()
    assert (result['Distance'] <= 2).all()
