import common.vector as vec

import numpy as np
import pytest


@pytest.mark.parametrize(
    "a, num_lower_dims, expected", [
    (np.array([[1, 2], [3, 4]]), 1, (np.array([1, 1]),)),
    (np.array([[1, 2, 3], [4, 5, 6]]), 1, (np.array([2, 2]),)),
    (np.array([-3]), 1, (0,)),
])
def test_argmax_lower(a, num_lower_dims, expected):
    indices = vec.argmax_lower(a, num_lower_dims)
    assert np.array_equal(indices, expected)
    