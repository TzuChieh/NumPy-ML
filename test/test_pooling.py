import common as com
import common.vector as vec

import numpy as np
import pytest


@pytest.mark.parametrize(
    "matrix_shape, kernel_shape, stride_shape, expected", [
    ((2, 2),       (1, 1),       (1, 1),       (2, 2)),
    ((1, 1),       (1, 1),       (1, 1),       (1, 1)),
    ((1, 1),       (1, 1),       (1234, 5678), (1, 1)),
    ((5, 5),       (4, 4),       (1, 1),       (2, 2)),
    ((4, 4),       (2, 2),       (2, 2),       (2, 2)),
    ((5, 5),       (1, 2),       (2, 1),       (3, 4)),
    ((6, 8),       (3, 3),       (2, 3),       (2, 2)),
])
def test_pool_shape_2d(matrix_shape, kernel_shape, stride_shape, expected):
    result = vec.pool_shape(matrix_shape, kernel_shape, stride_shape)
    assert np.array_equal(result, expected)
    
@pytest.mark.parametrize(
    "matrix_shape,   kernel_shape, stride_shape, expected", [
    ((123, 5, 5),    (1, 2),       (2, 1),       (123, 3, 4)),
    ((55, 66, 6, 8), (3, 3),       (2, 3),       (55, 66, 2, 2)),
])
def test_pool_shape_nd(matrix_shape, kernel_shape, stride_shape, expected):
    result = vec.pool_shape(matrix_shape, kernel_shape, stride_shape)
    assert np.array_equal(result, expected)
    
@pytest.mark.parametrize(
    "matrix, kernel_shape, stride_shape, expected", [
    (np.full((3, 3), 1), (1, 1), (1, 1), np.full((3, 3), 1)),
    (np.full((3, 3), 2), (2, 2), (1, 1), np.full((2, 2), 2)),
    (np.full((2, 3), -3), (2, 2), (1, 1), np.full((1, 2), -3)),
    (np.full((3, 4), 999), (2, 3), (1, 1), np.full((2, 2), 999)),
    (np.full((3, 4), 0), (2, 3), (1, 2), np.full((2, 1), 0)),
    (np.full((2, 4), 2.345), (2, 2), (123, 2), np.full((1, 2), 2.345)),
    (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), (1, 2), (2, 1), np.array([[2, 3], [8, 9]])),
])
def test_max_pool_2d(matrix, kernel_shape, stride_shape, expected):
    result = vec.pool(matrix, kernel_shape, stride_shape, mode=com.PoolingMode.MAX)
    assert np.array_equal(result, expected)
    
@pytest.mark.parametrize(
    "matrix, kernel_shape, stride_shape, expected", [
    (np.full((3, 3), 1), (1, 1), (1, 1), np.full((3, 3), 1)),
    (np.full((3, 3), 2), (2, 2), (1, 1), np.full((2, 2), 2)),
    (np.full((2, 3), -3), (2, 2), (1, 1), np.full((1, 2), -3)),
    (np.full((3, 4), 999), (2, 3), (1, 1), np.full((2, 2), 999)),
    (np.full((3, 4), 0), (2, 3), (1, 2), np.full((2, 1), 0)),
    (np.full((2, 4), 2.345), (2, 2), (123, 2), np.full((1, 2), 2.345)),
    (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), (1, 2), (2, 1), np.array([[1.5, 2.5], [7.5, 8.5]])),
    (np.arange(10), (10,), (-654,), [4.5]),
])
def test_average_pool_2d(matrix, kernel_shape, stride_shape, expected):
    result = vec.pool(matrix, kernel_shape, stride_shape, mode=com.PoolingMode.AVERAGE)
    assert np.array_equal(result, expected)
    