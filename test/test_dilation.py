import common.vector as vec

import numpy as np
import pytest


@pytest.mark.parametrize(
    "matrix_shape, stride_shape, pad_shape, expected", [
    ((2, 2),       (1, 1),       (0, 0),    (2, 2)),
    ((1, 1),       (1, 1),       (0, 0),    (1, 1)),
    ((4, 4),       (1, 1),       (0, 0),    (4, 4)),
    ((1, 1),       (1, 1),       (1, 1),    (3, 3)),
    ((1, 1),       (2, 2),       (1, 1),    (3, 3)),
    ((1, 1),       (7, 7),       (1, 1),    (3, 3)),
    ((2, 2),       (2, 2),       (2, 2),    (7, 7)),
    ((3, 4),       (1, 2),       (2, 1),    (7, 9)),
])
def test_dilate_shape_2d(matrix_shape, stride_shape, pad_shape, expected):
    result = vec.dilate_shape(matrix_shape, stride_shape, pad_shape)
    assert np.array_equal(result, expected)
    
@pytest.mark.parametrize(
    "matrix_shape,     stride_shape, pad_shape, expected", [
    ((2, 2, 2),        (1, 1),       (0, 0),    (2, 2, 2)),
    ((999, 3, 4),      (1, 2),       (2, 1),    (999, 7, 9)),
    ((111, 222, 4, 4), (1, 1),       (0, 0),    (111, 222, 4, 4)),
])
def test_dilate_shape_nd(matrix_shape, stride_shape, pad_shape, expected):
    result = vec.dilate_shape(matrix_shape, stride_shape, pad_shape)
    assert np.array_equal(result, expected)

@pytest.mark.parametrize(
    "matrix, stride_shape, pad_shape, expected", [
    (np.full((3, 3), 1), (1, 1), (0, 0), np.array([
        [1, 1, 1],
        [1, 1, 1], 
        [1, 1, 1]])),
    (np.full((3, 3), 1), (1, 1), (1, 0), np.array([
        [0, 0, 0], 
        [1, 1, 1], 
        [1, 1, 1], 
        [1, 1, 1], 
        [0, 0, 0]])),
    (np.full((3, 3), 1), (1, 1), (0, 1), np.array([
        [0, 1, 1, 1, 0], 
        [0, 1, 1, 1, 0], 
        [0, 1, 1, 1, 0]])),
    (np.full((3, 4), 1), (2, 3), (1, 1), np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])),
])
def test_dilate_2d(matrix, stride_shape, pad_shape, expected):
    result = vec.dilate(matrix, stride_shape, pad_shape)
    assert np.array_equal(result, expected)
    