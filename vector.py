import numpy as np
import numpy.typing as np_type


def zeros_from(m: np_type.NDArray):
    """
    Same as `numpy.zeros()`, except that settings are inferred from `m`.
    """
    return np.zeros(m.shape, dtype=m.dtype)

def vector_2d(m: np_type.NDArray):
    """
    Reshape the last two dimensions of the matrix into a column vector. Other dimensions remain unchanged.
    @param m The matrix to reshape into vector.
    """
    length = np.prod(m.shape[-2:])
    return m.reshape((*m.shape[:-2], length, 1))

def transpose_2d(m: np_type.NDArray):
    """
    Transpose the last two dimensions of a matrix. Other dimensions remain unchanged.
    @param m The matrix to transpose.
    """
    return np.swapaxes(m, -2, -1)

def correlate_shape(matrix_shape, kernel_shape, stride_shape) -> np_type.NDArray:
    """
    Given the shapes, computes the resulting shape after the correlation. Will compute with the last 2 dimemsions
    (broadcast the rest).
    @param kernel_shape Kernel dimensions.
    @param stride_shape Stride dimensions in 2-D.
    """
    correlate_size = np.floor_divide(np.subtract(matrix_shape[-2:], kernel_shape[-2:]), stride_shape) + 1
    return (*matrix_shape[:-2], *correlate_size)

def dilate_shape(matrix_shape, stride_shape, pad_shape=(0, 0)) -> np_type.NDArray:
    """
    Given the shapes, computes the resulting shape after the dilation. Will compute with the last 2 dimemsions
    (broadcast the rest).
    @param stride_shape Stride dimensions in 2-D.
    @param pad_shape Pad dimensions in 2-D.
    """
    skirt_size = np.multiply(pad_shape, 2)
    dilate_size = np.subtract(matrix_shape[-2:], 1) * stride_shape + 1
    return (*matrix_shape[:-2], *(skirt_size + dilate_size)) 

def dilate(matrix: np_type.NDArray, stride_shape, pad_shape=(0, 0)):
    """
    Dilate the matrix according to the specified shapes. Will compute with the last 2 dimemsions
    (broadcast the rest).
    @param stride_shape Stride dimensions in 2-D.
    @param pad_shape Pad dimensions in 2-D.
    """
    dilated_shape = dilate_shape(matrix.shape, stride_shape, pad_shape)
    
    # Create slice into the dilated matrix so we can assign `matrix` without looping
    pad = np.array(pad_shape)
    size = np.array(dilated_shape[-2:])
    step = np.array(stride_shape)
    slices = tuple(slice(pd, sz - pd, st) for pd, sz, st in zip(pad, size, step))

    dilated_matrix = np.zeros(dilated_shape, dtype=matrix.dtype)
    dilated_matrix[..., *slices] = matrix
    return dilated_matrix
    
def correlate(matrix: np_type.NDArray, kernel: np_type.NDArray, stride_shape=(1, 1)):
    """
    Correlate the matrix according to the specified shapes. Will compute with the last 2 dimemsions
    (broadcast the rest).
    @param kernel The kernel to correlate with.
    @param stride_shape Stride dimensions in 2-D.
    """
    assert matrix.dtype == kernel.dtype, f"types: {matrix.dtype}, {kernel.dtype}"

    correlated_shape = correlate_shape(matrix.shape, kernel.shape, stride_shape)
    
    view_shape = (
        *correlated_shape[:-2],
        *correlated_shape[-2:],
        *kernel.shape[-2:])
    view_stride = (
        *matrix.strides[:-2],
        stride_shape[-2] * matrix.strides[-2],
        stride_shape[-1] * matrix.strides[-1],
        *matrix.strides[-2:])
    strided_view = np.lib.stride_tricks.as_strided(matrix, view_shape, view_stride, writeable=False)
    correlated = np.einsum('...yxhw,...hw->...yx', strided_view, kernel)

    assert np.array_equal(correlated.shape, correlated_shape), f"shapes: {correlated.shape}, {correlated_shape}"
    return correlated
