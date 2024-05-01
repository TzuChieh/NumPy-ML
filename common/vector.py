import common as com

import numpy as np
import numpy.typing as np_type


def _make_kernel_dim_to_einsum_correlation_expr():
    chars = 'abcdefghijklmnopqrstuvwxyz'
    result = {}
    for dim in range(1, 26 + 1):
        subscripts = chars[:dim]
        result[dim] = f'...{subscripts},...{subscripts}->...'
    return result

_kernel_dim_to_einsum_correlation_expr = _make_kernel_dim_to_einsum_correlation_expr()


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

def correlate_shape(matrix_shape, kernel_shape, stride_shape=(1,)) -> np_type.NDArray:
    """
    Given the shapes, computes the resulting shape after the correlation. Will compute with the kernel's dimensions
    (other dimensions remain unchanged).
    @param kernel_shape Kernel dimensions.
    @param stride_shape Stride dimensions (for moving the kernel).
    """
    nd = len(kernel_shape)
    stride_shape = np.broadcast_to(stride_shape, nd)
    correlate_size = np.floor_divide(np.subtract(matrix_shape[-nd:], kernel_shape), stride_shape) + 1
    return (*matrix_shape[:-nd], *correlate_size)

def dilate_shape(matrix_shape, stride_shape, pad_shape=(0,)) -> np_type.NDArray:
    """
    Given the shapes, computes the resulting shape after the dilation. Will compute with the stride's and pad's
    dimensions (other dimensions remain unchanged).
    @param stride_shape Stride dimensions.
    @param pad_shape Pad dimensions.
    """
    stride_shape, pad_shape = np.broadcast_arrays(stride_shape, pad_shape)
    nd = len(stride_shape)
    skirt_size = np.multiply(pad_shape, 2)
    dilate_size = np.subtract(matrix_shape[-nd:], 1) * stride_shape + 1
    return (*matrix_shape[:-nd], *(skirt_size + dilate_size))

def pool_shape(matrix_shape, kernel_shape, stride_shape) -> np_type.NDArray:
    """
    Given the shapes, computes the resulting shape after pooling. Will compute with the pool's dimensions
    (other dimensions remain unchanged).
    @param kernel_shape Pool dimensions.
    @param stride_shape Stride dimensions.
    """
    return correlate_shape(matrix_shape, kernel_shape, stride_shape)

def sliding_window_view(matrix: np_type.NDArray, window_shape, stride_shape=(1,), is_writeable=False):
    """
    @param is_writeable Whether the returned view is writeable or not. For safety, the view is read-only. See
    `numpy.lib.stride_tricks.as_strided()` for more details. Basically, you at least need to ensure that the
    write-to locations are not overlapping in vectorized operations.
    """
    correlated_shape = correlate_shape(matrix.shape, window_shape, stride_shape)
    nd = len(correlated_shape)
    stride_shape = np.broadcast_to(stride_shape, nd)
    
    view_shape = (
        *correlated_shape[:-nd],
        *correlated_shape[-nd:],
        *window_shape)
    view_stride = (
        *matrix.strides[:-nd],
        *[stride_shape[di] * matrix.strides[di] for di in range(-nd, 0)],
        *matrix.strides[-nd:])
    return np.lib.stride_tricks.as_strided(matrix, view_shape, view_stride, writeable=False)

def dilate(matrix: np_type.NDArray, stride_shape, pad_shape=(0,)):
    """
    Dilate the matrix according to the specified shapes. Will compute with the stride's and pad's dimensions
    (broadcast the rest).
    @param stride_shape Stride dimensions.
    @param pad_shape Pad dimensions.
    """
    dilated_shape = dilate_shape(matrix.shape, stride_shape, pad_shape)
    
    # Create slice into the dilated matrix so we can assign `matrix` without looping
    step, pad = np.broadcast_arrays(stride_shape, pad_shape)
    nd = len(step)
    size = np.array(dilated_shape[-nd:])
    slices = tuple(slice(pd, sz - pd, st) for pd, sz, st in zip(pad, size, step))

    dilated_matrix = np.zeros(dilated_shape, dtype=matrix.dtype)
    dilated_matrix[..., *slices] = matrix
    return dilated_matrix
    
def correlate(matrix: np_type.NDArray, kernel: np_type.NDArray, stride_shape=(1,)):
    """
    Correlate the matrix according to the specified shapes. Will compute with the kernel's dimensions
    (broadcast the rest).
    @param kernel The kernel to correlate with.
    @param stride_shape Stride dimensions.
    """
    assert matrix.dtype == kernel.dtype, f"types: {matrix.dtype}, {kernel.dtype}"

    nd = len(kernel.shape)
    strided_view = sliding_window_view(matrix, kernel.shape, stride_shape)
    correlated = np.einsum(_kernel_dim_to_einsum_correlation_expr[nd], strided_view, kernel)

    assert np.array_equal(correlated.shape, strided_view.shape[:-nd]), f"shapes: {correlated.shape}, {strided_view.shape}"
    return correlated

def pool(matrix: np_type.NDArray, kernel_shape, stride_shape, mode: com.PoolingMode):
    """
    Perform pooling operation on the matrix according to the specified shapes. Will compute with the pool's dimensions
    (broadcast the rest).
    @param kernel_shape Pool dimensions.
    @param stride_shape Stride dimensions.
    """
    nd = len(kernel_shape)
    strided_view = sliding_window_view(matrix, kernel_shape, stride_shape)
    match mode:
        case com.PoolingMode.MAX:
            pooled = strided_view.max(axis=tuple(di for di in range(-nd, 0)))
        case com.PoolingMode.AVERAGE:
            pooled = strided_view.mean(axis=tuple(di for di in range(-nd, 0)))
        case _:
            raise ValueError("unknown pooling mode specified")

    assert np.array_equal(pooled.shape, strided_view.shape[:-nd]), f"shapes: {pooled.shape}, {strided_view.shape[:-nd]}"
    return pooled
