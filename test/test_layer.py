import common as com
from model.layer import Pool

import pytest
import numpy as np


@pytest.mark.parametrize(
    "kernel_shape", [
    ((2, 2)),
    ((1, 1)),
    ((3, 3)),
])
def test_max_pool_layer_single_output(kernel_shape):
    """
    Tests the most basic form: input is of the same shape as kernel (pool), so there is only a single output
    value (which is the maximum).
    """
    rng = np.random.default_rng(1325)
    input_shape = kernel_shape
    output_shape = (1, 1)
    max_pool = Pool(input_shape, kernel_shape, mode=com.EPooling.MAX, stride_shape=None)

    # Random inputs in [-10, 10]
    xs = [rng.random(input_shape) * 20 - 10 for i in range(100)]

    # Forward pass gives the maximum
    for x in xs:
        z = max_pool.weighted_input(x, None)
        assert np.array_equal(z.shape, output_shape)
        assert z[0] == x.max()

    # No trainable parameters
    for x in xs:
        d_params = max_pool.derived_params(x, np.ones(output_shape), None)
        for d_param in d_params:
            assert d_param.size == 0

    # Backward pass sets gradient (1) to the maximum element only
    for x in xs:
        dCdx = max_pool.backpropagate(x, np.ones(output_shape), None)

        # Subtracing the set gradient (1) off we should get a zero array
        dCdx.flat[dCdx.argmax()] -= 1
        assert np.array_equal(dCdx, np.zeros(input_shape))

@pytest.mark.parametrize(
    "batch_size, input_shape, kernel_shape, stride_shape", [
    (1, (4, 4), (2, 2), (2, 2)),
    (1, (2, 2), (1, 1), (1, 1)),
    (1, (6, 6), (3, 3), (3, 3)),
])
def test_max_pool_layer(batch_size, input_shape, kernel_shape, stride_shape):
    """
    Tests the general case.
    """
    rng = np.random.default_rng(7264)
    max_pool = Pool(input_shape, kernel_shape, mode=com.EPooling.MAX, stride_shape=stride_shape)

    ky, kx = kernel_shape
    sy, sx = stride_shape
    output_shape = ((input_shape[-2] - ky) // sy + 1, (input_shape[-1] - kx) // sx + 1)
    assert np.array_equal(output_shape, max_pool.output_shape)

    # Random inputs in [-10, 10]
    bxs = [rng.random((batch_size, *input_shape)) * 20 - 10 for i in range(100)]

    # Forward pass gives the maximum
    for bx in bxs:
        bz = max_pool.weighted_input(bx, None)
        for bi in range(batch_size):
            for pool_y in range(output_shape[-2]):
                for pool_x in range(output_shape[-1]):
                    # Find maximum in pool by brute force
                    pool = bx[bi, pool_y * sy : pool_y * sy + ky, pool_x * sx : pool_x * sx + kx]
                    pool_max = pool.max()
                    assert bz[bi, pool_y, pool_x] == pool_max

    # No trainable parameters
    for bx in bxs:
        b_d_params = max_pool.derived_params(bx, np.ones((batch_size, *output_shape)), None)
        for d_params in b_d_params:
            for d_param in d_params:
                assert d_param.size == 0

    # Backward pass sets gradient (1) to the maximum element only
    for bx in bxs:
        b_dCdx = max_pool.backpropagate(bx, np.ones((batch_size, *output_shape)), None)

        # Backpropagation by brute force
        b_dCdx_bf = np.zeros(b_dCdx.shape)
        for bi in range(batch_size):
            for pool_y in range(output_shape[-2]):
                for pool_x in range(output_shape[-1]):
                    # Find index of the maximum element
                    pool = bx[bi, pool_y * sy : pool_y * sy + ky, pool_x * sx : pool_x * sx + kx]
                    pool_max_i = pool.argmax()

                    # Increment resulting gradient by 1
                    pool = b_dCdx_bf[bi, pool_y * sy : pool_y * sy + ky, pool_x * sx : pool_x * sx + kx]
                    pool.flat[pool_max_i] += 1

        assert np.array_equal(b_dCdx_bf, b_dCdx)

    