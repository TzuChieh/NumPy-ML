import numpy as np
import numpy.typing as np_type


class Dataset:
    """
    Contains a set of data.
    """
    def __init__(
        self,
        xs: np_type.NDArray,
        ys: np_type.NDArray,
        batch_size=1):
        """
        The most basic form is to initialize with an input activation array `xs` and output activation array `ys`.
        Entries of the same index form a single sample of the data. For example, `xs[123]` and `ys[123]` will be a pair
        for input and output.
        """
        self.xs, self.ys = (xs, ys)
        self.batch_size = batch_size
        self.batched_xs, self.batched_ys = self._make_batches(xs, ys, batch_size)

    def shuffle(self, seed=None):
        """
        Randomly reorder the data entries.
        @param seed A seed for RNG.
        """
        rng = np.random.default_rng(seed)
        permuted_indices = rng.permutation(len(self))
        self.xs, self.ys = (self.xs[permuted_indices], self.ys[permuted_indices])
        self.batched_xs, self.batched_ys = self._make_batches(self.xs, self.ys, self.batch_size)

    def split(self, indices_or_sections):
        xs_splits = np.array_split(self.xs, indices_or_sections)
        ys_splits = np.array_split(self.ys, indices_or_sections)
        return [
            Dataset(
                xs, 
                ys, 
                batch_size=self.batch_size)
            for xs, ys in zip(xs_splits, ys_splits)]

    def batch_iter(self):
        return zip(self.batched_xs, self.batched_ys)

    @property
    def input_shape(self):
        assert len(self.xs) > 0

        # Get shape information from 0-th sample
        return self.xs[0].shape
    
    @property
    def output_shape(self):
        assert len(self.ys) > 0

        # Get shape information from 0-th sample
        return self.ys[0].shape
    
    @property
    def num_batches(self):
        """
        @return Number of batches.
        """
        assert len(self.batched_xs) == len(self.batched_ys)
        return len(self.batched_xs)

    def __len__(self):
        """
        @return Number of data samples.
        """
        assert len(self.xs) == len(self.ys)
        return len(self.xs)

    def __getitem__(self, key):
        return Dataset(
            self.xs[key], 
            self.ys[key],
            batch_size=self.batch_size)
    
    @staticmethod
    def _make_batches(xs: np_type.NDArray, ys: np_type.NDArray, batch_size):
        """
        Create batched views into the given samples. If the batch size is not divisible to the number of samples, the size of
        each batch will not equal.
        """
        assert batch_size >= 1

        batched_xs = []
        batched_ys = []
        for di in range(0, len(xs), batch_size):
            batched_xs.append(xs[di : di + batch_size])
            batched_ys.append(ys[di : di + batch_size])

        return (batched_xs, batched_ys)
