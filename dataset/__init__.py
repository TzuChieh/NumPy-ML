import numpy as np
import numpy.typing as np_type


class Dataset:
    """
    Contains a set of data. The most basic form is a pair contains input activations (xs) and
    output activations (ys).
    """
    def __init__(self, xs: np_type.NDArray, ys: np_type.NDArray):
        self.xs = xs
        self.ys = ys

    def shuffle(self, seed=None):
        """
        Randomly reorder the data pairs.
        """
        permuted_indices = np.random.default_rng(seed).permutation(len(self))
        self.xs = self.xs[permuted_indices]
        self.ys = self.ys[permuted_indices]

    def split(self, indices_or_sections):
        xs_splits = np.array_split(self.xs, indices_or_sections)
        ys_splits = np.array_split(self.ys, indices_or_sections)
        return [Dataset(xs, ys) for xs, ys in zip(xs_splits, ys_splits)]

    @property
    def input_shape(self):
        assert len(self.xs) > 0
        return self.xs[0].shape
    
    @property
    def output_shape(self):
        assert len(self.ys) > 0
        return self.ys[0].shape

    def __len__(self):
        assert len(self.xs) == len(self.ys)
        return len(self.xs)

    def __getitem__(self, key):
        return Dataset(
            self.xs[key], 
            self.ys[key])
