import numpy as np


class PreallocatedMatrix:
    """
    Class to handle preallocated numpy arrays for efficient data storage."""

    def __init__(self, rows, cols, dtype=np.float64):
        print(f"Preallocating matrix of shape ({rows}, {cols})")
        self.arr = np.zeros((rows, cols), dtype=dtype)
        self.row = 0

    def append_matrix(self, matrix):
        """Append multiple columns (matrix) to the preallocated array."""
        if (self.arr.shape[1] != matrix.shape[1]):
            raise ValueError(
                "Matrix shape does not match preallocated array shape.")

        self.arr[self.row:self.row + matrix.shape[0], :] = matrix
        self.row += matrix.shape[0]

    def append_column(self, col_data):
        """Append a column of data"""
        self.arr[self.row, :] = col_data
        self.row += 1

    def get_filled(self):
        """Return only the filled columns"""
        return self.arr[:self.row, :]

    def shape(self):
        """Return the shape of the preallocated array"""
        return self.arr.shape
