from __future__ import annotations

import dataclasses
from typing import Callable


@dataclasses.dataclass(frozen=True)
class Matrix:
    """
    An immutable matrix of integers, suitable for use as a dictionary key. Matrices may be constructed in the following
    ways, for example direct construction::

    >>> Matrix(2, 3, (1, 2, 3, 4, 5, 6))
    Matrix([
        [1, 2, 3],
        [4, 5, 6],
    ])

    Construction from a list of rows::

    >>> Matrix.from_rows([[1, 2, 3], [4, 5, 6]])
    Matrix([
        [1, 2, 3],
        [4, 5, 6],
    ])

    Construction of special matrices::

    >>> Matrix.identity(2)
    Matrix([
        [1, 0],
        [0, 1],
    ])
    >>> Matrix.scalar(2, 6)
    Matrix([
        [6, 0],
        [0, 6],
    ])
    >>> Matrix.zero(2, 3)
    Matrix([
        [0, 0, 0],
        [0, 0, 0],
    ])

    """
    nrows: int
    ncols: int
    data: tuple[int, ...]

    def __post_init__(self):
        if not (self.nrows >= 0 and self.ncols >= 0):
            raise ValueError("Cannot have a negative number of rows or columns.")
        if not self.nrows * self.ncols == len(self.data):
            raise ValueError("Length of data incompatible")
        if not isinstance(self.data, tuple):
            raise ValueError("Data should be a tuple")

    def indices(self):
        return ((i, j) for i in range(self.nrows) for j in range(self.ncols))

    @classmethod
    def from_rows(cls, rows: list[list[int]]):
        """
        Construct a matrix from a list of lists of rows, which must have uniform dimensions.

        >>> Matrix.from_rows([[1, 2, 3], [4, 5, 6]]).rows()
        [[1, 2, 3], [4, 5, 6]]
        """
        nrows = len(rows)
        ncols = len(rows[0])
        assert all(len(row) == ncols for row in rows)
        return cls(nrows, ncols, tuple(x for row in rows for x in row))

    @classmethod
    def identity(cls, size: int):
        """
        >>> Matrix.identity(2).rows()
        [[1, 0], [0, 1]]
        """
        return cls(size, size, tuple(1 if i == j else 0 for i in range(size) for j in range(size)))

    @classmethod
    def zero(cls, nrows: int, ncols: int):
        """
        >>> Matrix.zero(2, 3)
        Matrix([
            [0, 0, 0],
            [0, 0, 0],
        ])
        """
        return cls(nrows, ncols, tuple(0 for _ in range(nrows * ncols)))

    @classmethod
    def row_vector(cls, elems: list[int]):
        return cls(1, len(elems), tuple(elems))

    @classmethod
    def col_vector(cls, elems: list[int]):
        return cls(len(elems), 1, tuple(elems))

    @classmethod
    def col_coord_vector(cls, n, i):
        """Return the coordinate vector e_i in n-dimensional space."""
        return cls(n, 1, tuple(1 if i == j else 0 for j in range(n)))

    @classmethod
    def scalar(cls, size: int, scalar: int):
        """
        Create an n x n scalar matrix: the diagonal matrix with every entry equal to the scalar.

        >>> Matrix.scalar(3, 6).rows()
        [[6, 0, 0], [0, 6, 0], [0, 0, 6]]
        """
        return cls(size, size, tuple(scalar if i == j else 0 for i in range(size) for j in range(size)))

    def row(self, row: int):
        """
        Return a row of the matrix as a row vector.
        """
        if not 0 <= row < self.ncols:
            raise IndexError(f"Row {row} is out of bounds for a {self.nrows} x {self.ncols} matrix.")

        return Matrix(1, self.ncols, tuple(self[row, j] for j in range(self.ncols)))

    def column(self, col: int):
        """
        Return a column of the matrix as a column vector.
        """
        if not 0 <= col < self.ncols:
            raise IndexError(f"Column {col} is out of bounds for a {self.nrows} x {self.ncols} matrix.")

        return Matrix(self.nrows, 1, tuple(self[j, col] for j in range(self.nrows)))

    def rows(self) -> list[list[int]]:
        """Return the matrix as a list of lists of rows.

        >>> Matrix.from_rows([[1, 2], [3, 4]]).rows()
        [[1, 2], [3, 4]]
        """
        return [list(self.data[self.ncols * i:self.ncols * (i + 1)]) for i in range(self.nrows)]

    def _checkbounds(self, i, j):
        if not (0 <= i < self.nrows and 0 <= j < self.ncols):
            raise IndexError(f"Index ({i}, {j}) out of range for matrix with dimensions ({self.nrows}, {self.ncols})")

    def __getitem__(self, key):
        """
        For a matrix M, M[i, j] returns the zero-indexed (i, j)th entry.

        >>> M = Matrix.from_rows([[1, 2, 3], [4, 5, 6]])
        >>> M[0, 0]
        1
        >>> M[1, 1]
        5
        """
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError(f"Supplied key {key!r} should be a tuple of length 2.")

        i, j = key
        self._checkbounds(i, j)
        return self.data[self.ncols * i + j]

    def __mul__(self, other):
        """
        >>> M = Matrix.from_rows([[1, 1], [1, 0]]) # Fibonacci matrix
        >>> (M*M*M*M*M*M).rows()
        [[13, 8], [8, 5]]
        """
        if isinstance(other, int):
            return Matrix(self.nrows, self.ncols, tuple(x*other for x in self.data))

        if isinstance(other, Matrix):
            if self.ncols != other.nrows:
                raise ValueError(f"Matrix dimensions incompatible: {(self.nrows, self.ncols)} * {(other.nrows, other.ncols)}")

            newdata = [0] * (self.nrows * other.ncols)
            for i in range(self.nrows):
                for j in range(other.ncols):
                    for k in range(self.ncols):
                        newdata[other.ncols * i + j] += self.data[self.ncols * i + k] * other.data[other.ncols * k + j]

            return Matrix(self.nrows, other.ncols, tuple(newdata))

        return NotImplemented

    def __pow__(self, exp: int):
        if self.nrows != self.ncols:
            raise ValueError("Can only take powers of square matrices")

        if exp < 0:
            raise NotImplementedError("Negative powers of matrices unimplemented")

        acc = Matrix.identity(self.nrows)
        pow2 = self
        while exp:
            if exp % 2 == 1:
                acc = acc * pow2

            exp //= 2
            pow2 = pow2 * pow2

        return acc



    def __rmul__(self, other):
        if isinstance(other, int):
            return Matrix(self.nrows, self.ncols, tuple(x*other for x in self.data))

        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Matrix):
            if (self.nrows, self.ncols) != (other.nrows, other.ncols):
                raise ValueError("Cannot subtract incompatibly sized matrices.")
            return Matrix(self.nrows, self.ncols, tuple(a - b for a, b in zip(self.data, other.data)))

    def trace(self):
        """
        The trace of a square matrix is the sum of the diagonal entries.

        >>> Matrix.from_rows([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).trace()
        15
        """
        if self.nrows != self.ncols:
            raise ValueError("Trace defined only for square matrices")

        return sum(self.data[self.ncols*i + i] for i in range(self.nrows))

    def is_square(self):
        return self.nrows == self.ncols

    def is_symmetric(self):
        return all(self[i, j] == self[j, i] for i, j in zip(range(self.nrows), range(self.ncols)))

    def __repr__(self):
        """
        >>> Matrix(5, 0, ())
        Matrix(5, 0, [])
        >>> Matrix.from_rows([[1, 2, 3, 4]])
        Matrix([[1, 2, 3, 4]])
        >>> Matrix.from_rows([[1], [2], [3], [4]])
        Matrix([[1], [2], [3], [4]])
        >>> Matrix.from_rows([[2, 3, 4], [5, 6, 7]])
        Matrix([
            [2, 3, 4],
            [5, 6, 7],
        ])

        :return:
        """
        if self.nrows == 0 or self.ncols == 0:
            return f'Matrix({self.nrows}, {self.ncols}, [])'
        if self.nrows == 1:
            return 'Matrix([[' + ', '.join(repr(c) for c in self.data) + ']])'
        if self.ncols == 1:
            return 'Matrix([' + ', '.join(f'[{c!r}]' for c in self.data) + '])'
        return '\n'.join([
            'Matrix([',
            *(
                '    [' + ', '.join(repr(self.data[self.ncols * i + j]) for j in range(self.ncols)) + '],'
                for i in range(self.nrows)
            ),
            '])'
        ])

    def _repr_latex_(self):
        def get_repr(x):
            return x._repr_latex_() if hasattr(x, '_repr_latex_') else repr(x)

        return ''.join([
            r'\begin{pmatrix}',
            r' \\ '.join(
                ' & '.join(
                    get_repr(self.data[i * self.ncols + j])
                    for j in range(self.ncols)
                )
                for i in range(self.nrows)
            ),
            r'\end{pmatrix}',
        ])

    def __str__(self):
        rows = ['[' + ', '.join(str(c) for c in row) + ']' for row in self.rows()]
        return f"M[{', '.join(rows)}]"

    def transpose(self):
        return Matrix(self.ncols, self.nrows, tuple(self[j, i] for i in range(self.ncols) for j in range(self.nrows)))

    def is_integral(self):
        return all(isinstance(c, int) for c in self.data)

    def entries(self):
        """Return an iterator over the entries of the matrix, in unspecified order."""
        return iter(self.data)

    def map(self, f: Callable[[int], int]):
        """Map a function over the entries of the matrix."""
        return Matrix(self.nrows, self.ncols, tuple(f(c) for c in self.data))
