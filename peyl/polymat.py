"""
polymat: functions for matrices with polynomial entries.

A matrix M with polynomial entries like (1 + v^2) can be treated as polynomial in regular
matrices M = M_0 + M_1 v + M_2 v^2 + ... + M_k v^k, and hence as a list [M_0, ..., M_k]
of regular matrices. We pack this into a 3D numpy tensor indexed like (i, j, d), where the
(i, j) directions are the regular matrix direction, and (d) indicates the degree.
"""

from typing import Sequence, Tuple

import numpy as np
import numpy.typing as npt

from .lpoly import LPoly
from .matrix import Matrix


def eye(n: int):
    """
    The identity polymat, of shape (n, n, 1).
    """
    return np.identity(n, dtype=int)[:, :, None]


def trim(A: npt.NDArray):
    """
    Trim trailing zeros from a polymat.
    (α, D) ↦ (α, K), where K is least such that A[..., K:] consists entirely of zeros.
    """
    last = A.shape[-1]
    while last > 0 and not np.any(A[..., last - 1]):
        last -= 1

    return A[..., :last] if last < A.shape[-1] else A


def trim_left(A: npt.NDArray):
    """
    Trim leading zeros from a polymat, i.e. divide by v^d where d is the least degree occurring in
    any matrix. Note that this changes the value of the matrix.
    (α, D) ↦ d, (α, K), where K is least such that A[..., :K] consists entirely of zeros, and
    d is the degree which we divided out by.
    """
    start, last = 0, A.shape[-1]
    while start < last and not np.any(A[..., start]):
        start += 1

    return start, (A[..., start:] if start != 0 else A)


def mul(A: npt.NDArray, B: npt.NDArray, quotdeg: int = None):
    """
    Calculate the product AB where A, B are polynomial matrices.
    In short notation this function has type

        (α, I, J, D) × (β, J, K, E) ↦ (α · β, I, K, P)

    where (α · β) denotes the broadcasting of shapes according to numpy rules. The last dimension
    P is not necessarily minimal: use trim() or projectivise() to normalise the result.
    """
    assert len(A.shape) >= 3 and len(B.shape) >= 3, "Inputs must have length at least 3."
    *alpha, I, J, D = A.shape
    *beta, J2, K, E = B.shape
    assert J == J2, f"The matrix dimensions of A and B must be compatible, was given {J} and {J2}."
    assert quotdeg is None or quotdeg >= 0, f"quotdeg={quotdeg} is illegal."

    P = D + E - 1 if quotdeg is None else quotdeg
    prefix = np.broadcast_shapes(alpha, beta)

    # This approach seems to be a bit faster than making a large shape of zeros of the right size
    # and filling in each part.
    parts = np.zeros((*prefix, I, K, P), dtype=A.dtype)
    for d in range(P):
        parts[..., d] = sum(A[..., i] @ B[..., d - i] for i in range(d + 1) if i < D and d - i < E)

    return parts


def pack(As: Sequence[npt.NDArray]):
    """
    Given a list of polynomial matrices of size (I, J, *), pack them into a single tensor of shape
    (L, I, J, D), where L is the length of the list, and D is the maximum degree bound which occurs
    in one of them.
    """
    assert len(As) >= 1
    assert all(len(A.shape) == 3 for A in As)
    I, J, _ = As[0].shape
    assert all(A.shape[:2] == (I, J) for A in As)

    As = [trim(A) for A in As]
    D = max(A.shape[-1] for A in As)

    # I can't find a way to easily "pad by zeros" along a particular axis in numpy. But by now we
    # know the final shape (len(As), I, J, D) of the returned array, so let's just make an array
    # of zeros and fill it in piece by piece.
    result = np.zeros((len(As), I, J, D), dtype=As[0].dtype)
    for i, A in enumerate(As):
        result[i, :, :, :A.shape[-1]] = A

    return result


def zeropad(A: npt.NDArray, D: int):
    """(α, I, J, L) ↦ (α, I, J, D) by padding with zeros. Must have L ≤ D or an error will be thrown."""
    assert len(A.shape) >= 3 and A.shape[-1] <= D
    result = np.zeros((*A.shape[:-1], D), dtype=A.dtype)
    result[..., :A.shape[-1]] = A
    return result


def concatenate(As: Sequence[npt.NDArray]):
    """
    Similar to pack, but for concatenating batches of matrices. np.concatenate cannot be used, because
    the last dimension (degree dimension) of each may not agree - it needs to be enlarged appropriately.
    """
    if len(As) == 0:
        return np.zeros((0, 1, 1, 1), dtype=int)
    assert all(len(A.shape) == 4 for A in As)
    _, I, J, _ = As[0].shape
    assert all(A.shape[1:3] == (I, J) for A in As)

    As = [trim(A) for A in As]
    D = max(A.shape[-1] for A in As)

    return np.concatenate([zeropad(A, D) for A in As], axis=0)


def burau_generator_v2(n: int, s: int):
    """
    Return the polynomial matrix corresponding to the generator 0 ≤ s < n - 1 in the Burau representation,
    using the normalisation where the determinant of each generator is (-v^2) for any n.
    """
    assert n >= 1
    assert 0 <= s < n - 1
    return np.array([
        [
            [0, -1, 0] if i == s and abs(j - s) == 1 else
            [0, 0, -1] if i == s == j else
            [1, 0, 0] if i == j else
            [0, 0, 0]
            for j in range(n - 1)
        ]
        for i in range(n - 1)
    ], dtype=int)


def to_matrix(A):
    """Return a polynomial matrix A as a regular matrix over the ring of Laurent polynomials."""
    assert len(A.shape) == 3
    I, J, _ = A.shape
    return Matrix.from_rows([
        [LPoly(0, A[i, j, :]) for j in range(J)]
        for i in range(I)
    ])

def from_matrix(mat: Matrix, proj: bool = False, dtype: npt.DTypeLike = np.int64):
    """
    Convert a matrix over the integers or Laurent polynomials to a polynomial matrix.
    If proj is true, then the matrix will be multiplied by a power of v, making it a polynomial matrix first.
    """
    if all(isinstance(x, int) for x in mat.data):
        return np.array(mat.rows())[..., None]

    if all(isinstance(x, LPoly) for x in mat.data):
        if all(x.is_zero() for x in mat.data):
            return np.zeros((mat.nrows, mat.ncols, 1), dtype=int)

        minval = min(x.valuation() for x in mat.data if not x.is_zero())
        if minval < 0:
            if proj:
                mat = Matrix(mat.nrows, mat.ncols, tuple(x * LPoly(-minval, (1,)) for x in mat.data))
                minval = min(x.valuation() for x in mat.data if not x.is_zero())
            else:
                raise ValueError("Cannot convert a matrix with negative powers to a polynomial matrix.")

        maxdeg = max(x.degree() for x in mat.data if not x.is_zero())
        if minval < 0:
            raise ValueError("Cannot convert a matrix with negative powers to a polynomial matrix.")

        A = np.zeros((mat.nrows, mat.ncols, maxdeg + 1), dtype=dtype)
        for i in range(mat.nrows):
            for j in range(mat.ncols):
                x = mat[i, j]
                A[i, j, x.val:x.val+len(x.coeffs)] = x.coeffs

        return A


def projectivise(A: npt.NDArray):
    """
    (α, I, J, L) ↦ (α, I, J, D). Shifts each matrix down if it is divisible by v^d, so that the result
    has a constant term.
    """
    # Ensure the input array has the right shape, and that no polynomial matrix is zero.
    assert len(A.shape) >= 3
    assert np.all(np.any(A, axis=(-3, -2, -1)))

    # The input may already be projectivised, which we can check by the existence of a nonzero entry in
    # every first term, and a nonzero entry in some last term.
    if np.all(np.any(A[..., 0], axis=(-2, -1))) and np.any(A[..., -1]):
        return A

    starts, ends = proj_starts_ends(A)
    new_width = np.max(ends - starts)

    if len(A.shape) == 3:
        return A[..., starts:ends] # type: ignore

    result = np.zeros((*A.shape[:-1], new_width), dtype=A.dtype)
    for index in np.ndindex(A.shape[:-3]):
        start, end = starts[index], ends[index]
        # Need to use two square brackets here since index might be a tuple like (3, 2)
        result[index][..., :end-start] = A[index][..., start:end]

    return result





def degmax(A):
    """Return the maximum degree of the polynomial matrix A. Broadcasts along a prefix."""
    assert len(A.shape) >= 3

    # Argmax will return the first occurrence of the maximum (this array contains False or True,
    # so the first occurrence of True). We want the last occurence, so we need to reverse the array,
    # take the argmax, and then fix up the indexing at the end.
    return A.shape[-1] - 1 - np.argmax(np.any(A[..., ::-1], axis=(-3, -2)), axis=-1)


def valmin(A):
    """Return the minimum valuation of the polynomial matrix A. Broadcasts along a prefix."""
    assert len(A.shape) >= 3
    return np.argmax(np.any(A, axis=(-3, -2)), axis=-1)


def nz_terms(A):
    """Return the number of nonzero terms of the polynomial matrix A. Broadcasts along a prefix."""
    assert len(A.shape) >= 3
    return np.count_nonzero(A, axis=(-3, -2, -1))


def projrank(A: npt.NDArray, terms: int):
    """
    Returns the sum of ranks (over characteristic zero) of the first few matrices.
    Eg for A = (A_0, A_1, ..., A_8, A_9), the projrank with terms=2 would be the rank(A_0) + rank(A_1)
    """
    assert terms >= 0
    assert len(A.shape) >= 3 and A.shape[-1] >= 1

    # For now let's assume that outputs are maybe not a projective block, but all their zero degrees are in the right place.
    assert np.equal(valmin(A), 0).all()

    return sum(np.linalg.matrix_rank(A[..., i]) for i in range(terms) if i < A.shape[-1])


def rhogap(A: npt.NDArray):
    """Return the degmax minus the valmin on 2."""
    return (degmax(A) - valmin(A)) / 2

def efflen(A: npt.NDArray):
    """degmax - valmin, or zero if A == 0."""
    return np.where(
        (A == 0).all(axis=(-3, -2, -1)),
        0,
        degmax(A) - valmin(A),
    )

def projlen(A: npt.NDArray) -> npt.NDArray:
    """
    degmax - valmin + 1, or zero if A == 0.
    Name inspired since ignoring powers of v^d is a bit like projectivising.
    (α, I, J, D) -> (α)
    """
    nonzeros = np.any(A, axis=(-3, -2))
    starts = np.argmax(nonzeros, axis=-1)
    ends = nonzeros.shape[-1] - np.argmax(nonzeros[..., ::-1], axis=-1)
    return ends - starts

def proj_starts_ends(A: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    """
        (α, I, J, D) → (α), (α)

    Returns two arrays giving the "starts and ends" of the projectivised matrices,
    so that if A[i] is a polymat, then A[i][starts[i]:ends[i]] is the projectivised
    version with the leading and trailing zeros chopped off.

    If α is empty, then scalars are returned.
    """
    assert len(A.shape) >= 3
    assert np.all(np.any(A, axis=(-3, -2, -1)))
    nonzeros = np.any(A, axis=(-3, -2))
    starts = np.argmax(nonzeros, axis=-1)
    ends = nonzeros.shape[-1] - np.argmax(nonzeros[..., ::-1], axis=-1)
    return starts, ends
