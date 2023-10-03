"""
Utilities for evaluating braids in the quotients of the Jones representations, i.e.
two-rowed partitions of the form (n - r, r) mod p for some prime p (or zero for no mod).

Previously a member of braidsearch, braidsearch2, braidsearch3.
"""
from __future__ import annotations

import abc
import functools
import math
import operator
from typing import Sequence

import numpy as np

from . import polymat, tl
from .braid import DGNF, GNF, NFBase
from .lpoly import LPoly
from .matrix import Matrix
from .noncrossing import NPar
from .permutations import Permutation


def tl_gens_invs(n):
    """
    Generators for the braid group B_n inside TL(n, n).
    The convention used is that sigma_i maps to 1 - ve_i, which makes each matrix in the
    Burau have determinant (-v^2), independent of n.
    """
    v = LPoly(0, (0, 1))
    es = [tl.TLMor.e(i, n) for i in range(n - 1)]
    gens = [-v*(e - v**-1) for e in es]
    invs = [-v**-1 * (e - v) for e in es]
    assert all(x*y == tl.TLMor.id(n) for x, y in zip(gens, invs))
    return gens, invs


def cell_matrix(tlmor, r):
    """
    Given a TLMor of type (n, n), return the matrix of its
    action on the cell module with n - 2r strands.
    """
    assert tlmor.top == tlmor.bot

    basis = tl.cell_basis(tlmor.bot, r)
    columns = [
        [prod.quotient(tlmor.top - 2*r).coefficient(b) for b in basis]
        for prod in [tlmor * b for b in basis]
    ]
    return Matrix.from_rows(tuple(zip(*columns)))

def cell_matrices(n, r):
    """Return matrices for the Artin generators and their inverses in (n-r, r)."""
    tlgens, tlinvs = tl_gens_invs(n)
    return [cell_matrix(g, r) for g in tlgens], [cell_matrix(g, r) for g in tlinvs]


class RepBase:
    n: int  # Number of strands in braid group.
    p: int  # Prime p to reduce by.

    def __init__(self):
        self._polymat_braid_factor = functools.cache(self._polymat_braid_factor)
        self._polymat_delta_power = functools.cache(self._polymat_delta_power)
        self.polymat_artin_gens_invs = functools.cache(self.polymat_artin_gens_invs)

    @abc.abstractmethod
    def dimension(self) -> int:
        """Dimension of the representation."""

    @abc.abstractmethod
    def artin_gens_invs(self) -> tuple[list[Matrix], list[Matrix]]:
        """Return the σ_i and σ_i^-1 matrices."""

    def id(self) -> Matrix:
        """The identity matrix for this representation."""
        return Matrix.identity(self.dimension())

    def polymat_dtype(self):
        """The dtype which should be used for elements of this representation."""
        return np.int32 if self.p > 0 else np.int64

    def evaluate_artin_word(self, word: Sequence[tuple[int, int]]):
        """Evaluate a word in the Artin generators in this representation."""
        gens, invs = self.artin_gens_invs()
        product: Matrix = functools.reduce(operator.mul, [gens[i]**e if e >= 0 else invs[i]**abs(e) for i, e in word], self.id())
        if self.p != 0:
            product = product.map(lambda x: x if isinstance(x, int) else x.map(lambda n: n % self.p))

        return product

    def evaluate(self, obj: Permutation | NPar | GNF | DGNF) -> Matrix:
        """
        Evalutate a permutation, noncrossing partition, or braid in this representation.
        This function is not fast, and should not be used for bulk operations.
        """
        return self.evaluate_artin_word(obj.artin_word())

    def polymat_id(self):
        """The identity matrix for this representation, as a polymat."""
        return np.identity(self.dimension(), dtype=self.polymat_dtype())[..., None]

    def polymat_mul(self, left: np.ndarray, right: np.ndarray):
        """Multiply two polymats in this representation (performs mod p if p != 0, and projectivisation)."""
        product = polymat.mul(left, right)
        if self.p != 0:
            product = product % self.p

        return polymat.projectivise(product)

    def polymat_artin_gens_invs(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns two NDArrays of shape (n-1, dim, dim, ?) where dim is the dimension of the representation.
        The first NDArray are the images of the Artin generators in the representation, while the second NDArray
        are the images of their inverses.
        """
        gens, invs = self.artin_gens_invs()
        matgens = polymat.pack([polymat.from_matrix(x, dtype=self.polymat_dtype()) for x in gens])
        matinvs = polymat.pack([polymat.from_matrix(x, dtype=self.polymat_dtype()) for x in invs])
        if self.p != 0:
            matgens %= self.p
            matinvs %= self.p

        return polymat.projectivise(matgens), polymat.projectivise(matinvs)

    def _polymat_braid_factor(self, cls: type[NFBase], factor: int) -> np.ndarray:
        """Low-level internal function for returning the matrix of a particular normal form factor."""
        mat = self.evaluate_artin_word(cls._nf_table(self.n).artin_word[factor])
        return polymat.projectivise(polymat.from_matrix(mat, proj=True))

    def _polymat_delta_power(self, cls: type[NFBase], power: int) -> np.ndarray:
        """
        Low-level internal function for returning the matrix of the Garside element to a power.
        Assumes that Δ^2 = δ^n = id projectively, and so will error if a high power is given.
        """
        table = cls._nf_table(self.n)
        assert 0 <= power < table.tau_order, "Only reduced powers should be passed to this function."
        mat = self.evaluate_artin_word(table.artin_word[table.D])
        return polymat.projectivise(polymat.from_matrix(mat ** power, proj=True))

    def polymat_evaluate_artin_word(self, word: Sequence[tuple[int, int]]):
        """Evaluate a word in the Artin generators in this representation, as a polymat."""
        gens, invs = self.polymat_artin_gens_invs()
        product = self.polymat_id()
        for i, e in word:
            for _ in range(abs(e)):
                product = polymat.mul(product, gens[i] if e >= 0 else invs[i])
                if self.p != 0:
                    product %= self.p

        return polymat.projectivise(product)

    def polymat_evaluate(self, obj: Permutation | NPar | NFBase):
        """
        Evalutate a permutation, noncrossing partition, or braid in this representation.
        This function is not fast, and should not be used for bulk operations.
        """
        return self.polymat_evaluate_artin_word(obj.artin_word())

    def polymat_evaluate_braid(self, braid: NFBase) -> np.ndarray:
        """Evaluate a braid in the representation."""
        return self.polymat_evaluate_braids_of_same_length([braid])[0]


    def polymat_evaluate_braids_of_same_length(self, braids: Sequence[NFBase]) -> np.ndarray:
        """
        Evaluate a list of braids of a uniform length, returning a single NDArray. The braids
        must also be of a uniform kind, either all GNFs or DGNFs.
        """
        length = braids[0].canonical_length()
        cls = braids[0].__class__
        assert all(braid.canonical_length() == length and braid.__class__ == cls for braid in braids)

        # First get all the powers of delta packed together.
        table = cls._nf_table(self.n)
        result = polymat.pack([self._polymat_delta_power(cls, braid.power % table.tau_order) for braid in braids])
        for i in range(length):
            images = polymat.pack([self._polymat_braid_factor(cls, braid.factors[i]) for braid in braids])
            result = self.polymat_mul(result, images)

        return result


    def polymat_evaluate_braid_integers_of_same_length(self, cls: type[GNF] | type[DGNF], sequences: Sequence[Sequence[int]]) -> np.ndarray:
        """
        If type is GNF, then sequences should be a list of lists, all of the same length, containing indices of permutations.
        For example in B4, all indices should be in the range [0, 4! = 24). If we were evaluating three braids, each made out of
        6 positive lifts of permutations, an input might look like cls=GNF, and sequences:
        [[0, 1, 2, 3, 4, 5],
         [2, 4, 23, 22, 21, 1],
         [3, 4, 5, 7, 8, 9]]

        This would produce a numpy array of shape (3, dim, dim, deg) where dim is the dimension of the representation, and deg is the
        maximum degree of any of the three resulting matrices.
        """
        # Check all sequences are the same length.
        length = len(sequences[0])
        assert all(len(seq) == length for seq in sequences)

        # Check all sequences have indices in the correct range.
        table = cls._nf_table(self.n)
        order = table.order
        assert all(0 <= x < order for seq in sequences for x in seq)

        # Evaluate the braids.
        result = polymat.pack([self._polymat_braid_factor(cls, table.id)])
        for i in range(length):
            images = polymat.pack([self._polymat_braid_factor(cls, seq[i]) for seq in sequences])
            result = self.polymat_mul(result, images)

        return result


class JonesCellRep(RepBase):
    """
    Specifies a representation of a braid group, along with whether that representation should be mod p.
    Has functions for evaluating permutations, noncrossing partitions, and braids in this representation,
    both as regular matrices over Laurent polynomials, and as numpy-packed polymats.

    The polymat representations are projective in the sense that they ignore uniform scaling by v.
    """
    n: int             # Number of strands of the braid group.
    r: int             # Partition labelling the representation is (n - r, r), for n - 2r ≥ 0.
    p: int = 0         # Zero (coefficients are int64) or a prime (coefficients are int32 mod p).

    def __init__(self, n: int, r: int, p: int = 0):
        super().__init__()

        assert n >= 1
        assert n - 2 * r >= 0
        assert p >= 0

        self.n = n
        self.r = r
        self.p = p


    def __repr__(self):
        ring = 'ℤ' if self.p == 0 else f'F{self.p}'
        return f'Two-rowed representation ({self.n - self.r}, {self.r}) of B{self.n}, over {ring}'

    def dimension(self) -> int:
        """The dimension of this representation."""
        # Unfortunately math.comb errors if it gets a negative second argument. Hence this if statement.
        if self.r == 0:
            return 1

        return math.comb(self.n, self.r) - math.comb(self.n, self.r - 1)

    def tl_basis(self) -> list[tl.TLMor]:
        return tl.cell_basis(self.n, self.r)

    def artin_gens_invs(self) -> tuple[list[Matrix], list[Matrix]]:
        gens, invs = cell_matrices(self.n, self.r)

        # Use this to go back to the q-representation.
        # zero, mv, mvinv = LPoly(0, ()), LPoly(1, (-1,)), LPoly(-1, (-1,))
        # fwd = Matrix.from_rows([
        #     [mv, zero, zero],
        #     [zero, mv**2, zero],
        #     [zero, zero, mv**3]
        # ])
        # rev = Matrix.from_rows([
        #     [mvinv, zero, zero],
        #     [zero, mvinv**2, zero],
        #     [zero, zero, mvinv**3]
        # ])
        # fwd, rev = rev, fwd
        # gens = [rev * g * fwd for g in gens]
        # invs = [fwd * i * rev for i in invs]
        if self.p != 0:
            gens = [gen.map(lambda x: x if isinstance(x, int) else x.map(lambda n: n % self.p)) for gen in gens]
            invs = [inv.map(lambda x: x if isinstance(x, int) else x.map(lambda n: n % self.p)) for inv in invs]

        return gens, invs


class LKBRep(RepBase):
    def __init__(self, n: int, p: int, q: LPoly, t: LPoly):
        super().__init__()
        assert n >= 1
        assert p >= 0
        assert q.is_unit()
        assert t.is_unit()

        self.n = n
        self.p = p
        self.q = q
        self.t = t

    def dimension(self):
        return math.comb(self.n, 2)

    def artin_gens_invs(self) -> tuple[list[Matrix], list[Matrix]]:
        t, q = self.t, self.q
        self.dimension()
        gens = []
        for s in range(self.n - 1):
            gen: dict[tuple[int, int], dict[tuple[int, int], LPoly | int]] = {(i, j): {} for i in range(self.n) for j in range(i + 1, self.n)}
            for (i, j), col in gen.items():
                if i == s and j == s+1:
                    col[i, j] = t * q * q
                elif i < s and j == s:
                    col[i, j] = 1 - q
                    col[i, j+1] = q
                elif i < s and j == s + 1:
                    col[i, j-1] = 1
                    col[j-1, j] = t * q**(j - i) * (q - 1)
                elif i == s and s + 1 < j:
                    col[i, i+1] = t * q * (q - 1)
                    col[i+1, j] = q
                elif i == s + 1 and s + 1 < j:
                    col[i-1, j] = 1
                    col[i, j] = 1 - q
                elif i < j < s or s + 1 < i < j:
                    col[i, j] = 1
                elif i < s < s+1 < j:
                    col[i, j] = 1
                    col[s, s + 1] = t * q**(s - i) * (q - 1)**2
                else:
                    raise ValueError(f"Did not consider {(i, j)=} when {s=}")

            mat = Matrix.from_rows([
                [LPoly.coerce(gen[col].get(row, 0)) for col in gen]
                for row in gen
            ])
            gens += [mat]

        return gens, gens
