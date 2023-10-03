"""
braid: algorithms and data structures for working with the braid group (associated to the symmetric group).

Each braid β is represented in its left Garside normal form Δ^d β(1) ... β(k), represented as the data
- The infimum d of the braid β, which can be any integer, together with
- The canonical factors [β(1), ..., β(k)], a list of permutations in the symmetric group, where k ≥ 0 is
  called the Garside length. Each permutation must not be the identity or the longest element, and each pair
  must satisfy the condition R(β(i)) ⊇ L(β(i+1)), where R and L represent the right and left descent sets.

When the braid group B_n is first created, we set up an arbitrary bijection between permutations S_n and
integers in the range [0, n!), and tabulate all the functions of interest to us. For example, right
multiplication by a generator becomes a function [0, n!) × [0, n-1) ↦ [0, n!). The canonical factors of a
braid are stored as list of integers under this bijection, which keeps the representations of braids compact,
and the relevant algorithms fast. (If we want to work in braid groups where n is large, a different approach
should be taken).
"""

from __future__ import annotations

import abc
import ast
import dataclasses
import functools
import itertools
import math
import operator
import random
from collections import defaultdict
from typing import (
    Generic,
    Iterable,
    Sequence,
    TypeVar,
)

from . import permutations
from .noncrossing import NPar, catalan
from .permutations import Permutation, SymmetricGroup

T = TypeVar('T')

@dataclasses.dataclass
class NFTable(Generic[T]):
    """
    An NFTable is a lookup table which enumerates the divisors Div(D) of the Garside element D, and
    creates tables for operations which are useful during normal form computations. The type
    variable T will either be Permutation, in which case D is the half-twist Δ and Div(Δ) are all
    permutations, or T will be NPar, in which case D is the descending cycle δ, and Div(δ) are the
    noncrossing partitions.

    At one point we make a table of size |Div(D)| × |Div(D)|, which is really only suitable for braid
    groups of size at most n = 8 or so.
    """
    n: int                     # Strands in braid group
    order: int                 # Number of divisors of D.
    divs: list[T]              # Map from an index to a divisor.
    index: dict[T, int]        # Map from a divisor to an index.
    id: int                    # Index of the identity.
    D: int                     # Index of the Garside element D.
    artin_gens: list[int]      # Maps i to σ(i).
    left_comp: list[int]       # Left complement *x satisfies D = β(*x) β(x)
    right_comp: list[int]      # Right complement x* satisfies D = β(x) β(x*)
    tau_order: int             # The order of conjugation by D.
    tau_left: list[list[int]]  # tau_left[i][x] = D^i x D^-i
    tau_right: list[list[int]] # tau_right[i][x] = D^-i x D^i

    # During renormalisation, need to convert adjacent pairs (x, y) to (w, z) by
    # transferring factors from the left of y to the right of x. The renorm table
    # has all possible pairs memorised. The is_normalised table recalls if a pair
    # is already in normal form. The follows table maps x to all letters y such
    # that (x, y) are in normal form.
    # For these tables, we count (D, x) to be in normal form for all x, except in
    # the follows table, we never allow id or D to appear in the resulting list.
    renorm: list[list[tuple[int, int]]]
    is_normalised: list[list[bool]]
    follows: list[list[int]]

    # A word in the Artin generators for each divisor.
    artin_word: list[list[tuple[int, int]]]

    def is_factors_normalised(self, factors: Sequence[int]) -> bool:
        """
        Return whether the list of factors is in normal form, meaning that
        no D's appear at the start, no identities appear at the back, and
        every other pair (x, y) is in normal form.
        """
        for i in range(len(factors) - 1):
            if not self.is_normalised[factors[i]][factors[i+1]]:
                return False
        return len(factors) == 0 or (factors[0] != self.D and factors[-1] != self.id)

    def normalise_factors(self, seq: Sequence[int]) -> tuple[int, tuple[int, ...]]:
        """
        Normalise an arbitrary sequence of factors, returning the power of delta
        which got moved to the front.
        """
        factors = list(seq)
        for i in range(len(factors) - 1):
            factors[i], factors[i+1] = self.renorm[factors[i]][factors[i+1]]
            for j in range(i - 1, -1, -1):
                factors[j], factors[j+1] = self.renorm[factors[j]][factors[j+1]]
            for j in range(i+1, len(factors) - 1):
                factors[j], factors[j+1] = self.renorm[factors[j]][factors[j+1]]

        l = 0
        r = len(factors)
        while l < r and factors[l] == self.D:
            l += 1
        while l < r and factors[r - 1] == self.id:
            r -= 1

        return l, tuple(factors[l:r])

    def product_normalised_factors(self, left: Sequence[int], right: Sequence[int]) -> tuple[int, tuple[int, ...]]:
        """Given two sequences of factors in normal form, take their product."""

        # Fast path: if the sequences are compatible, then it's just their concatenation.
        if len(left) == 0 or len(right) == 0 or self.is_normalised[left[-1]][right[0]]:
            return 0, (*left, *right)

        # General case: concatenate the sequences and normalise. Let i be the place where normality (factors[i], factors[i+1]) might
        # be violated: initially this is at len(left) - 1, where the sequences were joined. Change these if need be, then violations
        # may happen at (i-1, i) and (i+1, i+2). Comb backwards through the sequence so that the only next violation could occur at
        # (i+1, i+2), then iterate.
        factors = [*left, *right]
        for i in range(len(left) - 1, len(factors) - 1):
            x, y = self.renorm[factors[i]][factors[i+1]]

            # If no replacement needs to happen, we have finished.
            if x == factors[i]:
                break

            # Otherwise, perform a replacement and comb backwards doing replacements.
            factors[i] = x
            factors[i+1] = y

            for j in range(i-1, -1, -1):
                x, y = self.renorm[factors[j]][factors[j+1]]
                if x == factors[j]:
                    break
                factors[j] = x
                factors[j+1] = y

        # Now this sequence may start with a sequence of δ and end with a sequence of id. Cut these off.
        l = 0
        r = len(factors)
        while l < r and factors[l] == self.D:
            l += 1
        while l < r and factors[r - 1] == self.id:
            r -= 1

        return l, tuple(factors[l:r])

    def normal_forms(self, length: int, following: int | None = None) -> Iterable[tuple[int, ...]]:
        """
        Return an iterator over all normal forms of the given length. If the following argument is
        given, only returns those normal forms which follow that entry.
        """
        assert length >= 0

        word = [-1] * length
        def helper(i: int, following: int):
            if i == length:
                yield tuple(word)
                return
            for x in self.follows[following]:
                word[i] = x
                yield from helper(i+1, x)

        yield from helper(0, following if isinstance(following, int) else self.D)

    def count_normal_forms(self, length: int, following: int | None = None) -> int:
        """
        Count the number of normal forms that the same call to normal_forms() would produce.
        """
        if length == 0:
            return 1

        last = following if isinstance(following, int) else self.D
        counts = {x: 1 for x in self.follows[last]}

        for _ in range(length - 1):
            new_counts: defaultdict[int, int] = defaultdict(int)
            for x, count in counts.items():
                for y in self.follows[x]:
                    new_counts[y] += count

            counts = new_counts

        return sum(counts.values())

    def sample(self, length: int, rand: random.Random) -> tuple[int, ...]:
        """Sample a normal form of the given length."""
        last = self.D
        factors = [-1] * length
        for i in range(length):
            last = factors[i] = rand.choice(self.follows[last])

        return tuple(factors)


@dataclasses.dataclass
class PermTable(NFTable[Permutation]):
    """NFTable specialised to permutations (usual Garside normal form)."""
    @functools.cache
    @staticmethod
    def create(n: int):
        assert n >= 0
        if n >= 7:
            raise ValueError(f"Warning: this would tabulate all elements of S{n} and create tables of size S{n}×S{n}.")

        order = math.factorial(n)
        symgrp = SymmetricGroup(n)
        perms = list(symgrp.elements())
        index={perm: i for i, perm in enumerate(perms)}
        table = PermTable(
            n=n,
            order=math.factorial(n),
            divs=perms,
            index=index,
            id=index[symgrp.id()],
            D=index[symgrp.longest_element()],
            artin_gens=[index[gen] for gen in symgrp.cox_gens()],
            left_comp=[-1]*order,
            right_comp=[-1]*order,
            tau_order=2,
            tau_left=[list(range(order)), [-1]*order],
            tau_right=[list(range(order)), [-1]*order],
            renorm=[[(0, 0)]*order for _ in range(order)],
            is_normalised=[[False]*order for _ in range(order)],
            follows=[[] for _ in range(order)],
            artin_word=[[] for _ in range(order)],
        )

        # Left complement is x ↦ w_0 x⁻¹, right complement is the other way around.
        w0 = symgrp.longest_element()
        for i, perm in enumerate(perms):
            table.left_comp[i] = index[w0 * perm.inv()]
            table.right_comp[i] = index[perm.inv() * w0]

        # Since tau has order 2, tau_left and tau_right are the same. They are both
        # conjugation by the longest element.
        for i, perm in enumerate(perms):
            table.tau_left[1][i] = table.tau_right[1][i] = index[w0 * perm * w0]

        # Renormalisation.
        coxgens = symgrp.cox_gens()
        for x, y in itertools.product(perms, perms):
            w, z = x, y

            # While there is a simple reflection s in L(z) - R(w), replace with (ws, sz).
            while diff := z.left_descent_set() - w.right_descent_set():
                s = diff.pop()
                w = w * coxgens[s]
                z = coxgens[s] * z

            table.renorm[index[x]][index[y]] = (index[w], index[z])
            if (x, y) == (w, z):
                table.is_normalised[index[x]][index[y]] = True
                if index[y] != table.id and index[y] != table.D:
                    table.follows[index[x]] += [index[y]]

        # Artin words
        for i, perm in enumerate(perms):
            table.artin_word[i] = [(s, 1) for s in perm.shortlex()]

        return table


@dataclasses.dataclass
class NParTable(NFTable[NPar]):
    """NFTable specialised to NPars (dual Garside normal form)."""

    bkl_gens: dict[tuple[int, int], int]

    @functools.cache
    @staticmethod
    def create(n: int):
        assert n >= 0
        order = catalan(n)

        if order >= 5000:
            raise ValueError(f"Warning: this would tabulate all elements of NPar_{n} ({order} elements) and create tables of size NPar_{n}×NPar_{n} ({order*order} elements).")

        npars = list(NPar.all_of_size(n))
        index = {npar: i for i, npar in enumerate(npars)}
        table = NParTable(
            n=n,
            order=order,
            divs=npars,
            index=index,
            artin_gens=[index[NPar.from_components(n, [[i, i+1]])] for i in range(n - 1)],
            bkl_gens={(i, j): index[NPar.from_components(n, [[i, j]])] for i, j in itertools.combinations(range(n), 2)},
            id=index[NPar.finest(n)],
            D=index[NPar.coarsest(n)],
            left_comp=[index[npar.left_complement()] for npar in npars],
            right_comp=[index[npar.right_complement()] for npar in npars],
            tau_order=n,
            tau_left=[[] for _ in range(n)],
            tau_right=[[] for _ in range(n)],
            renorm=[[(0, 0)]*order for _ in range(order)],
            is_normalised=[[False]*order for _ in range(order)],
            follows=[[] for _ in range(order)],
            artin_word=[npar.artin_word() for npar in npars],
        )

        # Tau
        table.tau_left[0] = list(range(order))
        table.tau_right[0] = list(range(order))
        table.tau_left[1] = [index[npar.left_complement().left_complement()] for npar in npars]
        table.tau_right[1] = [index[npar.right_complement().right_complement()] for npar in npars]
        for i in range(2, n):
            table.tau_left[i] = [table.tau_left[1][table.tau_left[i-1][x]] for x in range(order)]
            table.tau_right[i] = [table.tau_right[1][table.tau_right[i-1][x]] for x in range(order)]


        # Pairs
        for x, y in itertools.product(npars, npars):
            w, z = x, y

            while compatible_edges := set(w.right_complement().edges()) & set(z.edges()):
                edge = compatible_edges.pop()
                w_perm, edge_perm, z_perm = w.to_permutation(), NPar.from_components(n, [edge]).to_permutation(), z.to_permutation()
                w_perm = w_perm * edge_perm
                z_perm = edge_perm * z_perm
                w, z = NPar.from_permutation(w_perm), NPar.from_permutation(z_perm)

            table.renorm[index[x]][index[y]] = (index[w], index[z])
            if (w, z) == (x, y):
                table.is_normalised[index[x]][index[y]] = True
                if index[y] != table.id and index[y] != table.D:
                    table.follows[index[x]] += [index[y]]

        return table


class NFBase(Generic[T]):
    """
    NFBase is an abstract class that has functionality common to both GNFs and DGNFs.
    """
    n: int
    power: int
    factors: tuple[int, ...]

    # This NF thing is a little cheesy trick so that we can correctly type methods which are supposed to return
    # a GNF or a DGNF from this base class. There is meant to be a less cheesy way to do this in Python 3.11:
    # https://peps.python.org/pep-0673/
    NF = TypeVar("NF", bound="NFBase")

    @abc.abstractmethod
    def __init__(self, n: int, power: int, factors: tuple[int, ...]): ...

    # TODO: Figure out how to correctly type this method.
    @abc.abstractclassmethod
    def _nf_table(cls, n: int) -> NFTable[T]: ...

    def __post_init__(self):
        """Assert that the factors we have been given are in normal form."""
        assert self._nf_table(self.n).is_factors_normalised(self.factors), f"The factors {self.factors} are not normalised."

    @classmethod
    def identity(cls, n: int):
        """Return the identity braid on n strands."""
        return cls(n, 0, ())

    @classmethod
    def artin_gens(cls, n: int):
        """Return a list of the n-1 Artin generators."""

        return [cls(n, 0, (f,)) for f in cls._nf_table(n).artin_gens]

    def canonical_factors(self) -> tuple[T, ...]:
        """
        Return the list of canonical factors as elements in the symmetric group or noncrossing
        partitions, depending on whether this is a GNF or DGNF.
        """
        table = self._nf_table(self.n)
        return tuple(table.divs[f] for f in self.factors)

    def canonical_factor(self, i: int) -> T:
        """Return the ith canonical factor. Works with negative indices, like lists."""
        table = self._nf_table(self.n)
        return table.divs[self.factors[i]]

    def canonical_decomposition(self) -> tuple[int, tuple[T, ...]]:
        """
        Return the power, along with the list of canonical factors. To just get the canonical
        factors, use .canonical_factors() instead.
        """
        return (self.power, self.canonical_factors())

    def canonical_length(self) -> int:
        """Return the length of the normal form."""
        return len(self.factors)

    def inf(self) -> int:
        """Return the infimum of the braid, i.e. the power of the Garside element at the start."""
        return self.power

    def sup(self) -> int:
        """Return the supremum of the braid."""
        return self.power + len(self.factors)

    def __mul__(self: NF, other: NF) -> NF:
        """Group multiplication, within the same kind of normal form."""
        if isinstance(other, self.__class__):
            if self.n != other.n:
                raise ValueError("Cannot multiply braids from different groups")

            # When multiplying (D^d, x1, ..., xn) by (D^e, y1, ..., ym), the D^e
            # moves through to the right, applying the automorphism τ_R.
            table = self._nf_table(self.n)

            # Fast path
            can_concatenate = (other.power == 0) and (
                len(self.factors) == 0
                or len(other.factors) == 0
                or table.is_normalised[self.factors[-1]][other.factors[0]]
            )
            if can_concatenate:
                return self.__class__(self.n, self.power, self.factors + other.factors)

            factors_conjugated = [table.tau_right[other.power % table.tau_order][x] for x in self.factors]
            d, factors = table.product_normalised_factors(factors_conjugated, other.factors)
            return self.__class__(self.n, self.power + other.power + d, factors)

        return NotImplemented

    def inv(self: NF) -> NF:
        """Group inversion."""
        table = self._nf_table(self.n)
        factors = [table.left_comp[x] for x in self.factors[::-1]]
        delta_pow = -self.power
        for i in range(len(factors) - 1, -1, -1):
            factors[i] = table.tau_right[delta_pow % table.tau_order][factors[i]]
            delta_pow -= 1

        d, normalised = table.normalise_factors(factors)
        return self.__class__(self.n, delta_pow + d, tuple(normalised))

    def __pow__(self: NF, exp: int) -> NF:
        acc = self.identity(self.n)
        if exp == 0:
            return acc

        pow2 = self if exp > 0 else self.inv()
        exp = abs(exp)
        while exp:
            if exp % 2 == 1:
                acc = acc * pow2

            exp //= 2
            pow2 = pow2 * pow2

        return acc

    def artin_word(self) -> list[tuple[int, int]]:
        """Return a word in the Artin generators for this braid."""
        table = self._nf_table(self.n)
        Delta = table.artin_word[table.D]
        if self.power < 0:
            Delta = [(s, -e) for s, e in Delta[::-1]]

        return Delta*abs(self.power) + [letter for factor in self.factors for letter in table.artin_word[factor]]

    def magma_artin_word(self) -> list[int]:
        """
        Return an integer sequence which can be coerced in Magma into a braid. The integer sequence is the Artin word,
        where the generators are indexed {1, ..., n}, and negative numbers correspond to inverse generators.
        """

        # We assume that the only e's we see are ±1.
        artin_word = self.artin_word()
        assert all(abs(e) == 1 for _, e in artin_word)
        return [(i+1)*e for i, e in artin_word]

    def substring(self: NF, i: int, j: int) -> NF:
        """Take a substring of the canonical factors of the braid. δ^d remains the same."""
        assert 0 <= i <= j <= len(self.factors)
        return self.__class__(self.n, self.power, self.factors[i:j])

    def nf_descendants(self: NF) -> Iterable[NF]:
        """Return all immediate descendants of this braid in the normal form tree."""
        table = self._nf_table(self.n)

        return (
            self.__class__(self.n, self.power, (*self.factors, *suffix))
            for suffix in table.normal_forms(length=1, following=self.factors[-1] if self.factors else None)
        )

    def nf_suffixes(self: NF, length: int) -> Iterable[NF]:
        """Return all x such that self * x has no cancellation, and x has the given length."""
        table = self._nf_table(self.n)
        return (
            self.__class__(self.n, self.power, suffix)
            for suffix in table.normal_forms(length=length, following=self.factors[-1] if self.factors else None)
        )

    @classmethod
    def count_all_of_length(cls, n: int, length: int) -> int:
        """Count all normal forms of the given length."""
        return cls._nf_table(n).count_normal_forms(length)

    @classmethod
    def all_of_length(cls: type[NF], n: int, length: int, power: int = 0) -> Iterable[NF]:
        """
        An iterator over all normal forms of the given length.
        If power is specified, puts the Garside element to that power in front of each.
        """
        return (cls(n, power, factors) for factors in cls._nf_table(n).normal_forms(length))

    @classmethod
    def sample(cls: type[NF], n: int, length: int, rand: random.Random | None = None) -> NF:
        """Sample a braid of a given canonical length randomly."""
        rand = rand if rand is not None else random.Random()
        table = cls._nf_table(n)
        return cls(n, 0, table.sample(length, rand))



@dataclasses.dataclass(frozen=True, order=True)
class GNF(NFBase[Permutation]):
    """
    A GNF represents an element of the braid group in left Garside normal form Δ^d β(1) ... β(k).
    A GNF should not usually be created directly, one should first create a braid group and then
    use methods on that object to create them.

    The internal representation of a GNF is as a triple (n, power, factors), where n is the number
    of strands in the braid group, power is the power of Δ, and factors is a list of integers
    indexing permutations. The factors must be in normal form (an error will be thrown otherwise).
    """
    n: int                   # How many strands is this braid group.
    power: int               # The power Δ^k at the start of the normal form.
    factors: tuple[int, ...] # The factors in the normal form, as indices using the permutation table.

    # I don't know how to correctly type things here so let's ignore the typechecker.
    @classmethod
    def _nf_table(cls, n: int) -> NFTable[Permutation]: # type: ignore
        return PermTable.create(n)

    def __repr__(self):
        return f'GNF{self.canonical_decomposition()!r}'

    def _repr_latex_(self):
        """This method makes braids display nicely in IPython notebooks."""
        power, perms = self.canonical_decomposition()
        return fr'$(\Delta^{{{power}}};' + ', '.join(perm._repr_latex_().strip('$') for perm in perms) + ')$'

    @staticmethod
    def from_perm(perm: Permutation) -> GNF:
        """The positive lift of a permutation."""
        return GNF.from_perm_seq(len(perm.word), [perm])

    @staticmethod
    def from_perm_seq(n: int, perms: Sequence[Permutation]) -> GNF:
        assert all(n == len(perm.word) for perm in perms)
        table = PermTable.create(n)
        power, factors = table.normalise_factors([table.index[perm] for perm in perms])
        return GNF(n, power, tuple(factors))

    @staticmethod
    def from_artin_word(n: int, word: Iterable[tuple[int, int]]):
        table = PermTable.create(n)
        factors = []
        delta_pows = []
        for i, e in word:
            if e >= 0:
                factors += [table.artin_gens[i]] * e
                delta_pows += [0] * e
            else:
                factors += [table.left_comp[table.artin_gens[i]]] * abs(e)
                delta_pows += [-1] * abs(e)

        delta_pow = 0
        for i in range(len(factors)-1, -1, -1):
            factors[i] = table.tau_right[delta_pow % table.tau_order][factors[i]]
            delta_pow += delta_pows[i]

        power, factors_norm = table.normalise_factors(factors)
        return GNF(n, delta_pow + power, tuple(factors_norm))

     # The code from here down is a bit gross, but makes conversion fast enough to do it whenever
    # we're saving to the database.

    @functools.cache
    @staticmethod
    def _factors_to_DGNF(n: int):
        table = PermTable.create(n)
        return [DGNF.from_artin_word(n, GNF.from_perm(perm).artin_word()) for perm in table.divs]

    def to_DGNF(self) -> DGNF:
        table = PermTable.create(self.n)
        dgnfs = GNF._factors_to_DGNF(self.n)
        result = functools.reduce(operator.mul, [dgnfs[f] for f in self.factors], DGNF.identity(self.n))
        if self.power != 0:
            delta_pow = dgnfs[table.D] ** self.power
            result = delta_pow * result
        return result

    # Keep this method since some older notebooks depend on it.
    def garside_length(self) -> int:
        """Returns the Garside length of a braid. Please use .canonical_length() instead."""
        return self.canonical_length()


@dataclasses.dataclass(frozen=True, order=True)
class DGNF(NFBase[NPar]):
    """
    A DGNF represents an element of the braid group in left Garside normal form δ^d β(1) ... β(k)
    for the dual Garside structure.

    A DGNF should not usually be created directly, one should first create a braid group and then
    use methods on that object to create them.

    The internal representation of a DGNF is as a triple (n, power, factors), where n is the number
    of strands in the braid group, power is the power of δ, and factors is a list of integers
    indexing permutations. The factors must be in normal form (an error will be thrown otherwise).
    """
    n: int                   # How many strands is this braid group.
    power: int               # The power Δ^k at the start of the normal form.
    factors: tuple[int, ...] # The factors in the normal form, as indices using the permutation table.

    # I don't know how to correctly type things here so let's ignore the typechecker.
    @classmethod
    def _nf_table(cls, n: int) -> NParTable: # type: ignore
        return NParTable.create(n)

    def __repr__(self):
        return f'DGNF{self.canonical_decomposition()!r}'

    def _repr_latex_(self):
        """This method makes braids display nicely in IPython notebooks."""
        power, npars = self.canonical_decomposition()
        return fr'$(\delta^{{{power}}};' + ', '.join(npar._repr_latex_().strip('$') for npar in npars) + ')$'

    @staticmethod
    def from_npar(npar: NPar):
        return DGNF.from_npar_seq(npar.n, (npar,))

    @functools.cache
    @staticmethod
    def bkl_gens(n: int) -> dict[tuple[int, int], DGNF]:
        """
        Return the BKL generators. Note that (i, j) is the same generator as (j, i),
        and (i, i) is the identity.
        """
        id = DGNF.identity(n)
        table = DGNF._nf_table(n)
        return {
            (i, j): id if i == j else DGNF(n, 0, (table.bkl_gens[(min(i, j), max(i, j))],))
            for i in range(n) for j in range(n)
        }

    @staticmethod
    def from_npar_seq(n: int, npars: Sequence[NPar]):
        assert all(n == npar.n for npar in npars)
        table = NParTable.create(n)
        power, factors = table.normalise_factors([table.index[npar] for npar in npars])
        return DGNF(n, power, tuple(factors))

    @staticmethod
    def from_artin_word(n: int, word: Iterable[tuple[int, int]]):
        table = NParTable.create(n)

        # A generator to a positive power is easy. When we have a generator to a negative power,
        # we have that β(p)^-1 = δ^-1 β(*p) where *p is the left complement. The δ^-1 will have to
        # be applied backwards through the whole product.
        #
        # To this end, if we have something like [p, q^-1, r^-1, w] we make two parallel arrays:
        #  factors:   [p,  q,  r, w]
        #  delta_pow: [0, -1, -1, 0]
        # Together these mean [p, δ^-1, *q, δ^-1, *r, w]. The δ powers can then be commuted back to
        # the front by means of the right conjugation automorphism.
        factors = []
        delta_pows = []
        for i, e in word:
            if e >= 0:
                factors += [table.artin_gens[i]] * e
                delta_pows += [0] * e
            else:
                factors += [table.left_comp[table.artin_gens[i]]] * abs(e)
                delta_pows += [-1] * abs(e)

        delta_pow = 0
        for i in range(len(factors)-1, -1, -1):
            factors[i] = table.tau_right[delta_pow % table.tau_order][factors[i]]
            delta_pow += delta_pows[i]

        power, factors = table.normalise_factors(factors)
        return DGNF(n, delta_pow + power, tuple(factors))

    # The code from here down is a bit gross, but makes conversion fast enough to do it whenever
    # we're saving to the database.

    @functools.cache
    @staticmethod
    def _factors_to_GNF(n: int):
        table = NParTable.create(n)
        return [GNF.from_artin_word(n, DGNF.from_npar(npar).artin_word()) for npar in table.divs]

    def to_GNF(self) -> GNF:
        table = NParTable.create(self.n)
        gnfs = DGNF._factors_to_GNF(self.n)
        result = functools.reduce(operator.mul, [gnfs[f] for f in self.factors], GNF.identity(self.n))
        if self.power != 0:
            delta_pow = gnfs[table.D] ** self.power
            result = delta_pow * result
        return result


@dataclasses.dataclass(frozen=True)
class BraidGroup:
    """
    The braid group on n strands. Elements of the braid group will be of type GNF or DGNF, for
    whether they are represented in Garside normal form or dual Garside normal form, respectively.
    """
    n: int

    def __post_init__(self):
        assert self.n >= 1

    @functools.cache
    def id(self):
        return GNF(self.n, 0, ())

    @functools.cache
    def gens(self) -> tuple[GNF, ...]:
        """Return the Artin generators of the braid group, as GNFs."""
        table = PermTable.create(self.n)
        return tuple(GNF(self.n, 0, (table.artin_gens[i],)) for i in range(self.n - 1))

    @functools.cache
    def bkl_gens(self) -> dict[tuple[int, int], DGNF]:
        """
        Return a dictionary of the dual generators. Both (i < j) and (i > j) appear,
        although they are the same element.
        """
        table = NParTable.create(self.n)
        return {
            **{gen: DGNF(self.n, 0, (idx,)) for gen, idx in table.bkl_gens.items()},
            **{gen[::-1]: DGNF(self.n, 0, (idx,)) for gen, idx in table.bkl_gens.items()},
            **{(i, i): DGNF.identity(self.n) for i in range(self.n)},
        }

    @functools.cache
    def invs(self):
        """Return the inverses of the Artin geneators."""
        return [x.inv() for x in self.gens()]

    def positive_lift(self, perm: Permutation) -> GNF:
        """The positive lift of a permutation into the braid group."""
        if len(perm.word) != self.n:
            raise ValueError(f"Cannot lift permutation on {len(perm.word)} elements into braid group on {self.n} strands.")

        if permutations.is_identity(perm.word):
            return GNF(self.n, 0, ())
        if permutations.is_longest_element(perm.word):
            return GNF(self.n, 1, ())

        table = PermTable.create(self.n)
        return GNF(self.n, 0, (table.index[perm],))

    def from_npar(self, npar: NPar) -> DGNF:
        """The lift of a noncrossing partition into the braid group."""
        if npar.n != self.n:
            raise ValueError(f"Cannot lift the noncrossing parititon on {npar.n} elements into braid group on {self.n} strands.")

        table = NParTable.create(self.n)
        index = table.index[npar]
        if index == table.id:
            return DGNF(self.n, 0, ())
        elif index == table.D:
            return DGNF(self.n, 1, ())
        else:
            return DGNF(self.n, 0, (index,))
    
    def from_gnf_pair(self, pair: tuple[int, tuple[int, ...]]) -> GNF:
        """Convert from the canonical decomposition (inf, word of permutations) to a braid."""
        inf, perm_words = pair
        return GNF(self.n, inf, ()) * functools.reduce(operator.mul, map(self.positive_lift, map(Permutation, perm_words)), self.id())


    def all_of_garside_length(self, length: int, power: int = 0) -> Iterable[GNF]:
        """Use GNF.all_of_length() or DGNF.all_of_length() instead."""
        return GNF.all_of_length(n=self.n, length=length)

    def count_all_of_garside_length(self, length: int):
        """Use GNF.count_all_of_length() or DGNF.count_all_of_length() instead."""
        return GNF.count_all_of_length(self.n, length)

    def sample_braid_perm(self, length: int, rand: random.Random | None = None):
        """Use GNF.sample() or DGNF.sample() instead."""
        return GNF.sample(self.n, length, rand=rand)


def adapt_braid(braid: NFBase) -> str:
    """
    Convert to a representation like b'GNF(4, 1, (2, 3, 4))' for storage in SQLite.
    """
    return f'{braid.__class__.__name__}{dataclasses.astuple(braid)}'


def convert_braid(data: bytes) -> NFBase:
    """
    Read a braid out of SQLite into an object.
    """
    if data.startswith(b'GNF'):
        return GNF(*ast.literal_eval(data[3:].decode('utf-8')))
    if data.startswith(b'DGNF'):
        return DGNF(*ast.literal_eval(data[4:].decode('utf-8')))
    raise ValueError(f"Could not convert {data!r} to a braid.")


def register_sqlite3_adapters():
    import sqlite3
    sqlite3.register_adapter(GNF, adapt_braid)
    sqlite3.register_adapter(DGNF, adapt_braid)
    sqlite3.register_converter("BRAID", convert_braid)
    sqlite3.register_converter("BRAID", convert_braid)
