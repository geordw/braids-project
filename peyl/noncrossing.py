"""
npar - code for working with noncrossing partitions.

The noncrossing partitions of {0, ..., n-1} are in bijection with:
- Permutations on {0, ..., n-1} which are disjoint parallel descending cycles, and
- Temperley-Lieb diagrams of type TL(0, 2n), i.e. an involution of {0, ..., 2n - 1} with no fixed
  points, which additionally satisfies a crossingless condition as in `tl.is_crossingless_matching()`.
"""
from __future__ import annotations

import dataclasses
import itertools
import math
from typing import Iterable, TypeVar

from . import permutations, tl
from .permutations import Permutation

T = TypeVar('T')

def catalan(n: int) -> int:
    """
    Return the nth Catalan number.

    >>> [catalan(n) for n in range(10)]
    [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862]
    """
    return math.comb(2*n, n) // (n + 1)

def are_pairwise_disjoint(components: Iterable[Iterable[T]]):
    """Checks if the components are disjoint sets."""
    seen = set()
    for component in components:
        for elem in component:
            if elem in seen:
                return False

            seen.add(elem)

    return True

def bkl_gen_to_artin_word(s: int, t: int) -> Iterable[tuple[int, int]]:
    """
    Return a sequence of (i, e) for e = ±1 such that a_{st} is the product of the σ(i)^e.

    >>> bkl_gen_to_artin_word(3, 3)
    []
    >>> bkl_gen_to_artin_word(3, 4)
    [(3, 1)]
    >>> bkl_gen_to_artin_word(5, 2)
    [(4, 1), (3, 1), (2, 1), (3, -1), (4, -1)]
    """
    # For s < t we have a_{st} = σ(t-1) . σ(t-2) ... σ(s+1) . σ(s) . σ(s+1)^-1 ... σ(t-1)^-1
    if s == t:
        return []
    if s > t:
        s, t = t, s

    return [
        *[(i, 1) for i in range(t-1, s, -1)],
        (s, 1),
        *[(i, -1) for i in range(s+1, t)],
    ]

def npar_component_to_artin_word(component: Iterable[int]) -> Iterable[tuple[int, int]]:
    """
    Given a component, i.e. a set of integers, return a word in the Artin generators representing the
    same braid.

    >>> npar_component_to_artin_word({1})
    []
    >>> npar_component_to_artin_word({1, 2})
    [(1, 1)]
    >>> npar_component_to_artin_word({0, 1, 2, 3})
    [(2, 1), (1, 1), (0, 1)]
    """
    # A BKL word for a component {a, ..., d} can be made by deleting any edge in the cycle, and reading the rest
    # of the edges off in anticlockwise order. If we delete the longest edge which goes between the lowest and
    # highest elements, then the resulting Artin word should be shortest I think.
    comp = sorted(component)
    assert len(comp) == len(set(comp)), f"Noncrossing partition component {component} should not have duplicates."

    return [
        letter
        for i in range(len(comp) - 2, -1, -1)
        for letter in bkl_gen_to_artin_word(comp[i], comp[i+1])
    ]


@dataclasses.dataclass(frozen=True)
class NPar:
    """
    An NPar is a noncrossing partition of n points {0, ..., n - 1}. Internally it is represented as
    an involution with no fixed points on the set {0, ..., 2n - 1}, which additionally satisfies a
    bracket-matching condition.
    """
    n: int
    inv: tuple[int, ...]

    def __post__init__(self):
        """Check that inv is really an involution of the right kind."""
        assert len(self.inv) == 2*self.n
        assert tl.is_crossingless_matching(self.inv)

    def __repr__(self):
        return f'NPar({self.n}, [' + ', '.join(repr(set(comp)) for comp in self.components() if len(comp) > 1) + '])'

    @staticmethod
    def finest(n: int):
        """
        Returns the finest noncrossing partition of {0, ..., n-1}, i.e. {0} ∪ ... ∪ {n-1}.

        >>> NPar.finest(4).inv
        (1, 0, 3, 2, 5, 4, 7, 6)
        """
        return NPar(n, tuple(i + 1 if i % 2 == 0 else i - 1 for i in range(2*n)))

    @staticmethod
    def coarsest(n: int):
        """Returns the coarsest noncrossing partition of {0, ..., n-1}, i.e {0, ..., n-1}."""
        return NPar(n, tuple((i+1)%(2*n) if i%2 == 1 else (i-1)%(2*n) for i in range(2*n)))

    @staticmethod
    def all_of_size(n: int) -> Iterable[NPar]:
        """Return all noncrossing partitions of {0, ..., n-1}. There are catalan(n) many of these."""
        assert n >= 0
        return (NPar(n, seq) for seq in tl.generate_sequences(2*n))

    @staticmethod
    def from_components(n: int, components: Iterable[Iterable[int]]) -> NPar:
        """
        Creates a noncrossing partition from a list of components, i.e. an explicit specification
        of the partition. Singleton components {i} may be omitted.

        >>> NPar.finest(4) == NPar.from_components(4, [[0], [1], [2], [3]])
        True
        >>> NPar.coarsest(4) == NPar.from_components(4, [[0, 1, 3, 2]])
        True
        """
        if not are_pairwise_disjoint(components):
            raise ValueError(f"The components {components} do not form a partition.")

        if not all(0 <= x < n for comp in components for x in comp):
            raise ValueError(f"Not all component elements of {components} are in the range [0, {n})")

        inv = list(NPar.finest(n).inv)
        for comp in [sorted(comp) for comp in components]:
            for i in range(len(comp)):
                a, b = comp[i], comp[(i+1) % len(comp)]
                inv[2*a + 1] = 2*b
                inv[2*b] = 2*a + 1

        if not tl.is_crossingless_matching(inv):
            raise ValueError(f"The components {components} define a partition, but it is a crossing partition: {inv}.")

        return NPar(n, tuple(inv))

    @staticmethod
    def from_permutation(perm: Permutation):
        """
        Create a noncrossing partition from a permutation which is a product of disjoint parallel
        descending cycles, or throw an error if the permutation is not of the right form.

        >>> NPar.from_permutation(Permutation([4, 0, 1, 2, 3])) == NPar.coarsest(5)
        True
        """
        if not permutations.is_parallel_descending_cycles(perm.word):
            raise ValueError(f"The permutation {perm} is not made of disjoint parallel descending cycles.")

        return NPar.from_components(len(perm.word), perm.disjoint_cycles())

    def to_permutation(self):
        """
        Return the permutation made of disjoint parallel descending cycles associated to this
        noncrossing partition.

        >>> NPar.coarsest(5).to_permutation()
        Permutation(word=(4, 0, 1, 2, 3))
        """
        word = [0] * self.n
        for i in range(self.n):
            word[i] = self.inv[2*i] // 2

        return Permutation(tuple(word))

    def components(self) -> list[tuple[int, ...]]:
        """
        Return the components, i.e. the disjoint sets making up the partition.

        >>> NPar.from_components(9, [[5, 0, 4], [7, 6], [3, 2]]).components()
        [(0, 4, 5), (1,), (2, 3), (6, 7), (8,)]
        """
        seen = [False] * self.n
        components: list[tuple[int, ...]] = []
        for i in range(self.n):
            if seen[i]:
                continue

            j = i
            group = []
            while True:
                group.append(j)
                seen[j] = True
                j = self.inv[2*j+1] // 2
                if i == j:
                    break

            components += [tuple(group)]

        return components

    def edges(self) -> Iterable[tuple[int, int]]:
        """
        The edges in a crossingless matching are all pairs taken from within the components.

        >>> NPar.from_components(10, [(0, 1), (2,), (3, 5, 6)]).edges()
        [(0, 1), (3, 5), (3, 6), (5, 6)]
        """
        return [pair for comp in self.components() for pair in itertools.combinations(comp, 2)]

    def left_complement(self):
        """
        The left complement of a noncrossing partition p is *p, where β(*p) β(p) = δ in the braid group.
        As a TL diagram, it corresponds to pulling the string in the bottom left all the way over to
        the bottom right. Performing the left complement twice is the left complement by δ, so
        β(**p) = δ β(p) δ^-1

        >>> NPar.finest(5).left_complement() == NPar.coarsest(5)
        True
        """
        return NPar(self.n, tuple((self.inv[(i+1)%(2*self.n)] - 1) % (2*self.n) for i in range(2*self.n)))

    def right_complement(self):
        """
        The right complement is inverse to the left complement, and satisfies similar properties.

        >>> npar = NPar.from_components(9, [[5, 0, 4], [7, 6], [3, 2]])
        >>> npar.left_complement().right_complement() == npar
        True
        """
        return NPar(self.n, tuple((self.inv[(i-1)%(2*self.n)] + 1) % (2*self.n) for i in range(2*self.n)))

    def _repr_latex_(self):
        return '$p(' + ' ∪ '.join(r'\{' + ', '.join(map(str, comp)) + r'\}' for comp in self.components()) + ')$'

    def artin_word(self) -> list[tuple[int, int]]:
        """
        Return a sequence of (i, e) for e = ±1 such that the braid corresponding to this NPar is the product of the σ(i)^e.

        >>> NPar.finest(4).artin_word()
        []
        >>> NPar.coarsest(4).artin_word()
        [(2, 1), (1, 1), (0, 1)]
        """
        return [letter for comp in self.components() for letter in npar_component_to_artin_word(comp)]
