"""
Functions for working with permutations of the integers [0, n).

The formats for working with permutations are:
- Word notation, where the permutation x is represented by the array [x(0), ..., x(n-1)]. Most functions
  expect to be given a permutation in word form (where the exact type of the object may be any sequence),
  and will return permutations in word form as a tuple.
- CoxWord notation, where a permutation is specified by an expression for it in the Coxeter generators.
  The ShortLex form (the lexicographically least reduced expression) is a normal form.


The standard format for a permutation is "word" notation, where the permutation x is represented
by the array [x(0), ..., x(n-1)].
"""
from __future__ import annotations

import collections
import dataclasses
import functools
import itertools
import operator
from typing import Iterable, Sequence


def is_permutation(word: Sequence[int]):
    """
    Check that word is a permutation of the integers [0, n) where n = len(word).

    >>> words = [(), (0, 1), (0, 2), (0, 0, 2), (2, 1, 0)]
    >>> [is_permutation(word) for word in words]
    [True, True, False, False, True]
    """

    if len(word) == 0:
        return True

    # To try to avoid allocating when checking short permutations, first check that all entries like in [0, n), and then
    # bitwise-or them into a bitmask of their union. This should be equal to 2^n - 1 (all 1's), and any duplicate will
    # cause a zero to appear somewhere.

    if min(word) != 0 or max(word) != len(word) - 1:
        return False

    mask = functools.reduce(operator.or_, (1 << x for x in word), 0)
    return mask == 2**len(word) - 1


def identity(n: int) -> tuple[int, ...]:
    """
    The identity permutation of S_n.

    >>> [identity(n) for n in [0, 1, 2, 3]]
    [(), (0,), (0, 1), (0, 1, 2)]
    """
    return tuple(range(n))


def is_identity(perm: Sequence[int]):
    return all(perm[i] == i for i in range(len(perm)))


def transposition(n: int, i: int, j: int) -> tuple[int, ...]:
    """
    The transposition (ij) in S_n.
    """
    assert n >= 0
    assert 0 <= i < n and 0 <= j < n

    perm = list(range(n))
    perm[i], perm[j] = perm[j], perm[i]
    return tuple(perm)


def longest_element(n: int) -> tuple[int, ...]:
    """
    The longest element of S_n, i.e. the reversing permutation.

    >>> longest_element(3)
    (2, 1, 0)
    """
    return tuple(n-i-1 for i in range(n))


def is_longest_element(perm: Sequence[int]):
    return all(perm[i] == len(perm) - i - 1 for i in range(len(perm)))


def disjoint_cycles(perm: Sequence[int]) -> list[tuple[int, ...]]:
    """
    Return a list of disjoint cycles which make up the permutation. Cycles are ordered so that the
    cycles containing the lowest elements come first, and the order within a circle is then
    traversal order starting from the lowest element.

    >>> disjoint_cycles([2, 3, 1, 0])
    [(0, 2, 1, 3)]
    >>> disjoint_cycles([2, 1, 0, 3, 4, 6, 5])
    [(0, 2), (1,), (3,), (4,), (5, 6)]
    """
    cycles = []
    visited = [False] * len(perm)
    for i in range(len(perm)):
        if visited[i]:
            continue

        cycle = []
        pos = i
        while True:
            cycle.append(pos)
            visited[pos] = True
            pos = perm[pos]
            if pos == i:
                break

        cycles.append(tuple(cycle))

    return cycles


def cycle_type(perm: Sequence[int]) -> tuple[int, ...]:
    """
    Return the cycle type of a permutation, the lengths of the disjoint cycles in decreasing order.

    >>> cycle_type(())
    ()
    >>> cycle_type((0, 1, 2))
    (1, 1, 1)
    >>> cycle_type((1, 0, 2))
    (2, 1)
    >>> cycle_type((2, 0, 1))
    (3,)
    """
    lengths = []
    visited = [False] * len(perm)

    for i in range(len(perm)):
        if visited[i]:
            continue

        length = 0
        pos = i
        while True:
            length += 1
            visited[pos] = True
            pos = perm[pos]
            if pos == i:
                break

        lengths += [length]

    return tuple(sorted(lengths, reverse=True))


def parity(perm: Sequence[int]) -> int:
    """
    Calculate the parity of the permutation, i.e. its length mod 2.
    >>> parity((0, 1, 2))
    0
    >>> parity((1, 0, 2))
    1
    >>> parity((2, 0, 1))
    0
    >>> parity(())
    0
    """

    # Odd-length cycles have even parity, and even-length cycles have odd parity.
    # The obvious answer is to add up all of the (cycle length + 1) over cycles.
    # A less obvious answer is the number of cycles plus n.
    return (len(cycle_type(perm)) + len(perm)) % 2


def length(perm: Sequence[int]) -> int:
    """
    Calculate the length of a permutation, i.e. the number of inversions. Currently this is the O(n^2) straightforward
    method, which is fine for small permutations. A better O(n log n) method would use merge sort.

    >>> length(())
    0
    >>> length((1, 2, 3))
    0
    >>> length((3, 2, 1))
    3
    """

    return sum(1 for i in range(len(perm)) for j in range(i+1, len(perm)) if perm[i] > perm[j])


def inverse(perm: Sequence[int]) -> tuple[int, ...]:
    """
    The inverse of a permutation.

    >>> inverse((2, 0, 1))
    (1, 2, 0)
    >>> inverse(())
    ()
    """
    inv = [0] * len(perm)
    for i, pi in enumerate(perm):
        inv[pi] = i

    return tuple(inv)



def compose(x: Sequence[int], y: Sequence[int]) -> tuple[int, ...]:
    """
    Compose two permutations (x, y) -> xy. This composition is right-to-left, i.e. the result applies y, then x.
    """
    if len(x) != len(y):
        raise ValueError(f"Cannot compose permutations of different lengths {len(x)} and {len(y)}")

    return tuple(x[j] for j in y)


def right_descent_set(perm: Sequence[int]) -> set[int]:
    """
    Return the right descent set, i.e. those i in the range [0, n-1) such that the transposition (i, i+1) multiplied
    on the right lowers the length of the permutation.

    Multiplication on the right by a transposition (i, j) corresponds to swapping perm[i] and perm[j], so we can simply
    check whether we are making a previously out-of-order pair into an in-order pair.

    >>> right_descent_set((0, 1, 2))
    set()
    >>> right_descent_set((2, 1, 0))
    {0, 1}
    """
    return {i for i in range(len(perm) - 1) if perm[i] > perm[i+1]}


def is_right_descent(perm: Sequence[int], i: int) -> bool:
    assert 0 <= i < len(perm) - 1
    return perm[i] > perm[i + 1]


def left_descent_set(perm: Sequence[int]) -> set[int]:
    # I'm pretty sure the most efficient thing to do here is to calculate the inverse. Otherwise for each i, we need to
    # go searching through the permutation for i and i + 1, which gives O(n^2) running time.
    return right_descent_set(inverse(perm))


def is_left_descent(perm: Sequence[int], i: int) -> bool:
    assert 0 <= i < len(perm) - 1
    return perm.index(i) > perm.index(i+1)


def shortlex(perm: Sequence[int]) -> tuple[int, ...]:
    """
    Return the short-lex normal form of the permutation in the Coxeter generators.

    >>> shortlex(identity(4))
    ()
    >>> shortlex(longest_element(4))
    (0, 1, 0, 2, 1, 0)
    """

    # We would normally take each smallest left descent, giving word^-1 . perm = id where word is lex smallest.
    # Since right descents are easier to access in the word representation, we'll instead take each smallest
    # right descent of the inverse, giving the lex smallest word such that perm^-1 . word = id.
    #
    # We go from left to right of the inverse permutation, looking for the smallest i such that w[i] > w[i+1],
    # make a note of i, and swap them (applying the simple reflection on the right). This may cause w[i-1] > w[i]
    # and so we need to backtrack a little before proceeding again.
    #
    # This is basically bubble-sorting the inverse permutation, and recording the swaps we do.
    inv_perm = list(inverse(perm))
    word = []
    i = 0
    while i < len(inv_perm) - 1:
        if inv_perm[i] > inv_perm[i+1]:
            word.append(i)
            inv_perm[i], inv_perm[i+1] = inv_perm[i+1], inv_perm[i]
            i = max(i - 1, 0)
        else:
            i += 1

    # Ensure that we ended up sorting the array.
    assert all(i == j for i, j in enumerate(inv_perm))

    return tuple(word)


def from_cox(n: int, seq: Sequence[int]) -> tuple[int, ...]:
    """Recover the word of a permutation from a sequence in the Coxeter generators."""
    assert all(0 <= s < n - 1 for s in seq)
    word = list(range(n))
    for s in seq:
        word[s], word[s+1] = word[s+1], word[s]

    return tuple(word)


def is_descending_cycles(perm: Sequence[int]) -> bool:
    """
    Check if a permutation is made of descending cycles. This is relevant for noncrossing partitions.

    >>> is_descending_cycles([0, 1, 2, 3])     # (0)(1)(2)(3)
    True
    >>> is_descending_cycles([1, 0, 3, 2])     # (1, 0)(3, 2)
    True
    >>> is_descending_cycles([3, 0, 1, 2])     # (3, 2, 1, 0)
    True
    >>> is_descending_cycles([1, 2, 3, 0])     # (0, 1, 2, 3)
    False
    """
    return all(a >= b for cycle in disjoint_cycles(perm) for a, b in zip(cycle[1:], cycle[2:]))


def is_parallel_descending_cycles(perm: Sequence[int]) -> bool:
    """
    Check if a permutation is made of parallel descending cycles. In the symmetric group on n
    letters, there are Catalan(n) many of these, and they are in bijection with noncrossing
    partitions.

    >>> is_parallel_descending_cycles([2, 3, 0, 1])
    False
    """

    # The idea is that if a permutation is made of descending (or ascending) cycles, we can WLOG
    # that the cycles are ascending, and look something like (0, 4, 5, 9)(1, 2, 3)(6, 8)(7). We
    # are testing that we can "lay down" each edge without crossing a node which has already been
    # marked, which would indicate a non-parallellness.
    if not is_descending_cycles(perm):
        return False

    marked = [False] * len(perm)
    for cycle in disjoint_cycles(perm):
        cycle = tuple(sorted(cycle))
        for i in range(len(cycle) - 1):
            if any(marked[j] for j in range(cycle[i] + 1, cycle[i + 1])):
                return False

            marked[cycle[i]] = marked[cycle[i+1]] = True

    return True


@dataclasses.dataclass(frozen=True)
class SymmetricGroup:
    n: int

    def __post__init__(self):
        assert self.n >= 0

    @functools.cache
    def id(self):
        return Permutation(identity(self.n))

    @functools.cache
    def longest_element(self):
        return Permutation(longest_element(self.n))

    @functools.cache
    def cox_gens(self):
        return tuple(Permutation(transposition(self.n, i, i+1)) for i in range(self.n - 1))

    def reflections(self):
        """Return a list of all reflections, i.e. transpositions, in the symmetric group."""
        return [Permutation(transposition(self.n, i, j)) for i in range(self.n - 1) for j in range(i+1, self.n)]

    def from_cox(self, seq: Sequence[int]):
        """Create a permutation from a word in the Coxeter generators."""
        return Permutation(from_cox(self.n, seq))

    def from_cycle(self, cycle: Sequence[int]):
        """Create a permutation from a cycle."""
        assert all(0 <= x < self.n for x in cycle)
        assert max(collections.Counter(cycle).values()) <= 1
        word = list(range(self.n))
        for i in range(len(cycle)):
            word[cycle[i]] = cycle[(i+1)%self.n]

        return Permutation(tuple(word))


    def from_cycles(self, cycles: Sequence[Sequence[int]]):
        """Create a permutation from a list of cycles (not necessarily disjoint)."""
        return functools.reduce(operator.mul, map(self.from_cycle, cycles), self.id)


    def elements(self) -> Iterable[Permutation]:
        """Iterate over all the symmetric group elements."""
        return (Permutation(perm) for perm in itertools.permutations(range(self.n)))


@dataclasses.dataclass(frozen=True)
class Permutation:
    word: tuple[int, ...]

    def __post__init__(self):
        assert is_permutation(self.word)

    def group(self):
        return SymmetricGroup(len(self.word))

    def length(self):
        return length(self.word)

    def disjoint_cycles(self):
        return disjoint_cycles(self.word)

    def cycle_type(self):
        return cycle_type(self.word)

    def shortlex(self):
        return shortlex(self.word)

    def artin_word(self) -> list[tuple[int, int]]:
        return [(i, 1) for i in self.shortlex()]

    def is_right_descent(self, s: int):
        return is_right_descent(self.word, s)

    def right_descent_set(self):
        return right_descent_set(self.word)

    def is_left_descent(self, s: int):
        return is_left_descent(self.word, s)

    def left_descent_set(self):
        return left_descent_set(self.word)

    def inv(self):
        return Permutation(inverse(self.word))

    def __mul__(self, other):
        if isinstance(other, Permutation):
            if len(self.word) != len(other.word):
                raise ValueError(f"Cannot compose permutations of different lengths {len(self.word)} and {len(other.word)}")

            return Permutation(compose(self.word, other.word))

        return NotImplemented

    def _repr_latex_(self):
        if is_identity(self.word):
            return r'$\mathrm{id}$'
        return '$' + ' '.join(f's_{{{i}}}' for i in shortlex(self.word)) + '$'
