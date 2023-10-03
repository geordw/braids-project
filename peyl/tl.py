"""
TL: An implementation of the Temperley–Lieb category and algebras.

A TL diagram of type (m, n) has n dots along the bottom and m dots along the top. The bottom dots are numbered [0, n)
from left to right, and the top dots are numbered [n, n+m) from right to left. Since the diagram is a perfect matching
of points, it defines an involution on the set [0, n+m) with no fixed points: we record this data in an array M. Simply
put, the dot with index i is matched to the dot with index M[i]. Consequently, (m+n) must be even. Furthermore, the
matching M is crossingless if and only if it "looks like" a balanced bracketed sequence eg (()()(())). (This is the
reason that we read the top from right-to-left, so that the whole diagram can be "rotated down" to a morphism from
n+m to 0).
"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import Iterable, Sequence

from .lpoly import LPoly
from .matrix import Matrix


def is_crossingless_matching(seq: Sequence[int]) -> bool:
    """Return True if seq is a crossingless matching of N points, where N = len(seq)."""
    # Firstly, the sequence must be an involution of [0, N) with no fixed points.
    if not all(
        0 <= seq[i] < len(seq) and seq[i] != i and seq[seq[i]] == i
        for i in range(len(seq))
    ):
        return False

    # Secondly, the bracket sequence defined by the diagram must be balanced.
    stack = []
    for i in range(len(seq)):
        if i < seq[i]:
            stack.append(seq[i])
        else:
            if stack.pop() != i:
                return False

    return True


def generate_sequences(n: int) -> Iterable[tuple[int, ...]]:
    """Iterate over all crossingless matchings on n dots (n must be even)."""
    if n < 0 or n % 2 == 1:
        raise ValueError("Crossingless matchings may only be generated for positive even n.")

    # Use a bijection of binary trees on m vertices with strings of 2m balanced brackets. Each node in the three opens
    # a pair of parentheses: the left subtree goes inside these parantheses, while the right subtree goes after them.
    # For example, ((())) corresponds to a binary tree root -> left child -> left child, while ()()() corresponds to
    # root -> right child -> right child.
    seq = [-1] * n

    def place_trees(i: int, size: int):
        if size == 0:
            yield

        for left_size in range(size):
            # Place the root.
            l, r = i, i + 1 + 2*left_size
            seq[l], seq[r] = r, l

            # Place the left and right subtrees.
            for _ in place_trees(l+1, left_size):
                for _ in place_trees(r + 1, size - left_size - 1):
                    yield

    for _ in place_trees(0, n // 2):
        yield tuple(seq)


def compose(upper: Sequence[int], lower: Sequence[int], mid: int) -> tuple[tuple[int, ...], int]:
    """
    Compose involutions lower and upper, treating the number of dots in the middle as the number "mid". In other words,
    lower is treated as a TL diagram of type (mid, len(lower) - mid) and upper of type (len(upper) - mid, mid).
    """

    # Let U = len(upper). The numbering scheme on upper is:
    # [U-1, U-2, ..., mid]  size top = U - mid. Indexing scheme in result is v => v - mid + bot.
    # [0, 1, ..., mid - 1]  size mid.

    # Let L = len(lower). The numbering scheme on lower is:
    # [L-1, L-2, ..., L - mid]  size mid.
    # [0, ..., L - mid - 1]     size bot = L - mid. Indexing scheme is the same in result.

    # Labels on the middle row can be converted between indexing schemes via v => L - v - 1.

    U, L = len(upper), len(lower)
    assert 0 <= mid <= L and 0 <= mid <= U

    top, bot = U - mid, L - mid
    result = [-1] * (bot + top)

    # We will run through each vertex in the result, and follow the path it takes through the composition.
    # We will also mark the vertices we visit on the middle row (in coordinates given by the upper diagram) so that
    # we can trace the loops afterwards.
    visited = [False] * mid
    for i in range(len(result)):
        # Figure out if the source vertex is in the upper or lower diagram, and its index in that diagram.
        v, side = (i, 0) if i < bot else (i - bot + mid, 1)
        while True:
            # If we are in the lower diagram
            if side == 0:
                v = lower[v]
                if v < bot:
                    # v is now the target.
                    result[i] = v
                    break
                else:
                    v = L - v - 1
                    visited[v] = True
                    side = 1 - side

            # If we are in the upper diagram
            else:
                v = upper[v]
                if v >= mid:
                    result[i] = v - mid + bot
                    break
                else:
                    visited[v] = True
                    v = L - v - 1
                    side = 1 - side

    assert is_crossingless_matching(result)

    # Now we count the bubbles. Start at an unvisited vertex in the middle row, on the upper side. Apply the upper
    # involution followed by the lower involution repeatedly until we reach where we started from.
    bubbles = 0
    for i in range(mid):
        if visited[i]:
            continue

        v = i
        while True:
            visited[v] = True
            visited[upper[v]] = True
            v = L - lower[L - upper[v] - 1] - 1
            if visited[v]:
                break

        bubbles += 1

    return tuple(result), bubbles

def trace_count(seq: Sequence[int]) -> int:
    """
    Treating seq as a TL(2n, 0) morphism, compose with the nested-cups morphism to calculate the Markov trace.
    """
    nested_cups = list(range(len(seq)))[::-1]
    result, bubbles = compose(seq, nested_cups, len(seq))
    assert result == ()
    return bubbles

def insert(outer: Sequence[int], inner: Sequence[int], i: int) -> tuple[int, ...]:
    """Insert the inner sequence at index i of the outer sequence."""
    assert 0 <= i <= len(outer)
    return (
        *(x if x < i else x + len(inner) for x in outer[:i]),
        *(x + i for x in inner),
        *(x if x < i else x + len(inner) for x in outer[i:]),
    )


@dataclasses.dataclass(init=False, eq=True, unsafe_hash=True, order=True)
class TLDiag:
    top: int
    bot: int
    seq: tuple[int, ...]

    def __init__(self, top: int, bot: int, seq: Sequence[int], bubbles: int = 0):
        assert 0 <= top
        assert 0 <= bot
        assert top + bot == len(seq)
        assert is_crossingless_matching(seq)

        self.top = top
        self.bot = bot
        self.seq = tuple(seq)

    @classmethod
    def id(cls, n: int) -> TLDiag:
        return TLDiag(n, n, tuple(2 * n - i - 1 for i in range(2 * n)), 0)

    def __mul__(upper: TLDiag, lower: TLDiag) -> tuple[TLDiag, int]:
        """Vertical composition of TL diagrams, returning a pair (new diagram, # bubbles)."""
        if isinstance(lower, TLDiag):
            if upper.bot != lower.top:
                raise ValueError(f"Cannot compose diagrams of incompatible shapes ({upper.top}, {upper.bot}) and ({lower.top}, {lower.bot})")

            seq, bubbles = compose(upper.seq, lower.seq, upper.bot)
            return TLDiag(upper.top, lower.bot, seq), bubbles

    def __matmul__(self, other: TLDiag) -> TLDiag:
        """Horizontal composition of TL diagrams, returning a new TLDiag."""
        return TLDiag(
            self.top + other.top,
            self.bot + other.bot,
            insert(self.seq, other.seq, self.bot),
        )

    def flip(self) -> TLDiag:
        """Vertical flip of the diagram."""
        return TLDiag(
            self.bot,
            self.top,
            tuple(self.top + self.bot - 1 - i for i in reversed(self.seq)),
        )

    def _repr_html_(self):
        return draw_svg(self.seq, self.bot)

def through_strands(diag: TLDiag) -> int:
    """Return the number of through-strands of this diagram."""
    return sum(1 for i in range(diag.bot) if diag.seq[i] >= diag.bot)

def tl_basis(top: int, bot: int):
    """Return the list of basis elements for TL(top, bot)."""
    assert 0 <= top and 0 <= bot and (top + bot) % 2 == 0
    return [TLDiag(top, bot, seq) for seq in generate_sequences(top + bot)]

def tl_tableaux_basis(top: int, strands: int):
    assert 0 <= strands <= top and top % 2 == strands % 2
    return [diag for diag in tl_basis(top, strands) if through_strands(diag) == strands]

def cell_dimension(n: int, r: int):
    """Return the dimension of the cell module TLCell(n, r). See :py:func:`tl.cell_basis` for more information."""
    assert 0 <= n and 0 <= r and 0 <= n - 2 * r
    return math.comb(n, r) - (0 if r - 1 < 0 else math.comb(n, r - 1))

@functools.cache
def cell_basis(n: int, r: int) -> list[TLDiag]:
    """
    Return the diagrams of type (n, n - 2r) with exactly n - 2r through strands.

    Let TLCell(n, r) ⊆ TL(n, n - 2r) be those diagrams with exactly n - 2r through-strands. Split these diagrams into
    two groups: those with a vertical bar all the way to the right, and those without. For the group with a bar on the
    right, deleting that bar is a bijection with TLCell(n-1, r). For those without, pulling the top-right corner vertex
    to the bottom-right (i.e. reinterpreting the sequence in a different frame) is a bijection with TLCell(n-1, r-1).

    This gives a constructive recursion for the set of diagrams TLCell(n, r).
    """
    assert 0 <= n and 0 <= r
    if 0 > n - 2*r:
        return []
    if r == 0 or n == 0:
        return [TLDiag.id(n)]

    bar = TLDiag.id(1)
    return sorted([
        *[diag @ bar for diag in cell_basis(n-1, r)],
        *[TLDiag(n, n - 2*r, diag.seq) for diag in cell_basis(n-1, r-1)],
    ])


@dataclasses.dataclass(init=False, eq=True)
class TLExtDiag:
    """
    A TLDiag is a Temperley-Lieb diagram, together with a natural number of floating bubbles. The bubbles are treated
    as fully commutative, i.e. they don't get stuck on through-strands or anything.
    """
    top: int
    bot: int
    seq: tuple[int, ...]
    bubbles: int

    def __init__(self, top: int, bot: int, seq: Sequence[int], bubbles: int = 0):
        assert 0 <= top
        assert 0 <= bot
        assert top + bot == len(seq)
        assert is_crossingless_matching(seq)

        self.top = top
        self.bot = bot
        self.seq = tuple(seq)
        self.bubbles = bubbles

    @staticmethod
    def id(n: int):
        return TLExtDiag(n, n, tuple(2 * n - i - 1 for i in range(2 * n)), 0)

    @staticmethod
    def e(i: int, n: int):
        """The quasi-idempotent e_i in the algebra TLn."""
        assert n >= 0
        assert 0 <= i < n - 1

        seq = [2*n - i - 1 for i in range(2*n)]
        seq[i], seq[i+1] = i+1, i
        seq[2*n-i-1], seq[2*n-i-2] = 2*n-i-2, 2*n-i-1
        return TLExtDiag(n, n, tuple(seq), 0)

    def add_bubbles(self, bubbles: int):
        """Add a certain number of bubbles to this diagram."""
        return TLExtDiag(self.top, self.bot, self.seq, self.bubbles + bubbles)

    def split_bubbles(self) -> tuple[TLExtDiag, int]:
        """Split into a "pure" diagram with no bubbles, and the number of bubbles."""
        return TLExtDiag(self.top, self.bot, self.seq, 0), self.bubbles

    def __mul__(self, other: TLExtDiag):
        """Vertical composition of TL diagrams, x * y = x on top of y."""
        if isinstance(other, TLExtDiag):
            if self.bot != other.top:
                raise ValueError("Trying to compose incompatible TLDiags.")

            seq, bubbles = compose(other.seq, self.seq, self.bot)
            return TLExtDiag(self.top, other.bot, seq, self.bubbles + other.bubbles + bubbles)

        return NotImplemented

    def __matmul__(self, other: TLExtDiag):
        """Horizontal composition of TL diagrams, x @ y = x to the left of y."""
        if isinstance(other, TLExtDiag):
            return TLExtDiag(
                self.top + other.top,
                self.bot + other.bot,
                insert(self.seq, other.seq, self.bot),
                self.bubbles + other.bubbles,
            )

        return NotImplemented


# Quantum 2 = v + v^-1.
Q2 = LPoly(-1, (1, 0, 1))

@dataclasses.dataclass(eq=True)
class TLMor:
    """
    A TLMor represents a morphism in the Temperley-Lieb category, where bubbles evaluate to (v + v^-1).
    """
    top: int
    bot: int
    terms: dict[TLDiag, LPoly]

    def __init__(self, top: int, bot: int, terms: dict[TLDiag, LPoly]):
        for diag, coeff in terms.items():
            if diag.top != top or diag.bot != bot:
                raise ValueError(f"A diagram has type ({diag.top}, {diag.bot}), incompatible with ({top}, {bot})")

        self.top = top
        self.bot = bot
        self.terms = {diag: coeff for diag, coeff in sorted(terms.items()) if not coeff.is_zero()}

    @classmethod
    def id(cls, n: int):
        assert n >= 0
        return cls(n, n, {TLDiag.id(n): LPoly(0, (1,))})

    @classmethod
    def e(cls, i: int, n: int):
        """The quasi-idempotent e_i in the algebra TL(n, n)."""
        assert n >= 0
        assert 0 <= i < n - 1

        seq = [2 * n - i - 1 for i in range(2 * n)]
        seq[i], seq[i + 1] = i + 1, i
        seq[2 * n - i - 1], seq[2 * n - i - 2] = 2 * n - i - 2, 2 * n - i - 1
        return TLMor(n, n, {TLDiag(n, n, seq): LPoly(0, (1,))})

    @classmethod
    def basis(cls, top: int, bot: int) -> Iterable[TLMor]:
        """Return a basis of HomTL(bot, top)."""
        return [TLMor(top, bot, {diag: LPoly(0, (1,))}) for diag in tl_basis(top, bot)]

    @classmethod
    def from_tldiag(cls, diag: TLDiag) -> TLMor:
        return TLMor(diag.top, diag.bot, {diag: LPoly(0, (1,))})

    def support(self) -> Iterable[TLDiag]:
        """Return the elements over which this linear combination is supported."""
        return self.terms.keys()

    def pairs(self) -> Iterable[tuple[TLDiag, LPoly]]:
        return [(diag, coeff) for diag, coeff in self.terms.items()]

    def coefficient(self, diag: TLDiag):
        if self.top != diag.top or self.bot != diag.bot:
            raise ValueError("Incompatibly sized diagrams")

        return self.terms.get(diag, LPoly(0, ()))

    def is_zero(self):
        return len(self.terms) == 0

    def quotient(self, strands: int):
        """Kill all diagrams with less than the given number of through strands."""
        return TLMor(self.top, self.bot, {diag: coeff for diag, coeff in self.terms.items() if through_strands(diag) >= strands})

    def trace(self):
        """The Markov trace of this element."""
        result = LPoly(0, ())
        for diag, coeff in self.terms.items():
            result += coeff * Q2**trace_count(diag.seq)
        return result

    def __mul__(upper: TLMor, lower: TLMor):
        if isinstance(lower, (int, LPoly)):
            return TLMor(upper.top, upper.bot, {diag: coeff * lower for diag, coeff in upper.terms.items()})

        if isinstance(lower, TLDiag):
            lower = TLMor(lower.top, lower.bot, {lower: LPoly(0, (1,))})

        if not isinstance(lower, TLMor):
            return NotImplemented

        if upper.bot != lower.top:
            raise ValueError(f"Cannot compose TLMors of types ({upper.top}, {upper.bot}) and ({lower.top}, {lower.bot})")

        result = {}
        for up, c_up in upper.terms.items():
            for low, c_low in lower.terms.items():
                diag, bubbles = compose(up.seq, low.seq, upper.bot)
                tldiag = TLDiag(upper.top, lower.bot, diag)
                if tldiag not in result:
                    result[tldiag] = LPoly(0, ())

                result[tldiag] += Q2 ** bubbles * c_up * c_low

        return TLMor(upper.top, lower.bot, result)

    def __add__(self, other):
        if isinstance(other, int):
            other = LPoly(0, (other,))

        if isinstance(other, LPoly):
            if self.bot == self.top:
                return self + other * TLMor.id(self.bot)
            else:
                raise ValueError("Cannot insert a scalar into a non-algebra.")

        if isinstance(other, TLMor):
            if self.top != other.top or self.bot != other.bot:
                raise ValueError(f"Cannot add morphisms of different types ({self.top, self.bot}) and ({other.top, other.bot})")
            terms = {**self.terms}
            for diag, coeff in other.terms.items():
                if diag not in terms:
                    terms[diag] = coeff
                else:
                    terms[diag] += coeff

            return TLMor(self.top, self.bot, terms)

        return NotImplemented

    def flip(self, twist_v: bool = True):
        """Apply the upside-down flip to the diagrams. If twist_v is True, also map v -> v^-1."""
        vinv = LPoly(-1, (1,))
        return TLMor(self.bot, self.top, {
            diag.flip(): coeff.evaluate(vinv) for diag, coeff in self.terms.items()
        })

    def __neg__(self):
        return TLMor(self.top, self.bot, {diag: -coeff for diag, coeff in self.terms.items()})

    def __sub__(self, other):
        return self + (-other)

    def __rmul__(self, other):
        if isinstance(other, (int, LPoly)):
            return self * other

        return NotImplemented

    def __pow__(self, exp: int):
        if self.top != self.bot:
            raise ValueError("Cannot raise a morphism to the power in a non-algebra")

        if exp < 0:
            raise ValueError("Cannot use negative powers.")

        acc = TLMor.id(self.top)
        pow2 = self
        while exp:
            if exp % 2 == 1:
                acc = acc * pow2

            exp //= 2
            pow2 = pow2 * pow2

        return acc

    def _repr_html_(self):
        style = ''.join(
            line + ';' for line in [
                'display: flex',
                'flex-direction: row',
                'flex-wrap: wrap',
                'justify-content: flex-start',
                'align-items: center',
            ]
        )
        return f'<p style="{style}">' + '<span style="padding: 0 0.3em">$+$</span>'.join(
            '$(' + coeff.fmt('latex') + ')$' + draw_svg(diag.seq, diag.bot)
            for diag, coeff in self.terms.items()
        ) + '</p>'


@dataclasses.dataclass(frozen=True)
class TLCell:
    """Represents the cell module TLCell(n, r) ⊆ TL(n, n - 2r) of the Temperley-Leib algebra."""
    n: int
    r: int

    def __post_init__(self):
        assert 0 <= self.n and 0 <= self.r and 0 <= self.n - 2 * self.r, "Cell module invalid"

    def dim(self):
        return cell_dimension(self.n, self.r)

    def basis_diagrams(self):
        return cell_basis(self.n, self.r)

    def project_vector(self, mor: TLMor):
        """Return the projection of mor, as a vector, into this cell module."""
        assert mor.bot == self.n, f"Dimensions {mor.bot=} and {self.n=} incompatible."
        return mor.quotient(self.n - 2 * self.r)

    def project_leftmult_matrix(self, mor: TLMor):
        """
        Left-multiplication by mor induces an action on the cell module. Return the matrix of this action, in the
        order as given by .basis().
        """
        dim = self.dim()
        rows = [[None] * dim for _ in range(dim)]
        for j, vj in enumerate(self.basis_diagrams()):
            result = (mor * vj).quotient(self.n - 2 * self.r)
            for i, vi in enumerate(self.basis_diagrams()):
                rows[i][j] = result.terms.get(vi, LPoly(0, ()))

        return Matrix.from_rows(rows)

    def trace_scalar(self) -> LPoly:
        """
        The Markov trace acts on the matrix representation of this cell by the matrix trace, scaled by
        this particular quantum integer.
        """
        return LPoly.quantum_integer(self.n - 2 * self.r + 1)


class JonesRep:
    """Represents a direct sum of all cell modules."""
    n: int                  # Number of strands
    cells: list[TLCell]     # TLCell(n, 0), TLCell(n, 1), ...

    def __init__(self, n: int):
        assert n >= 0

        self.n = n
        self.cells = [TLCell(n, r) for r in range(n//2)]




def draw_svg(seq, bot):
    import svg

    # A better approach in the future would be to take max(top, bot) which determines what the
    # "most nested" diagram could be, and scale up the height a little?

    assert bot <= len(seq)
    top = len(seq) - bot

    M = 10  # Margin
    W = 15  # Spacing between dots
    C = 25  # Control point vertical rise

    width = 2 * M + (max(bot, top) - 1) * W
    height = 35

    def numx(v):
        return v if v < bot else len(seq) - v - 1
    def pos(v):
        return width / 2 + (0.5 + numx(v) - [top, bot][v < bot] / 2) * W, height if v < bot else 0

    drawn = [False] * len(seq)

    # Accumulate a large SVG path description.
    path = []

    for i in range(len(seq)):
        if drawn[i]:
            continue

        # Draw a cubic bezier from the source (sx, sy) to the target (tx, ty).
        # If the source and target lie on the same side, we need to scale the control points
        # by the relative distance between them.
        sx, sy = pos(i)
        tx, ty = pos(seq[i])
        dy = -C if i < bot else C

        # Connecting the same side to itself.
        if (i < bot and seq[i] < bot) or (i >= bot and seq[i] >= bot):
            dist = 1 / (1 + math.exp(-abs(i - seq[i]) / 4))
            path += [svg.M(sx, sy), svg.C(sx, sy + dy * dist, tx, ty + dy * dist, tx, ty)]

        # Connecting different sides
        else:
            path += [svg.M(sx, sy), svg.C(sx, sy + dy, tx, ty - dy, tx, ty)]

        drawn[i] = drawn[seq[i]] = True

    return svg.SVG(
        width=width,
        height=height,
        elements=[svg.Path(
            d=path,
            stroke='black',
            stroke_width=1,
            fill='none',
        )],
    ).as_str()
