from __future__ import annotations

import dataclasses
import itertools
from typing import Callable, Literal, Sequence

from .poly import Poly


class LPolyRing:
    """The commutative ring of Laurent polynomials."""


@dataclasses.dataclass(init=False, eq=True, unsafe_hash=True)
class LPoly:
    """
    A Laurent polynomial over the integers, represented by a valuation and a dense list of coefficients, starting from
    the valuation power and going up. The zero polynomial is represented by a valuation of 0 and an empty coefficient
    list.

    >>> LPoly(0, ())
    LPoly('0')
    >>> LPoly(0, (0, 1))
    LPoly('v')
    >>> LPoly(1, (1,))
    LPoly('v')
    >>> LPoly(1, (1,))**-1
    LPoly('v^-1')
    """
    val: int
    coeffs: tuple[int, ...]

    def __init__(self, val: int, coeffs: Sequence[int]):
        # Trim leading and trailing zeros.
        l, r = 0, len(coeffs)
        while l < r and coeffs[l] == 0:
            l += 1
            val += 1
        while l < r and coeffs[r - 1] == 0:
            r -= 1

        if l == r:
            self.val = 0
            self.coeffs = ()
        else:
            self.val = val
            self.coeffs = tuple(coeffs[l:r])

    def is_zero(self) -> bool:
        return len(self.coeffs) == 0

    def is_unit(self) -> bool:
        return len(self.coeffs) == 1 and self.coeffs[0] in (1, -1)

    def evaluate(self, x):
        """
        Evaluate the Laurent polynomial at some point.

        >>> LPoly(0, (1, 1, 1)).evaluate(LPoly(0, (0, 2)))
        LPoly('4v^2 + 2v + 1')
        """
        return sum(c * x**(i + self.val) for i, c in enumerate(self.coeffs) if c != 0)

    def valuation(self):
        """
        The valuation of a Laurent polynomial is the degree of the lowest power of v, or garbage (0)
        for the zero polynomial.
        """
        return self.val

    def degree(self):
        """
        The degree of a Laurent polynomial is the degree of the highest power of v, or garbage (-1)
        for the zero polynomial.
        """
        return self.val + len(self.coeffs) - 1

    def __repr__(self):
        """
        >>> LPoly(0, ())
        LPoly('0')
        >>> LPoly(0, (1,))
        LPoly('1')
        >>> LPoly(0, (-1,))
        LPoly('-1')
        >>> LPoly(0, (0, 1))
        LPoly('v')
        >>> LPoly(0, (0, 0, 2))
        LPoly('2v^2')
        >>> LPoly(0, (-1, 0, 2))
        LPoly('2v^2 - 1')
        >>> LPoly(-1, (1, 0, 1))**2
        LPoly('v^2 + 2 + v^-2')
        """

        return f"LPoly('{self.fmt()}')"

    def fmt(self, mode: Literal[None, 'latex'] = None):
        if len(self.coeffs) == 0:
            return '0'

        power_fmt = 'v^{}' if mode is None else 'v^{{{}}}'

        parts: list[str] = []
        for i, c in reversed(list(enumerate(self.coeffs, self.val))):
            if c == 0:
                continue

            sign = ' + ' if (c > 0 and parts) else ' - ' if (c < 0 and parts) else '' if c > 0 else '-'
            term = '' if i == 0 else 'v' if i == 1 else power_fmt.format(i)
            coeff = f'{abs(c)}' if (i == 0 and term == '') else '' if abs(c) == 1 else f'{abs(c)}'
            parts += [sign + coeff + term]

        return ''.join(parts)

    def _repr_latex_(self):
        return self.fmt(mode='latex')

    def __add__(self, other: int | LPoly) -> LPoly:
        if isinstance(other, int):
            return self + LPoly(0, (other,))
        if isinstance(other, LPoly):
            # Minval is the smallest valuation in the result.
            # Maxpow is one more than the largest power which occurs.
            minval = min(self.val, other.val)
            maxpow = max(self.val + len(self.coeffs), other.val + len(other.coeffs))

            coeffs = [0] * (maxpow - minval)
            for i, c in enumerate(self.coeffs):
                coeffs[self.val + i - minval] += c
            for i, c in enumerate(other.coeffs):
                coeffs[other.val + i - minval] += c

            return LPoly(minval, coeffs)

        return NotImplemented

    def __sub__(self, other: int | LPoly):
        return self + (-other)

    def __rsub__(self, other: int | LPoly):
        return (-self) + other

    def __neg__(self):
        return LPoly(self.val, tuple(-x for x in self.coeffs))

    def __mul__(self, other: int | LPoly) -> LPoly:
        if isinstance(other, int):
            return LPoly(self.val, tuple(c*other for c in self.coeffs))
        elif isinstance(other, LPoly):
            result = [0] * (len(self.coeffs) + len(other.coeffs))
            for (i, c), (j, d) in itertools.product(enumerate(self.coeffs), enumerate(other.coeffs)):
                result[i+j] += c*d
            return LPoly(self.val + other.val, result)

        return NotImplemented

    __radd__ = __add__
    __rmul__ = __mul__

    def __pow__(self, n: int):
        if n < 0:
            if self.coeffs == (1,) or self.coeffs == (-1,):
                return LPoly(-self.val, self.coeffs)**-n
            raise ValueError("Cannot invert a general Laurent polynomial.")
        if n == 0:
            return LPoly(0, (1,))
        if n == 1:
            return self

        sqrt = self ** (n // 2)
        return sqrt * sqrt if n % 2 == 0 else sqrt * sqrt * self

    def __truediv__(self, other: LPoly):
        return LPoly(self.val - other.val, (Poly(*self.coeffs) / Poly(*other.coeffs)).coeffs)

    def map(self, f: Callable[[int], int]):
        """
        Map a function over the coefficients.
        Takes a + bv + cv^2 to f(a) + f(b) v + f(c) v^2.
        """
        return LPoly(self.val, [f(c) for c in self.coeffs])

    @staticmethod
    def coerce(other: int | LPoly) -> LPoly:
        if isinstance(other, int):
            return LPoly(0, (other,))
        if isinstance(other, LPoly):
            return other
        raise ValueError(f"Cannot coerce {other} to an LPoly")

    @staticmethod
    def quantum_integer(n: int):
        """Return the quantum integer [n]."""
        if abs(n) <= 1:
            return LPoly(0, (n,))

        sign = 1 if n >= 1 else -1
        return LPoly(-abs(n) + 1, [sign if i % 2 == 0 else 0 for i in range(2*abs(n) - 1)])
