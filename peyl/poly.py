"""
Cyclotomic polynomials.

In this module, a polynomial over the integers is represented as a tuple, for instance 1 - 2x + x^3 would be the tuple
(1, -2, 0, 1).
"""
from __future__ import annotations

import dataclasses
import functools
import itertools


@dataclasses.dataclass(init=False, eq=True)
class Poly:
    """
    A polynomial over the integers, represented by a dense list of coefficients starting with the constant term. Thus
    Poly(1) is the integer 1, and Poly(0, 1) is the variable x.

    >>> Poly(1, 0, 1)
    Poly('x^2 + 1')
    """
    coeffs: tuple[int, ...]

    def __init__(self, *args: int):
        # Trim trailing zeros.
        end = len(args)
        while end >= 1 and args[end - 1] == 0:
            end -= 1

        self.coeffs = tuple(args[:end])

    def deg(self) -> int:
        """The degree of a polynomial is the degree of its leading term. The zero polynomial has degree -1."""
        return len(self.coeffs) - 1

    def is_zero(self) -> bool:
        return len(self.coeffs) == 0

    def evaluate(self, x):
        """
        Evaluate the polynomial at some point.

        >>> Poly(1, 1, 1).evaluate(Poly(0, 2))
        Poly('4x^2 + 2x + 1')
        """
        return sum(c * x**i for i, c in enumerate(self.coeffs) if c != 0)

    def __repr__(self):
        """
        >>> Poly(0,)
        Poly('0')
        >>> Poly(1,)
        Poly('1')
        >>> Poly(-1,)
        Poly('-1')
        >>> Poly(0, 1)
        Poly('x')
        >>> Poly(0, 0, 2)
        Poly('2x^2')
        >>> Poly(-1, 0, 2)
        Poly('2x^2 - 1')
        """
        if len(self.coeffs) == 0:
            return "Poly('0')"

        parts = []
        for i, c in reversed(list(enumerate(self.coeffs))):
            if c == 0:
                continue

            sign = ' + ' if (c > 0 and parts) else ' - ' if (c < 0 and parts) else '' if c > 0 else '-'
            term = '' if i == 0 else 'x' if i == 1 else f'x^{i}'
            coeff = '1' if (i == 0 and term == '') else '' if abs(c) == 1 else f'{abs(c)}'
            parts += [sign + coeff + term]

        return f"Poly('{''.join(parts)}')"

    def __add__(self, other: int | Poly) -> Poly:
        coeffs = [other] if isinstance(other, int) else other.coeffs
        return Poly(*(c + d for c, d in itertools.zip_longest(self.coeffs, coeffs, fillvalue=0)))

    def __sub__(self, other: int | Poly) -> Poly:
        coeffs = [other] if isinstance(other, int) else other.coeffs
        return Poly(*(c - d for c, d in itertools.zip_longest(self.coeffs, coeffs, fillvalue=0)))

    def __neg__(self) -> Poly:
        return Poly(*(-c for c in self.coeffs))

    def __mul__(self, other: int | Poly) -> Poly:
        if isinstance(other, int):
            return Poly(*(c*other for c in self.coeffs))
        elif isinstance(other, Poly):
            result = [0] * (len(self.coeffs) + len(other.coeffs))
            for (i, c), (j, d) in itertools.product(enumerate(self.coeffs), enumerate(other.coeffs)):
                result[i+j] += c*d
            return Poly(*result)

    __radd__ = __add__
    __rmul__ = __mul__

    def __pow__(self, n: int):
        if n < 0:
            raise ValueError("Cannot invert a polynomial.")
        if n == 0:
            return Poly(1)
        if n == 1:
            return self

        sqrt = self ** (n // 2)
        return sqrt * sqrt if n % 2 == 0 else sqrt * sqrt * self



    def __divmod__(n: Poly, d: Poly) -> tuple[Poly, Poly]:
        """
        Return the quotient and remainder of n/d, i.e. the unique solution to n = dq + r where deg(r) < deg(n).

        >>> divmod(Poly(), Poly(1))                 # 0 / 1
        (Poly('0'), Poly('0'))
        >>> divmod(Poly(-1, 0, 1), Poly(1, 1))      # x^2 - 1 / x + 1
        (Poly('x - 1'), Poly('0'))
        >>> divmod(Poly(-1, 0, 0, 1), Poly(-1, 1))  # x^3 - 1 / x - 1
        (Poly('x^2 + x + 1'), Poly('0'))
        """
        assert not all(c == 0 for c in d.coeffs)

        q, r = Poly(), n
        while r != () and len(r.coeffs) >= len(d.coeffs):
            t, remainder = divmod(r.coeffs[-1], d.coeffs[-1])
            if remainder != 0:
                raise ValueError(f"{r.coeffs[-1]} is not divisible by {d.coeffs[-1]}")

            t_deg = len(r.coeffs) - len(d.coeffs)
            tt = Poly(*((0,) * t_deg + (t,)))
            q, r = q + tt, r - d * tt

        return q, r

    def __truediv__(n: Poly, d: Poly) -> Poly:
        """Return n/d if n is divisible by d, otherwise raise an error."""
        quo, rem = divmod(n, d)
        if not rem.is_zero():
            raise ValueError(f"{n} is not divisible by {d}: it has remainder {rem}")

        return quo

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def cyclotomic(n: int) -> Poly:
        """

        >>> print(*(f"Cyc({n}) = {Poly.cyclotomic(n)}" for n in range(1, 5)), sep='\\n')
        Cyc(1) = Poly('x - 1')
        Cyc(2) = Poly('x + 1')
        Cyc(3) = Poly('x^2 + x + 1')
        Cyc(4) = Poly('x^2 + 1')
        """
        if n <= 0:
            raise ValueError("The argument to cyclotomic must be positive.")

        # Start with the polynomial x^n - 1, and divide by the all the cyclotomic polynomials for the divisors of n.
        poly = Poly(-1, *[0]*(n-1), 1)
        for d in (d for d in range(1, n) if n % d == 0):
            poly, rem = divmod(poly, Poly.cyclotomic(d))
            assert rem.is_zero()

        return poly

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def chebyshevT(n: int) -> Poly:
        """
        Return the Chebyshev polynomial of the first kind T(n).

        >>> Poly.chebyshevT(5)
        Poly('16x^5 - 20x^3 + 5x')
        """

        if n < 0:
            raise ValueError('The argument to chebyshevT must be >= 0')

        if n == 0:
            return Poly(1)
        if n == 1:
            return Poly(0, 1)

        return Poly(0, 2) * Poly.chebyshevT(n - 1) - Poly.chebyshevT(n - 2)

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def minpolycos(n: int):
        """
        Return the minimal polynomial of the algebraic integer 2 cos(2 pi / n).

        >>> Poly.minpolycos(9)
        Poly('x^3 - 3x + 1')
        """

        # According to https://www.jstor.org/stable/2324301, we have that if n = 2s + 1 is odd, then the difference
        # T(s+1) - T(s) of Chebyshev polynomials of the first kind is equal to 2^s times the product of these minimal
        # polynomials over all d dividing n. Similarly for the even case n = 2s, by taking T(s+1) - T(s-1).
        # Actually they are talking about the minimal polynomial of cos(2 pi / n) rather than 2 cos(2 pi / n), so we
        # will need to do some adjustment at the end, essentially subbing x=1/2 in.
        s = n // 2
        poly = 2 * (Poly.chebyshevT(s + 1) - Poly.chebyshevT(s if n % 2 == 1 else s - 1))
        for d in (d for d in range(1, n) if n % d == 0):
            poly /= Poly.minpolycos(d).evaluate(x=Poly(0, 2))

        assert(c % 2**i == 0 for i, c in enumerate(poly.coeffs))
        return Poly(*(c // 2**i for i, c in enumerate(poly.coeffs)))
