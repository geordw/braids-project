import unittest

from peyl.lpoly import LPoly


class TestLPoly(unittest.TestCase):
    def test_binomial(self):
        q2 = LPoly(-1, (1, 0, 1))
        self.assertEqual(LPoly(0, (1,)), q2 ** 0)
        self.assertEqual(LPoly(-1, (1, 0, 1)), q2 ** 1)
        self.assertEqual(LPoly(-2, (1, 0, 2, 0, 1)), q2 ** 2)
        self.assertEqual(LPoly(-3, (1, 0, 3, 0, 3, 0, 1)), q2 ** 3)

    def test_near_addition(self):
        x = LPoly(0, (0, 1))
        y = LPoly(0, (0, -1))
        self.assertEqual(LPoly(0, ()), x + y)

    def test_far_addition(self):
        x = LPoly(-2, (1, 1))
        y = LPoly(2, (1, 1))
        self.assertEqual(LPoly(-2, (1, 1, 0, 0, 1, 1)), x + y)

    def test_far_multiplication(self):
        x = LPoly(-5, (1, 1))
        y = LPoly(5, (1, 1))
        self.assertEqual(LPoly(0, (1, 2, 1)), x * y)

    def test_rsub(self):
        # Because I'm not sure if arguments need to be switched for rsub.
        v = LPoly(0, (0, 1))
        assert 1 - v == LPoly(0, (1, -1))

    def test_power(self):
        one = LPoly(0, (1,))
        v = LPoly(0, (0, 1))
        vi = LPoly(-1, (1,))
        assert v**0 == one

        assert v**1 == v
        assert v**2 == v*v
        assert v**3 == v*v*v
        assert v**4 == v*v*v*v

        assert v**-1 == vi
        assert v**-2 == vi*vi
        assert v**-3 == vi*vi*vi
        assert v**-4 == vi*vi*vi*vi

    def test_quantum_integer(self):
        one = LPoly(0, (1,))
        v = LPoly(0, (0, 1))

        assert LPoly.quantum_integer(0) == 0 * one
        assert LPoly.quantum_integer(1) == one
        assert LPoly.quantum_integer(2) == v + v**-1
        assert LPoly.quantum_integer(3) == v**2 + 1 + v**-2
        assert LPoly.quantum_integer(4) == v**3 + v + v**-1 + v**-3
        assert LPoly.quantum_integer(5) == v**4 + v**2 + 1 + v**-2 + v**-4

        for i in range(10):
            assert LPoly.quantum_integer(i) == -LPoly.quantum_integer(-i)
