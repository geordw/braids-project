import unittest

from peyl.tl import TLExtDiag, generate_sequences


class TestTLFunctions(unittest.TestCase):
    def test_generate(self):
        self.assertEqual({()}, set(generate_sequences(0)))
        self.assertEqual({(1, 0)}, set(generate_sequences(2)))
        self.assertEqual({(1, 0, 3, 2), (3, 2, 1, 0)}, set(generate_sequences(4)))
        self.assertEqual({
            (1, 0, 3, 2, 5, 4), # ()()()
            (1, 0, 5, 4, 3, 2), # ()(())
            (3, 2, 1, 0, 5, 4), # (())()
            (5, 2, 1, 4, 3, 0), # (()())
            (5, 4, 3, 2, 1, 0), # ((()))
        }, set(generate_sequences(6)))


class TestTLDiag(unittest.TestCase):
    def test_id(self):
        for i in range(5):
            id = TLExtDiag.id(i)
            self.assertEqual(id * id, id)

    def test_tensor_id(self):
        self.assertEqual(TLExtDiag.id(2), TLExtDiag.id(1) @ TLExtDiag.id(1))
        self.assertEqual(TLExtDiag.id(5), TLExtDiag.id(3) @ TLExtDiag.id(2))

    def test_ei_rels(self):
        for n in range(2, 6):
            for i in range(n - 1):
                ei = TLExtDiag.e(i, n)
                self.assertEqual(ei*ei, ei.add_bubbles(1))

                for j in [j for j in [i - 1, i + 1] if 0 <= j < n - 1]:
                    ej = TLExtDiag.e(j, n)
                    self.assertEqual(ei, ei*ej*ei)

                for j in [j for j in range(n - 1) if abs(j - i) >= 2]:
                    ej = TLExtDiag.e(j, n)
                    self.assertEqual(ei*ej, ej*ei)
