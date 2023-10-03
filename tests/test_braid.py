import functools
import operator

import pytest

from peyl import DGNF, GNF, BraidGroup

# def test_s3_braids():
#     id = GNF.identity(3)
#     assert id.normal_form() == (0, [])

#     s = id.rmult(0)
#     assert s.normal_form() == (0, [(1, 0, 2)])

#     t = id.rmult(1)
#     assert t.normal_form() == (0, [(0, 2, 1)])

#     st = s.rmult(1)
#     assert st.normal_form() == (0, [(1, 2, 0)])

#     ts = t.rmult(0)
#     assert ts.normal_form() == (0, [(2, 0, 1)])

#     sts = st.rmult(0)
#     tst = ts.rmult(1)
#     assert sts == tst
#     assert sts.normal_form() == (1, [])


def test_consistency():
    # Should not be able to construct a GNF with a factor of zero (corresponding to identity).
    with pytest.raises(AssertionError):
        GNF(4, 0, (0,))

    # Should not be able to construct a DGNF with a factor of zero (corresponding to finest partition).
    with pytest.raises(AssertionError):
        DGNF(4, 0, (0,))

def test_nf_operations():
    # Creation
    GNF.identity(4)
    DGNF.identity(4)

    assert len(list(GNF.identity(4).nf_suffixes(length=0))) == 1
    assert len(list(GNF.identity(4).nf_suffixes(length=1))) == 22
    assert len(list(GNF.identity(4).nf_suffixes(length=2))) == GNF.count_all_of_length(n=4, length=2)


# This braid is presented in the Artin generators, and should have Garisde length 13
# and dual length 14. It is in the kernel of the Burau for B4 over F2.
KER2_BRAID = [
    1, 0, 2, 0, 1, 2, 1, 1, 2, 1, 0, 0, 2, 2, 1, 1, 0, 2, 0, 1, 2,
    1, 0, 0, 2, 1, 1, 0, 2, 0, 2, 1, 0, 1, 0, 2, 0, 2, 1, 1, 0, 2,
]

def test_gnf():
    """Spot-test some functions on the GNF representation."""
    gens = BraidGroup(4).gens()
    ker2_braid: GNF = functools.reduce(operator.mul, [gens[i] for i in KER2_BRAID], GNF.identity(4))

    assert ker2_braid.inf() == 0
    assert ker2_braid.canonical_length() == 13

    square = ker2_braid * ker2_braid

    assert square.inf() == 2
    assert square.canonical_length() == 22
    assert square * square.inv() == GNF.identity(4)

    assert GNF.from_artin_word(square.n, square.artin_word()) == square
    assert square.to_DGNF().to_GNF() == square


def test_dgnf():
    id = DGNF.identity(4)
    assert id.canonical_decomposition() == (0, ())

    bkl_gens = BraidGroup(4).bkl_gens()
    assert bkl_gens[2, 1] * bkl_gens[1, 0] == bkl_gens[1, 0] * bkl_gens[0, 2]

    ker2_braid: DGNF = functools.reduce(operator.mul, [bkl_gens[i, i+1] for i in KER2_BRAID], DGNF.identity(4))

    assert ker2_braid.inf() == 6
    assert ker2_braid.canonical_length() == 14

    square = ker2_braid * ker2_braid

    assert square.inf() == 15
    assert square.canonical_length() == 23
    assert square * square.inv() == DGNF.identity(4)

    assert DGNF.from_artin_word(square.n, square.artin_word()) == square
    assert square.to_GNF().to_DGNF() == square

    assert square.to_GNF().to_DGNF() == square


def test_braid_group():
    B = BraidGroup(4)
    assert B.id() == GNF.identity(4)
    assert list(B.gens()) == list(GNF.artin_gens(4))
    assert B.bkl_gens() == DGNF.bkl_gens(4)
