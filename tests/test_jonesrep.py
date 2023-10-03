import pytest

from peyl import LPoly
from peyl.jonesrep import JonesCellRep, LKBRep, RepBase


@pytest.mark.parametrize("rep", [
    JonesCellRep(3, 1, 0),
    JonesCellRep(4, 1, 0),
    JonesCellRep(4, 2, 0),
    JonesCellRep(5, 1, 0),
    JonesCellRep(5, 2, 0),
    LKBRep(4, 0, LPoly(7, (1,)), LPoly(5, (1,)))
])
def test_artin_generator_relations(rep: RepBase):
    """
    Check that the generators of a representation satisfy the braid relations.
    """
    gens, invs = rep.artin_gens_invs()

    for i in range(len(gens) - 1):
        assert gens[i] * gens[i+1] * gens[i] == gens[i+1] * gens[i] * gens[i+1]

        for j in range(i + 2, len(gens)):
            assert gens[i] * gens[j] == gens[j] * gens[i]
