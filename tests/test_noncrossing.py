from peyl import noncrossing, permutations


def test_is_parallel_descending_cycles():
    """Make sure we get Catalan-many of these for each symmetric group."""

    for n in range(6):
        count = sum(
            1 for perm in permutations.SymmetricGroup(n).elements()
            if permutations.is_parallel_descending_cycles(perm.word)
        )
        assert count == noncrossing.catalan(n)
