from trajectory.rank_compare import disagreement_count


def test_identical_rankings_disagree_on_nothing():
    assert disagreement_count(["a", "b", "c"], ["a", "b", "c"]) == 0


def test_fully_reversed_ranking_disagrees_on_all_three():
    assert disagreement_count(["a", "b", "c"], ["c", "b", "a"]) == 2
    # middle position (b) matches in a 3-element full reversal; only the
    # two end positions swap, which is the case this test pins down.


def test_single_swap_disagrees_on_exactly_two_positions():
    assert disagreement_count(["a", "b", "c"], ["b", "a", "c"]) == 2
