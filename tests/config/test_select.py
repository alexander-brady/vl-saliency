from vl_saliency.config.select import HeadSelect, LayerSelect


def test_layer_select_unique_sorted():
    ls = LayerSelect(2, 0, 2, 1)
    assert ls.layers == (0, 1, 2)
    assert list(ls) == [0, 1, 2]


def test_head_select_grouping_and_iteration():
    hs = HeadSelect((0, 2), (0, 0), (1, 1), (0, 2))
    exp_dict = {0: (0, 2), 1: (1,)}

    assert hs.heads == exp_dict
    assert hs.items() == exp_dict.items()

    assert list(hs) == [
        (0, 0),
        (0, 2),
        (1, 1),
    ]


def test_head_select_contains():
    hs = HeadSelect((0, 0), (1, 1))

    assert (0, 0) in hs
    assert (1, 1) in hs
    assert (0, 1) not in hs
