import numpy as np

from rljax.buffer.segment_tree import MinTree, SumTree


def test_tree_set():
    tree = SumTree(4)
    tree[2] = 1.0
    tree[3] = 3.0

    assert np.isclose(tree.reduce(0, 2), 0.0)
    assert np.isclose(tree.reduce(0, 3), 1.0)
    assert np.isclose(tree.reduce(0, 4), 4.0)
    assert np.isclose(tree.reduce(2, 3), 1.0)
    assert np.isclose(tree.reduce(2, 4), 4.0)


def test_tree_set_overlap():
    tree = SumTree(4)
    tree[2] = 1.0
    tree[2] = 3.0

    assert np.isclose(tree.reduce(0, 2), 0.0)
    assert np.isclose(tree.reduce(0, 4), 3.0)
    assert np.isclose(tree.reduce(1, 2), 0.0)
    assert np.isclose(tree.reduce(2, 3), 3.0)
    assert np.isclose(tree.reduce(2, 4), 3.0)
    assert np.isclose(tree.reduce(3, 4), 0.0)


def test_prefixsum_idx():
    tree = SumTree(4)
    tree[2] = 1.0
    tree[3] = 3.0

    assert tree.find_prefixsum_idx(0.0) == 2
    assert tree.find_prefixsum_idx(0.5) == 2
    assert tree.find_prefixsum_idx(0.99) == 2
    assert tree.find_prefixsum_idx(1.01) == 3
    assert tree.find_prefixsum_idx(3.00) == 3
    assert tree.find_prefixsum_idx(4.00) == 3


def test_prefixsum_idx2():
    tree = SumTree(4)
    tree[0] = 0.5
    tree[1] = 1.0
    tree[2] = 1.0
    tree[3] = 3.0

    assert tree.find_prefixsum_idx(0.00) == 0
    assert tree.find_prefixsum_idx(0.55) == 1
    assert tree.find_prefixsum_idx(0.99) == 1
    assert tree.find_prefixsum_idx(1.51) == 2
    assert tree.find_prefixsum_idx(3.00) == 3
    assert tree.find_prefixsum_idx(5.50) == 3


def test_max_interval_tree():
    tree = MinTree(4)
    tree[0] = 1.0
    tree[2] = 0.5
    tree[3] = 3.0

    assert np.isclose(tree.reduce(0, 2), 1.0)
    assert np.isclose(tree.reduce(0, 3), 0.5)
    assert np.isclose(tree.reduce(0, 4), 0.5)
    assert np.isclose(tree.reduce(2, 4), 0.5)
    assert np.isclose(tree.reduce(3, 4), 3.0)

    tree[2] = 0.7

    assert np.isclose(tree.reduce(0, 2), 1.0)
    assert np.isclose(tree.reduce(0, 3), 0.7)
    assert np.isclose(tree.reduce(0, 4), 0.7)
    assert np.isclose(tree.reduce(2, 4), 0.7)
    assert np.isclose(tree.reduce(3, 4), 3.0)

    tree[2] = 4.0

    assert np.isclose(tree.reduce(0, 2), 1.0)
    assert np.isclose(tree.reduce(0, 3), 1.0)
    assert np.isclose(tree.reduce(0, 4), 1.0)
    assert np.isclose(tree.reduce(2, 3), 4.0)
    assert np.isclose(tree.reduce(2, 4), 3.0)
    assert np.isclose(tree.reduce(3, 4), 3.0)
