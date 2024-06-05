import numpy
import torch

from dops import grid


def test_arange():
    values = [1, 3, 5, 7]
    assert all(numpy.array(values) == grid.arange(1, 8, 2, default="numpy"))
    assert all(torch.tensor(values) == grid.arange(1, 8, 2, default="torch"))


def test_cat():
    a1 = [1, 2, 3, 4]
    a2 = [5, 6, 7, 8]
    assert all(numpy.array(a1 + a2) == grid.cat([a1, a2], default="numpy"))
    assert all(torch.tensor(a1 + a2) == grid.cat([a1, a2], default="torch"))


def test_empty_like():
    a = [[1, 2], [3, 4]]
    assert grid.empty_like(a, default="numpy").shape == (2, 2)
    assert grid.empty_like(a, default="torch").shape == (2, 2)


def test_linspace():
    for module in ["numpy", "torch"]:
        n = grid.linspace(0, 1, 5, default=module)
        assert min(n) == 0, f"min | {module}"
        assert max(n) == 1, f"max | {module}"
        assert len(n) == 5, f"len | {module}"


def test_meshgrid():
    a = [1, 2, 3]

    for module in ["numpy", "torch"]:
        mesh = grid.meshgrid(a, a, default=module)
        assert len(mesh) == 2, module
        assert mesh[0].shape == (len(a), len(a)), module
        assert mesh[1].shape == (len(a), len(a)), module
        for i, ai in enumerate(a):
            assert all(mesh[0][i, :] == ai), mesh[0]
            assert all(mesh[1][:, i] == ai), mesh[1]


def test_reshape():
    a = [1, 2, 3, 4]
    shape = (4, 1)
    a_numpy = numpy.array(a).reshape(shape)
    a_torch = torch.tensor(a).reshape(shape)

    assert all(grid.reshape(a, shape=(4, 1), default="numpy") == a_numpy)
    assert all(grid.reshape(a, shape=(4, 1), default="torch") == a_torch)


def test_stack():
    a = [1, 2, 3, 4]
    assert grid.stack([a, a], default="numpy", axis=0).shape == (2, 4)
    assert grid.stack([a, a], default="numpy", axis=1).shape == (4, 2)

    assert grid.stack([a, a], default="torch", axis=0).shape == (2, 4)
    assert grid.stack([a, a], default="torch", axis=1).shape == (4, 2)


def test_zeros_like():
    a = [[1, 2], [3, 4]]
    assert grid.zeros_like(a, default="numpy").shape == (2, 2)
    assert grid.zeros_like(a, default="torch").shape == (2, 2)
