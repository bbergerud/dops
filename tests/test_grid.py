import numpy
import torch

from dops import grid, reduce
from dops.manage import MODULE_TYPECAST, cast_to_dtype


def test_arange():
    y = [1, 3, 5, 7]
    for module in MODULE_TYPECAST.keys():
        xm = grid.arange(1, 8, 2, default=module)
        ym = cast_to_dtype(y, module=module)
        assert reduce.all(xm == ym), module


def test_cat():
    a1 = [1, 2, 3, 4]
    a2 = [5, 6, 7, 8]
    for module in MODULE_TYPECAST.keys():
        y = cast_to_dtype(a1 + a2, module=module)
        m1 = cast_to_dtype(a1, module=module)
        m2 = cast_to_dtype(a2, module=module)
        assert reduce.all(y == grid.cat([a1, a2], default=module)), module
        assert reduce.all(y == grid.cat([m1, m2], default=module)), module


def test_empty_like():
    a = [[1, 2], [3, 4]]
    for module in MODULE_TYPECAST.keys():
        assert grid.empty_like(a, default=module).shape == (2, 2)
        assert grid.empty_like(cast_to_dtype(a, module)).shape == (2, 2)


def test_linspace():
    for module in MODULE_TYPECAST.keys():
        n = grid.linspace(0, 1, 5, default=module)
        assert min(n) == 0, f"min | {module}"
        assert max(n) == 1, f"max | {module}"
        assert len(n) == 5, f"len | {module}"


def test_meshgrid():
    a = [1, 2, 3]

    for module in MODULE_TYPECAST.keys():
        for mesh in [
            grid.meshgrid(a, a, default=module),
            grid.meshgrid((x := cast_to_dtype(a, module)), x),
        ]:
            assert len(mesh) == 2, module
            assert mesh[0].shape == (len(a), len(a)), module
            assert mesh[1].shape == (len(a), len(a)), module
            for i, ai in enumerate(a):
                assert reduce.all(mesh[0][i, :] == ai), mesh[0]
                assert reduce.all(mesh[1][:, i] == ai), mesh[1]


def test_reshape():
    a = [1, 2, 3, 4]
    shape = (4, 1)

    for module in MODULE_TYPECAST.keys():
        assert grid.reshape(a, shape=shape, default=module).shape == shape, module
        assert grid.reshape(cast_to_dtype(a, module), shape).shape == shape, module


def test_stack():
    a = [1, 2, 3, 4]
    for module in MODULE_TYPECAST.keys():

        assert grid.stack([a, a], default=module, axis=0).shape == (2, 4), module
        assert grid.stack([a, a], default=module, axis=1).shape == (4, 2), module

        x = cast_to_dtype(a, module)
        assert grid.stack([x, x], axis=0).shape == (2, 4), module
        assert grid.stack([x, x], axis=1).shape == (4, 2), module


def test_zeros_like():
    a = [[1, 2], [3, 4]]
    y = 0

    for module in MODULE_TYPECAST.keys():
        for x in [
            grid.zeros_like(a, default=module),
            grid.zeros_like(cast_to_dtype(a, module)),
        ]:
            assert x.shape == (2, 2), module
            assert reduce.all(x == y), module
