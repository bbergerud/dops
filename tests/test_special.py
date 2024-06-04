import torch

from dops import special


def test_exp():
    x = 0.0
    y = 1.0
    assert abs(special.exp(x, default="numpy") - y) < 1e-6, "numpy"
    assert abs(special.exp(torch.tensor(x)) - y) < 1e-6, "torch"


def test_log():
    x = 1.0
    y = 0.0
    assert abs(special.log(x, default="numpy") - y) < 1e-6, "numpy"
    assert abs(special.log(torch.tensor(x)) - y) < 1e-6, "torch"
