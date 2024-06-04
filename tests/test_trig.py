import math

import torch

from dops import trig


def test_angle():
    val = 1.0j
    angle = math.pi / 2
    assert abs(trig.angle(val, default="numpy") - angle) < 1e-6, "numpy"
    assert abs(trig.angle(torch.tensor(val)) - angle) < 1e-6, "torch"


def test_atan():
    val = 1.0
    angle = math.pi / 4
    assert abs(trig.atan(val, default="numpy") - angle) < 1e-6, "numpy"
    assert abs(trig.atan(torch.tensor(val)) - angle) < 1e-6, "torch"


def test_atan2():
    x = 1.0
    y = -1.0
    angle = -math.pi / 4
    assert abs(trig.atan2(y=y, x=x, default="numpy") - angle) < 1e-6, "numpy"
    assert abs(trig.atan2(y=torch.tensor(y), x=torch.tensor(x)) - angle) < 1e-6, "torch"


def test_cos():
    x = math.pi / 3
    val = 0.5
    assert abs(trig.cos(x, default="numpy") - val) < 1e-6, "numpy"
    assert abs(trig.cos(torch.tensor(x)) - val) < 1e-6, "torch"


def test_sin():
    x = math.pi / 6
    val = 0.5
    assert abs(trig.sin(x, default="numpy") - val) < 1e-6, "numpy"
    assert abs(trig.sin(torch.tensor(x)) - val) < 1e-6, "torch"


def test_tan():
    x = math.pi / 4
    val = 1.0
    assert abs(trig.tan(x, default="numpy") - val) < 1e-6, "numpy"
    assert abs(trig.tan(torch.tensor(x)) - val) < 1e-6, "torch"
