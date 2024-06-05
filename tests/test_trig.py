import math

from tenops import trig
from tenops.manage import MODULE_TYPECAST, cast_to_dtype


def test_angle():
    x = 1.0j
    angle = math.pi / 2
    for module in MODULE_TYPECAST.keys():
        assert abs(trig.angle(x, default=module) - angle) < 1e-6, module
        assert abs(trig.angle(cast_to_dtype(x, module)) - angle) < 1e-6, module


def test_atan():
    x = 1.0
    angle = math.pi / 4
    for module in MODULE_TYPECAST.keys():
        assert abs(trig.atan(x, default=module) - angle) < 1e-6, module
        assert abs(trig.atan(cast_to_dtype(x, module)) - angle) < 1e-6, module


def test_atan2():
    x = 1.0
    y = -1.0
    angle = -math.pi / 4

    for module in MODULE_TYPECAST.keys():
        assert abs(trig.atan2(y=y, x=x, default=module) - angle) < 1e-6, module
        xm = cast_to_dtype(x, module)
        ym = cast_to_dtype(y, module)
        assert abs(trig.atan2(y=ym, x=xm) - angle) < 1e-6, module


def test_cos():
    x = math.pi / 3
    y = 0.5
    for module in MODULE_TYPECAST.keys():
        assert abs(trig.cos(x, default=module) - y) < 1e-6, module
        assert abs(trig.cos(cast_to_dtype(x, module)) - y) < 1e-6, module


def test_sin():
    x = math.pi / 6
    y = 0.5
    for module in MODULE_TYPECAST.keys():
        assert abs(trig.sin(x, default=module) - y) < 1e-6, module
        assert abs(trig.sin(cast_to_dtype(x, module)) - y) < 1e-6, module


def test_tan():
    x = math.pi / 4
    y = 1.0
    for module in MODULE_TYPECAST.keys():
        assert abs(trig.tan(x, default=module) - y) < 1e-6, module
        assert abs(trig.tan(cast_to_dtype(x, module)) - y) < 1e-6, module
