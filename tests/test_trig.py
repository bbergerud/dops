import cmath
import math

from tenops import trig
from tenops.manage import MODULE_TYPECAST, cast_to_dtype


def test_angle():
    x = 1.0j
    angle = math.pi / 2
    for module in MODULE_TYPECAST.keys():
        assert abs(trig.angle(x, default=module) - angle) < 1e-6, module
        assert abs(trig.angle(cast_to_dtype(x, module)) - angle) < 1e-6, module


def test_acos():
    x = 0.5
    y = math.acos(x)
    for module in MODULE_TYPECAST.keys():
        assert abs(trig.acos(x, default=module) - y) < 1e-6, module
        assert abs(trig.acos(cast_to_dtype(x, module)) - y) < 1e-6, module


def test_acosh():
    x = 1 + 1j
    y = cmath.log(x + cmath.sqrt(x - 1) * cmath.sqrt(x + 1))

    for module in MODULE_TYPECAST.keys():
        assert abs(trig.acosh(x, default=module) - y) < 1e-6, module
        assert abs(trig.acosh(cast_to_dtype(x, module)) - y) < 1e-6, module


def test_asin():
    x = 1.0
    y = math.asin(x)
    for module in MODULE_TYPECAST.keys():
        assert abs(trig.asin(x, default=module) - y) < 1e-6, module
        assert abs(trig.asin(cast_to_dtype(x, module)) - y) < 1e-6, module


def test_asinh():
    x = 1 + 1j
    y = cmath.asinh(x)
    for module in MODULE_TYPECAST.keys():
        assert abs(trig.asinh(x, default=module) - y) < 1e-6, module
        assert abs(trig.asinh(cast_to_dtype(x, module)) - y) < 1e-6, module


def test_atan():
    x = 1.0
    y = math.atan(x)
    for module in MODULE_TYPECAST.keys():
        assert abs(trig.atan(x, default=module) - y) < 1e-6, module
        assert abs(trig.atan(cast_to_dtype(x, module)) - y) < 1e-6, module


def test_atan2():
    x = 1.0
    y = -1.0
    angle = math.atan2(y, x)

    for module in MODULE_TYPECAST.keys():
        assert abs(trig.atan2(y=y, x=x, default=module) - angle) < 1e-6, module
        xm = cast_to_dtype(x, module)
        ym = cast_to_dtype(y, module)
        assert abs(trig.atan2(y=ym, x=xm) - angle) < 1e-6, module


def test_atanh():
    x = 1 + 1j
    y = cmath.atanh(x)
    for module in MODULE_TYPECAST.keys():
        assert abs(trig.atanh(x, default=module) - y) < 1e-6, module
        assert abs(trig.atanh(cast_to_dtype(x, module)) - y) < 1e-6, module


def test_cos():
    x = math.pi / 3
    y = math.cos(x)
    for module in MODULE_TYPECAST.keys():
        assert abs(trig.cos(x, default=module) - y) < 1e-6, module
        assert abs(trig.cos(cast_to_dtype(x, module)) - y) < 1e-6, module


def test_sin():
    x = math.pi / 6
    y = math.sin(x)
    for module in MODULE_TYPECAST.keys():
        assert abs(trig.sin(x, default=module) - y) < 1e-6, module
        assert abs(trig.sin(cast_to_dtype(x, module)) - y) < 1e-6, module


def test_tan():
    x = math.pi / 4
    y = math.tan(x)
    for module in MODULE_TYPECAST.keys():
        assert abs(trig.tan(x, default=module) - y) < 1e-6, module
        assert abs(trig.tan(cast_to_dtype(x, module)) - y) < 1e-6, module
