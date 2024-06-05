from tenops import special
from tenops.manage import MODULE_TYPECAST, cast_to_dtype


def test_exp():
    x = 0.0
    y = 1.0

    for module in MODULE_TYPECAST.keys():
        assert abs(special.exp(x, default=module) - y) < 1e-6, module
        assert abs(special.exp(cast_to_dtype(x, module)) - y) < 1e-6, module


def test_log():
    x = 1.0
    y = 0.0
    for module in MODULE_TYPECAST.keys():
        assert abs(special.log(x, default=module) - y) < 1e-6, module
        assert abs(special.log(cast_to_dtype(x, module)) - y) < 1e-6, module
