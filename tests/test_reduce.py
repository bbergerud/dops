from dops import reduce
from dops.manage import MODULE_TYPECAST, cast_to_dtype


def test_all():
    a = [[True, True], [True, True]]

    for module in MODULE_TYPECAST.keys():
        assert reduce.all(cast_to_dtype(a, module)), module
        assert reduce.all(a, default=module), module
