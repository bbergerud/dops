from collections import defaultdict

from .manage import (
    DEFAULT_MODULE,
    TYPEHINT_DTYPE,
    TYPEHINT_MODULE,
    cast_to_dtype,
    get_module_attr,
    get_module_from_object,
)
from .utils import AttrHandler


def exp(x: TYPEHINT_DTYPE, default: TYPEHINT_MODULE = DEFAULT_MODULE, **kwargs):
    a = AttrHandler("exp")
    module = get_module_from_object(x, default=default)
    x = cast_to_dtype(x, module=module)
    return get_module_attr(module, a[module])(x, **kwargs)


def log(x: TYPEHINT_DTYPE, default: TYPEHINT_MODULE = DEFAULT_MODULE, **kwargs):
    a = AttrHandler("log", tensorflow="math.log")
    module = get_module_from_object(x, default=default)
    x = cast_to_dtype(x, module=module)
    return get_module_attr(module, a[module])(x, **kwargs)
