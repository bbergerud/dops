from collections import defaultdict

from .manage import (
    DEFAULT_MODULE,
    TYPEHINT_DTYPE,
    TYPEHINT_MODULE,
    cast_to_dtype,
    get_module_attr,
    get_module_from_object,
)


def exp(x: TYPEHINT_DTYPE, default: TYPEHINT_MODULE = DEFAULT_MODULE, **kwargs):
    d = defaultdict(lambda: "exp")
    module = get_module_from_object(x, default=default)
    x = cast_to_dtype(x, module=module)
    return get_module_attr(module, d[module])(x, **kwargs)


def log(x: TYPEHINT_DTYPE, default: TYPEHINT_MODULE = DEFAULT_MODULE, **kwargs):
    d = defaultdict(lambda: "log")
    module = get_module_from_object(x, default=default)
    x = cast_to_dtype(x, module=module)
    return get_module_attr(module, d[module])(x, **kwargs)
