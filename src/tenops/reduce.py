from .manage import (
    DEFAULT_MODULE,
    TYPEHINT_DTYPE,
    TYPEHINT_MODULE,
    cast_to_dtype,
    get_module_attr,
    get_module_from_object,
)
from .utils import AttrHandler


def all(
    x: TYPEHINT_DTYPE, default: TYPEHINT_MODULE = DEFAULT_MODULE, **kwargs
) -> TYPEHINT_DTYPE:
    a = AttrHandler("all", tensorflow="reduce_all")
    module = get_module_from_object(x, default=default)
    x = cast_to_dtype(x, module)
    return get_module_attr(module, a[module])(x, **kwargs)


# Aliases
reduce_all = all
