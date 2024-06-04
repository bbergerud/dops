from collections import defaultdict

from .manage import (
    DEFAULT_MODULE,
    TYPEHINT_DTYPE,
    TYPEHINT_MODULE,
    get_module_attr,
    get_module_from_object,
    get_module_from_objects,
)


def angle(
    x: TYPEHINT_DTYPE, default: TYPEHINT_MODULE = DEFAULT_MODULE, **kwargs
) -> TYPEHINT_DTYPE:
    d = defaultdict(lambda: "angle")
    module = get_module_from_object(x, default=default)
    return get_module_attr(module, d[module])(x, **kwargs)


def atan(
    x: TYPEHINT_DTYPE, default: TYPEHINT_MODULE = DEFAULT_MODULE, **kwargs
) -> TYPEHINT_DTYPE:
    d = defaultdict(lambda: "arctan")
    module = get_module_from_object(x, default=default)
    return get_module_attr(module, d[module])(x, **kwargs)


def atan2(
    y: TYPEHINT_DTYPE,
    x: TYPEHINT_DTYPE,
    default: TYPEHINT_MODULE = DEFAULT_MODULE,
    **kwargs
) -> TYPEHINT_DTYPE:
    d = defaultdict(lambda: "arctan2")
    module = get_module_from_objects([x, y], default=default)
    return get_module_attr(module, d[module])(y, x, **kwargs)


def cos(
    x: TYPEHINT_DTYPE, default: TYPEHINT_MODULE = DEFAULT_MODULE, **kwargs
) -> TYPEHINT_DTYPE:
    d = defaultdict(lambda: "cos")
    module = get_module_from_object(x, default=default)
    return get_module_attr(module, d[module])(x, **kwargs)


def sin(
    x: TYPEHINT_DTYPE, default: TYPEHINT_MODULE = DEFAULT_MODULE, **kwargs
) -> TYPEHINT_DTYPE:
    d = defaultdict(lambda: "sin")
    module = get_module_from_object(x, default=default)
    return get_module_attr(module, d[module])(x, **kwargs)


def tan(
    x: TYPEHINT_DTYPE, default: TYPEHINT_MODULE = DEFAULT_MODULE, **kwargs
) -> TYPEHINT_DTYPE:
    d = defaultdict(lambda: "tan")
    module = get_module_from_object(x, default=default)
    return get_module_attr(module, d[module])(x, **kwargs)


# Aliases
arctan = atan
arctan2 = atan2