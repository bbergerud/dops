from .manage import (
    DEFAULT_MODULE,
    TYPEHINT_DTYPE,
    TYPEHINT_MODULE,
    cast_to_dtype,
    get_module_attr,
    get_module_from_object,
    get_module_from_objects,
)
from .utils import AttrHandler


def angle(
    x: TYPEHINT_DTYPE, default: TYPEHINT_MODULE = DEFAULT_MODULE, **kwargs
) -> TYPEHINT_DTYPE:
    a = AttrHandler("angle", tensorflow="math.angle")
    module = get_module_from_object(x, default=default)
    x = cast_to_dtype(x, module=module)
    return get_module_attr(module, a[module])(x, **kwargs)


def acos(
    x: TYPEHINT_DTYPE, default: TYPEHINT_MODULE = DEFAULT_MODULE, **kwargs
) -> TYPEHINT_DTYPE:
    a = AttrHandler("acos", numpy="arccos")
    module = get_module_from_object(x, default=default)
    x = cast_to_dtype(x, module=module)
    return get_module_attr(module, a[module])(x, **kwargs)


def acosh(x: TYPEHINT_DTYPE, default: TYPEHINT_MODULE = DEFAULT_MODULE, **kwargs):
    a = AttrHandler("acosh", numpy="arccosh")
    module = get_module_from_object(x, default=default)
    x = cast_to_dtype(x, module=module)
    return get_module_attr(module, a[module])(x, **kwargs)


def asin(x: TYPEHINT_DTYPE, default: TYPEHINT_MODULE = DEFAULT_MODULE, **kwargs):
    a = AttrHandler("asin", numpy="arcsin")
    module = get_module_from_object(x, default=default)
    x = cast_to_dtype(x, module=module)
    return get_module_attr(module, a[module])(x, **kwargs)


def asinh(x: TYPEHINT_DTYPE, default: TYPEHINT_MODULE = DEFAULT_MODULE, **kwargs):
    a = AttrHandler("asinh", numpy="arcsinh")
    module = get_module_from_object(x, default=default)
    x = cast_to_dtype(x, module=module)
    return get_module_attr(module, a[module])(x, **kwargs)


def atan(
    x: TYPEHINT_DTYPE, default: TYPEHINT_MODULE = DEFAULT_MODULE, **kwargs
) -> TYPEHINT_DTYPE:
    a = AttrHandler("atan", numpy="arctan")
    module = get_module_from_object(x, default=default)
    x = cast_to_dtype(x, module=module)
    return get_module_attr(module, a[module])(x, **kwargs)


def atan2(
    y: TYPEHINT_DTYPE,
    x: TYPEHINT_DTYPE,
    default: TYPEHINT_MODULE = DEFAULT_MODULE,
    **kwargs
) -> TYPEHINT_DTYPE:
    a = AttrHandler("atan2", numpy="arctan2")
    module = get_module_from_objects([x, y], default=default)
    x = cast_to_dtype(x, module=module)
    y = cast_to_dtype(y, module=module)
    return get_module_attr(module, a[module])(y, x, **kwargs)


def atanh(
    x: TYPEHINT_DTYPE, default: TYPEHINT_MODULE = DEFAULT_MODULE, **kwargs
) -> TYPEHINT_DTYPE:
    a = AttrHandler("atanh", numpy="arctanh")
    module = get_module_from_object(x, default=default)
    x = cast_to_dtype(x, module=module)
    return get_module_attr(module, a[module])(x, **kwargs)


def cos(
    x: TYPEHINT_DTYPE, default: TYPEHINT_MODULE = DEFAULT_MODULE, **kwargs
) -> TYPEHINT_DTYPE:
    a = AttrHandler("cos")
    module = get_module_from_object(x, default=default)
    x = cast_to_dtype(x, module=module)
    return get_module_attr(module, a[module])(x, **kwargs)


def sin(
    x: TYPEHINT_DTYPE, default: TYPEHINT_MODULE = DEFAULT_MODULE, **kwargs
) -> TYPEHINT_DTYPE:
    a = AttrHandler("sin")
    module = get_module_from_object(x, default=default)
    x = cast_to_dtype(x, module=module)
    return get_module_attr(module, a[module])(x, **kwargs)


def tan(
    x: TYPEHINT_DTYPE, default: TYPEHINT_MODULE = DEFAULT_MODULE, **kwargs
) -> TYPEHINT_DTYPE:
    a = AttrHandler("tan")
    module = get_module_from_object(x, default=default)
    x = cast_to_dtype(x, module=module)
    return get_module_attr(module, a[module])(x, **kwargs)


# Aliases
arccos = acos
arccosh = acosh
arcsin = asin
arcsinh = asinh
arctan = atan
arctan = atan
arctan2 = atan2
