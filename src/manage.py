import importlib
from types import ModuleType
from typing import Any, Union

module_type = {
    "numpy": lambda: get_module_attr("numpy", "ndarray"),
    "torch": lambda: get_module_attr("torch", "Tensor"),
}

module_typecast = {
    "numpy": lambda: get_module_attr("numpy", "asarray"),
    "torch": lambda: get_module_attr("torch", "as_tensor"),
}


def cast_to_dtype(object: Any, module: str, **kwargs):
    """Casts the object to the basic data type associated with the module"""
    if not module in MODULES:
        raise ValueError(f"{module} is not a valid module")
    return module_typecast[module]()(object, **kwargs)


def get_module_type(module: str) -> Any:
    if not module in MODULES:
        raise Exception(f"{module} not a valid module")
    return module_type[module]()


def get_module_from_object(object: Any, default: str) -> str:
    for module in MODULES:
        if is_module_dtype(object, module):
            return module
    return default


def get_module_attr(module: str, attr: str) -> Any:
    """Returns the attr associated with the module"""
    return getattr(get_module(module), attr)


def get_module(module: str) -> ModuleType:
    """
    Returns the module object if the module is installed,
    otherwise raises an ModuleNotFoundError.
    """
    if is_module_installed(module):
        return importlib.import_module(module)
    else:
        raise ModuleNotFoundError(f"Module `{module}` was not found")


def is_module_installed(module: str) -> bool:
    """Checks if a module is installed."""
    return importlib.util.find_spec(module) is not None


def is_module_dtype(object: Any, module: str) -> bool:
    """Determines if the object is an instance of the module"""
    if not module in MODULES:
        raise Exception(f"{module} not a valid module")
    return isinstance(object, get_module_type(module))


DEFAULT_MODULE = None
MODULES = []
for module, dtype in module_type.items():
    if is_module_installed(module):
        if DEFAULT_MODULE is None:
            DEFAULT_MODULE = module
        importlib.import_module(module)
        MODULES.append(module)

TYPEHINT_DTYPE = Union[tuple(get_module_type(module) for module in MODULES)]
TYPEHINT_MODULE = Union[tuple(MODULES)]
