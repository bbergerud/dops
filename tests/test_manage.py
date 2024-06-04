import torch

from src.manage import (
    cast_to_dtype,
    get_module,
    get_module_attr,
    get_module_from_object,
    get_module_type,
    is_module_dtype,
    is_module_installed,
)


def test_is_module_installed():
    assert is_module_installed("torch")


def test_get_module():
    module = get_module("torch")
    assert module == torch


def test_get_module_type():
    dtype = get_module_type("torch")
    assert dtype == torch.Tensor


def test_get_module_from_object():
    object = torch.tensor([1.0, 2.0, 3.0])
    assert get_module_from_object(object, "numpy") == "torch"


def test_get_module_attr():
    func = get_module_attr("torch", "exp")
    assert func == torch.exp


def test_is_module_dtype():
    data = torch.Tensor([1.0, 2.0, 3.0])
    assert is_module_dtype(data, "torch")


def test_cast_to_type():
    data = [1.0, 2.0, 3.0]
    assert (
        torch.tensor(data, dtype=torch.float32)
        == cast_to_dtype(object=data, module="torch", dtype=torch.float32)
    ).all()
