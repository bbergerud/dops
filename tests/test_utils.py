from tenops.utils import AttrHandler, ModuleHandler, ParameterHandler


def test_AttrHandler():
    a = AttrHandler("asin", numpy="arcsin")
    assert a["torch"] == "asin"
    assert a["numpy"] == "arcsin"


def test_ModuleHandler():
    m = ModuleHandler(test="not test")
    assert m["torch"] == "torch"
    assert m["test"] == "not test"


def test_ParameterHandler():
    p = ParameterHandler(params=dict(axis=1), axis=dict(torch="dim"))
    assert p["numpy"] == {"axis": 1}
    assert p["torch"] == {"dim": 1}
