from dops.utils import ParameterAlias


def test_Parameter():
    p = ParameterAlias(params=dict(axis=1), axis=dict(torch="dim"))
    assert p["torch"] == {"dim": 1}
