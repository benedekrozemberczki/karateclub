from karateclub import DeepWalk

def test_get_params():
    model = DeepWalk()
    params = model.get_params()
    assert len(params) != 0
    assert type(params) is dict
    assert '_embedding' not in params