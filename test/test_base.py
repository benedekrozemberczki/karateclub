from karateclub import DeepWalk

def test_get_params():
    model = DeepWalk()
    params = model.get_params()
    assert len(params) != 0
    assert type(params) is dict
    assert '_embedding' not in params

def test_set_params():
    model = DeepWalk()
    default_params = model.get_params()
    params = {'dimensions': 1,
              'seed': 123}
    model.set_params(**params)
    new_params = model.get_params()
    assert new_params != default_params
    assert new_params['dimensions'] == 1
    assert new_params['seed'] == 123
