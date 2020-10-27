import numpy as np

from rljax.util.saving import load_params, save_params


def test_saving():
    params = {"w": np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32)}
    save_params(params, "/tmp/rljax/test/pamars.npz")
    assert np.isclose(load_params("/tmp/rljax/test/pamars.npz")["w"], params["w"]).all()
