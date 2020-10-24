import numpy as np

from rljax.util.optim import clip_gradient, clip_gradient_norm, soft_update, weight_decay


def test_clip_gradient():
    grad = {"w": np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)}

    assert np.isclose(clip_gradient(grad, 2.0)["w"], [-2.0, -1.0, 0.0, 1.0, 2.0]).all()
    assert np.isclose(clip_gradient(grad, 1.5)["w"], [-1.5, -1.0, 0.0, 1.0, 1.5]).all()
    assert np.isclose(clip_gradient(grad, 1.0)["w"], [-1.0, -1.0, 0.0, 1.0, 1.0]).all()
    assert np.isclose(clip_gradient(grad, 0.5)["w"], [-0.5, -0.5, 0.0, 0.5, 0.5]).all()


def test_clip_gradient_norm():
    grad = {"w": np.array([1.0, 0.0], dtype=np.float32)}

    assert np.isclose(clip_gradient_norm(grad, 0.0)["w"], [0.0, 0.0]).all()
    assert np.isclose(clip_gradient_norm(grad, 0.5)["w"], [0.5, 0.0]).all()
    assert np.isclose(clip_gradient_norm(grad, 1.0)["w"], [1.0, 0.0]).all()
    assert np.isclose(clip_gradient_norm(grad, 2.0)["w"], [1.0, 0.0]).all()

    grad = {"w": np.array([3.0, 4.0], dtype=np.float32)}

    assert np.isclose(clip_gradient_norm(grad, 0.0)["w"], [0.0, 0.0]).all()
    assert np.isclose(clip_gradient_norm(grad, 1.0)["w"], [0.6, 0.8]).all()
    assert np.isclose(clip_gradient_norm(grad, 2.0)["w"], [1.2, 1.6]).all()
    assert np.isclose(clip_gradient_norm(grad, 5.0)["w"], [3.0, 4.0]).all()
    assert np.isclose(clip_gradient_norm(grad, 10.0)["w"], [3.0, 4.0]).all()


def test_soft_update():
    source = {"w": np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32)}
    target = {"w": np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)}

    assert np.isclose(soft_update(target, source, 0.0)["w"], target["w"]).all()
    assert np.isclose(soft_update(target, source, 1.0)["w"], source["w"]).all()
    assert np.isclose(soft_update(target, source, 0.5)["w"], [-6.0, -3.0, 0.0, 3.0, 6.0]).all()


def test_weight_decay():
    assert np.isclose(weight_decay({"w": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)}), 0.0)
    assert np.isclose(weight_decay({"w": np.array([-1.0, -1.0, 0.0, 1.0, 1.0], dtype=np.float32)}), 2.0)
    assert np.isclose(weight_decay({"w": np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)}), 5.0)
