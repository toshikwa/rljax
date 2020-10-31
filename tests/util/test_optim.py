import haiku as hk
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import optix

from rljax.util.optim import clip_gradient, clip_gradient_norm, optimize, soft_update, weight_decay


@pytest.mark.parametrize("lr, w, x", [(0.1, 1.0, 1.0), (0.1, 20.0, 10.0), (1e-3, 0.0, -10.0)])
def test_optimize(lr, w, x):
    net = hk.without_apply_rng(hk.transform(lambda x: hk.Linear(1, with_bias=False, w_init=hk.initializers.Constant(w))(x)))
    params = net.init(next(hk.PRNGSequence(0)), jnp.zeros((1, 1)))
    opt_init, opt = optix.sgd(lr)
    opt_state = opt_init(params)

    def _loss(params, x):
        return net.apply(params, x).mean(), None

    opt_state, params, loss, _ = optimize(_loss, opt, opt_state, params, None, x=jnp.ones((1, 1)) * x)
    assert np.isclose(loss, w * x)
    assert np.isclose(params["linear"]["w"], w - lr * x)


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
