import numpy as np
import pytest

from rljax.util.loss import huber, quantile_loss


def test_huber():
    assert np.isclose(huber(0.0), 0.0)
    assert np.isclose(huber(-0.5), 0.25)
    assert np.isclose(huber(0.5), 0.25)
    assert np.isclose(huber(-1.0), 1.0)
    assert np.isclose(huber(1.0), 1.0)
    assert np.isclose(huber(-2.0), 2.0)
    assert np.isclose(huber(2.0), 2.0)


@pytest.mark.parametrize("loss_type", [("l2"), ("huber")])
def test_quantile_loss(loss_type):
    cum_p = np.arange(1, 11, dtype=np.float32).reshape([1, 10]) / 10.0
    td = np.ones((1, 10, 10), dtype=np.float32)

    assert np.isclose(quantile_loss(td, cum_p, 1.0, "l2"), 5.5)
    assert np.isclose(quantile_loss(td, cum_p, 1.0, "huber"), 5.5)
    assert np.isclose(quantile_loss(-td, cum_p, 1.0, "l2"), 4.5)
    assert np.isclose(quantile_loss(-td, cum_p, 1.0, "huber"), 4.5)
    assert np.isclose(quantile_loss(2.0 * td, cum_p, 1.0, "l2"), 22.0)
    assert np.isclose(quantile_loss(2.0 * td, cum_p, 1.0, "huber"), 11.0)
    assert np.isclose(quantile_loss(0.5 * td, cum_p, 1.0, "l2"), 1.375)
    assert np.isclose(quantile_loss(0.5 * td, cum_p, 1.0, "huber"), 1.375)
