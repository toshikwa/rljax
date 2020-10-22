import numpy as np
from haiku import PRNGSequence

from rljax.util.preprocess import add_noise, get_q_at_action, get_quantile_at_action, preprocess_state


def test_add_noise():
    rng = PRNGSequence(0)
    assert np.isclose(add_noise(0.0, next(rng), 0.0), 0.0)
    assert np.isclose(add_noise(0.0, next(rng), 0.0, -1.0, 1.0), 0.0)
    assert np.isclose(add_noise(0.0, next(rng), 100.0, -1.0, 1.0, 0.0, 0.0), 0.0)
    assert -1.0 <= add_noise(0.0, next(rng), 100.0, -1.0, 1.0) <= 1.0
    assert -20.0 <= add_noise(0.0, next(rng), 100.0, -20.0, 20.0) <= 20.0


def test_preprocess_state():
    rng = PRNGSequence(0)
    state = np.random.randint(0, 256, size=(64, 64, 3)).astype(np.uint8)
    state = preprocess_state(state, next(rng))
    assert (-0.5 <= state).all() and (state <= 0.5).all()


def test_get_q_at_action():
    q_s = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.isclose(get_q_at_action(q_s, np.array([[0], [0]])), [[1.0], [3.0]]).all()
    assert np.isclose(get_q_at_action(q_s, np.array([[0], [1]])), [[1.0], [4.0]]).all()
    assert np.isclose(get_q_at_action(q_s, np.array([[1], [0]])), [[2.0], [3.0]]).all()
    assert np.isclose(get_q_at_action(q_s, np.array([[1], [1]])), [[2.0], [4.0]]).all()


def test_get_quantile_at_action():
    quantile_s = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    assert np.isclose(get_quantile_at_action(quantile_s, np.array([[0]])), [[1.0], [3.0]]).all()
    assert np.isclose(get_quantile_at_action(quantile_s, np.array([[1]])), [[2.0], [4.0]]).all()
