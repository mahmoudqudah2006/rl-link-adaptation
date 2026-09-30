import numpy as np

from rl_link.agent import QLearningAgent


def test_q_learning_update_changes_value() -> None:
    agent = QLearningAgent(np.array([0.0, 10.0]), 2, epsilon=0.0)
    agent.update(5.0, 1, 2.0, 6.0)
    assert agent.q[1, 1] > 0
