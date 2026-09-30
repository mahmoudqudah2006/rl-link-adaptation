from __future__ import annotations

import numpy as np


class QLearningAgent:
    def __init__(
        self,
        snr_bins: np.ndarray,
        n_actions: int,
        *,
        alpha: float = 0.15,
        gamma: float = 0.95,
        epsilon: float = 0.15,
        seed: int = 7,
    ) -> None:
        self.snr_bins = np.asarray(snr_bins, dtype=float)
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros((len(self.snr_bins) + 1, n_actions), dtype=float)

    def state_index(self, snr_db: float) -> int:
        return int(np.digitize(snr_db, self.snr_bins))

    def act(self, snr_db: float, explore: bool = True) -> int:
        state = self.state_index(snr_db)
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.q[state]))

    def update(self, snr_db: float, action: int, reward: float, next_snr_db: float) -> None:
        state = self.state_index(snr_db)
        next_state = self.state_index(next_snr_db)
        target = reward + self.gamma * float(np.max(self.q[next_state]))
        self.q[state, action] += self.alpha * (target - self.q[state, action])
