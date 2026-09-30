from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class MCS:
    name: str
    efficiency: float
    threshold_db: float


MCS_TABLE = (
    MCS("robust", 1.0, 0.0),
    MCS("qpsk", 2.0, 6.0),
    MCS("16qam", 4.0, 12.0),
    MCS("64qam", 6.0, 18.0),
)


def bler_probability(snr_db: float, mcs: MCS, slope: float = 1.6) -> float:
    value = 1.0 / (1.0 + np.exp((snr_db - mcs.threshold_db) / slope))
    return float(np.clip(value, 0.0, 1.0))


class LinkAdaptationEnv:
    """Small Gymnasium-style environment without a mandatory Gymnasium dependency."""

    def __init__(
        self,
        *,
        mean_snr_db: float = 12.0,
        snr_std_db: float = 3.0,
        correlation: float = 0.92,
        target_bler: float = 0.1,
        reliability_penalty: float = 8.0,
        max_steps: int = 200,
        seed: int | None = None,
    ) -> None:
        if not 0 <= correlation < 1:
            raise ValueError("correlation must satisfy 0 <= correlation < 1")
        self.mean_snr_db = mean_snr_db
        self.snr_std_db = snr_std_db
        self.correlation = correlation
        self.target_bler = target_bler
        self.reliability_penalty = reliability_penalty
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.snr_db = mean_snr_db
        self.previous_action = 0
        self.steps = 0

    @property
    def n_actions(self) -> int:
        return len(MCS_TABLE)

    def _observation(self) -> np.ndarray:
        return np.array([self.snr_db, float(self.previous_action)], dtype=np.float32)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, float]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.snr_db = float(self.rng.normal(self.mean_snr_db, self.snr_std_db))
        self.previous_action = 0
        self.steps = 0
        return self._observation(), {"snr_db": self.snr_db}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, float | str]]:
        if not 0 <= action < self.n_actions:
            raise ValueError("invalid action")
        mcs = MCS_TABLE[action]
        bler = bler_probability(self.snr_db, mcs)
        goodput = mcs.efficiency * (1.0 - bler)
        penalty = self.reliability_penalty * max(bler - self.target_bler, 0.0)
        reward = goodput - penalty
        info: dict[str, float | str] = {
            "snr_db": self.snr_db,
            "mcs": mcs.name,
            "bler": bler,
            "goodput": goodput,
        }

        noise_scale = self.snr_std_db * np.sqrt(1.0 - self.correlation**2)
        self.snr_db = float(
            self.correlation * self.snr_db
            + (1.0 - self.correlation) * self.mean_snr_db
            + self.rng.normal(0.0, noise_scale)
        )
        self.snr_db = float(np.clip(self.snr_db, -10.0, 35.0))
        self.previous_action = action
        self.steps += 1
        truncated = self.steps >= self.max_steps
        return self._observation(), float(reward), False, truncated, info
