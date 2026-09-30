from __future__ import annotations

from typing import Any

import numpy as np

from .agent import QLearningAgent
from .environment import LinkAdaptationEnv


def threshold_action(snr_db: float) -> int:
    if snr_db < 4.0:
        return 0
    if snr_db < 10.0:
        return 1
    if snr_db < 17.0:
        return 2
    return 3


def train_q_learning(episodes: int = 1000, seed: int = 7) -> tuple[QLearningAgent, dict[str, Any]]:
    env = LinkAdaptationEnv(seed=seed)
    agent = QLearningAgent(np.linspace(-5, 30, 36), env.n_actions, seed=seed)
    episode_returns: list[float] = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        total = 0.0
        while True:
            snr = float(obs[0])
            action = agent.act(snr, explore=True)
            next_obs, reward, _, truncated, _ = env.step(action)
            agent.update(snr, action, reward, float(next_obs[0]))
            total += reward
            obs = next_obs
            if truncated:
                break
        episode_returns.append(total)

    eval_env = LinkAdaptationEnv(seed=seed + 100_000)
    obs, _ = eval_env.reset()
    rewards: list[float] = []
    while True:
        action = agent.act(float(obs[0]), explore=False)
        obs, reward, _, truncated, _ = eval_env.step(action)
        rewards.append(reward)
        if truncated:
            break

    baseline_env = LinkAdaptationEnv(seed=seed + 100_000)
    baseline_obs, _ = baseline_env.reset()
    baseline_rewards: list[float] = []
    while True:
        baseline_action = threshold_action(float(baseline_obs[0]))
        baseline_obs, baseline_reward, _, baseline_truncated, _ = baseline_env.step(baseline_action)
        baseline_rewards.append(baseline_reward)
        if baseline_truncated:
            break

    return agent, {
        "episodes": episodes,
        "mean_training_return_last_100": float(np.mean(episode_returns[-100:])),
        "evaluation_mean_reward": float(np.mean(rewards)),
        "threshold_baseline_mean_reward": float(np.mean(baseline_rewards)),
        "q_table": agent.q.tolist(),
    }
