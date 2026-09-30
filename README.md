# RL Link Adaptation

A compact reinforcement-learning environment for **adaptive modulation and coding research** under a time-varying wireless channel.

The first release includes a Gymnasium-style standalone environment and a tabular Q-learning baseline. Keeping the core environment dependency-light makes the reward model and channel dynamics easy to inspect before adding larger RL frameworks.

## Problem formulation

At each time step the agent observes the current SNR and previous MCS, then selects one of four abstract MCS profiles:

| Action | Spectral efficiency | Approx. SNR operating point |
|---:|---:|---:|
| 0 | 1.0 bit/s/Hz | 0 dB |
| 1 | 2.0 bit/s/Hz | 6 dB |
| 2 | 4.0 bit/s/Hz | 12 dB |
| 3 | 6.0 bit/s/Hz | 18 dB |

The packet/block error probability is represented by a smooth logistic curve around each operating point. These are **research-model parameters**, not standardized BLER curves.

The reward balances useful spectral efficiency and reliability:

[
r = \eta(1-\mathrm{BLER}) - \lambda\max(\mathrm{BLER}-B_{target}, 0)
]

where (eta) is spectral efficiency.

## Channel dynamics

SNR follows a bounded first-order stochastic process:

[
\gamma_{t+1}=\rho\gamma_t+(1-\rho)\mu+\epsilon_t.
]

This creates temporal correlation without pretending to replace a full fading simulator.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

rl-link train --episodes 1500 --seed 7 --output results/q_learning.json
```

## Environment API

```python
from rl_link.environment import LinkAdaptationEnv

env = LinkAdaptationEnv(seed=7)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(2)
```

The signature intentionally mirrors modern Gymnasium environment conventions, so wrapping it in `gymnasium.Env` later is straightforward.

## Baselines

- threshold policy
- tabular Q-learning

The repository stores learning curves and evaluation summaries, making it possible to compare RL against a transparent engineering heuristic rather than only against itself.

## Roadmap

- [ ] DQN/PPO baselines
- [ ] delayed/partial CSI
- [ ] queue state and latency-aware reward
- [ ] HARQ/retransmissions
- [ ] multi-user scheduling
- [ ] integration with OFDM simulator
- [ ] measured channel traces

## License

MIT

---

**Mahmoud Alqudah** · Reinforcement Learning · Wireless Link Adaptation · AI for Communications
