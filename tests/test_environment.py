from rl_link.environment import MCS_TABLE, LinkAdaptationEnv, bler_probability


def test_bler_decreases_with_snr() -> None:
    mcs = MCS_TABLE[2]
    assert bler_probability(20.0, mcs) < bler_probability(5.0, mcs)


def test_environment_is_reproducible() -> None:
    first = LinkAdaptationEnv(seed=4, max_steps=3)
    second = LinkAdaptationEnv(seed=4, max_steps=3)
    obs1, _ = first.reset(seed=9)
    obs2, _ = second.reset(seed=9)
    assert obs1.tolist() == obs2.tolist()
    step1 = first.step(1)
    step2 = second.step(1)
    assert step1[0].tolist() == step2[0].tolist()
    assert step1[1] == step2[1]
