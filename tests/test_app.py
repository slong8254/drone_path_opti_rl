import os
import tempfile

import numpy as np

from app import GridEnvConfig, GridEnv, RLConfig, TabularAgent


def test_shapes_and_encoding():
    cfg = GridEnvConfig(width=8, height=8, n_targets=2, obstacle_ratio=0.0)
    env = GridEnv(cfg, seed=123)
    agent = TabularAgent(env, RLConfig(batch_size=100, max_global_steps=1000, conv_min_batches=1), seed=123)

    SZ = env.W * env.H
    MASKS = 1 << len(env.targets)
    assert agent.num_states == SZ * MASKS
    assert agent.V.size == agent.num_states
    assert agent.N.size == agent.num_states


def test_train_one_phase_runs():
    cfg = GridEnvConfig(width=8, height=8, n_targets=2, obstacle_ratio=0.05, max_steps=100)
    env = GridEnv(cfg, seed=7)
    rl = RLConfig(
        epsilon_schedule=[0.5, 0.1],
        batch_size=200,
        max_phases=2,
        max_global_steps=2000,
        conv_min_batches=2,
    )
    agent = TabularAgent(env, rl, seed=7)

    start_steps = agent.global_steps
    conv, steps_used, batches = agent.train_one_phase(eps=0.5, max_global_steps=rl.max_global_steps)
    assert steps_used >= 0
    assert batches >= 1
    assert agent.global_steps >= start_steps


def test_evaluate_episode_runs():
    cfg = GridEnvConfig(width=10, height=10, n_targets=1, obstacle_ratio=0.0, max_steps=50)
    env = GridEnv(cfg, seed=42)
    agent = TabularAgent(env, RLConfig(batch_size=50, max_global_steps=200), seed=42)
    total_r, trail, term, trunc = agent.evaluate_episode(eps=0.1)
    assert isinstance(total_r, float)
    assert isinstance(trail, list) and len(trail) >= 1
    assert isinstance(term, (bool, np.bool_))
    assert isinstance(trunc, (bool, np.bool_))


def test_save_and_load_mismatch_raises():
    cfg = GridEnvConfig(width=8, height=8, n_targets=1, obstacle_ratio=0.0)
    env = GridEnv(cfg, seed=11)
    agent = TabularAgent(env, RLConfig(), seed=11)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "model.npz")
        agent.save(path)

        # Create env with different dims
        cfg2 = GridEnvConfig(width=10, height=8, n_targets=1, obstacle_ratio=0.0)
        env2 = GridEnv(cfg2, seed=11)
        agent2 = TabularAgent(env2, RLConfig(), seed=11)
        try:
            agent2.load(path)
            raised = False
        except ValueError:
            raised = True
        assert raised, "Expected ValueError due to dimension mismatch"

