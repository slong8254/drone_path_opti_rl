"""
Streamlit-based Tabular RL for Single Surveillance Drone (No Gym)

Key features
- Configurable grid (default 30x30), obstacles, targets
- Tabular value learning with epsilon-phased schedule
- Automatic convergence check via ΔV average/std thresholds
- Real-time episode visualization via Streamlit

Run
  pip install streamlit==1.36.0 numpy matplotlib
  streamlit run app.py

Notes
- UI is under a main guard so the module can be safely imported for tests.
- Action selection now correctly accounts for target-bit updates on the next
  state when evaluating candidate actions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

# Use a non-interactive matplotlib backend (safe in headless/CI environments)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int) -> None:
    np.random.seed(seed)


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _sanitize_pos(pos: Tuple[int, int], h: int, w: int) -> Tuple[int, int]:
    y = clamp(int(pos[0]), 0, h - 1)
    x = clamp(int(pos[1]), 0, w - 1)
    return (y, x)


def compute_optimal_path_length(env: "GridEnv") -> int | None:
    """Compute minimal steps to visit all targets (any order) and reach goal.

    - Uses BFS to compute grid shortest paths between all POIs: start, targets, goal.
    - Then solves a small TSP-like DP over target order to minimize total steps.
    - Returns None if unreachable.
    """
    from collections import deque

    H, W = env.H, env.W
    grid = env.grid
    start = env.start
    targets = list(env.targets)
    goal = env.goal

    pois = [start] + targets + [goal]
    n = len(pois)
    INF = 10**9

    def bfs(src: Tuple[int, int]) -> np.ndarray:
        dist = np.full((H, W), INF, dtype=np.int32)
        dq = deque()
        sy, sx = src
        dist[sy, sx] = 0
        dq.append((sy, sx))
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)] if not env.moves8 else [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]
        while dq:
            y, x = dq.popleft()
            for dy, dx in moves:
                ny = y + dy
                nx = x + dx
                if 0 <= ny < H and 0 <= nx < W and grid[ny, nx] == 0:
                    nd = dist[y, x] + 1
                    if nd < dist[ny, nx]:
                        dist[ny, nx] = nd
                        dq.append((ny, nx))
        return dist

    # pairwise shortest path distances
    dmat = np.full((n, n), INF, dtype=np.int32)
    dists_from = [bfs(p) for p in pois]
    for i, (py, px) in enumerate(pois):
        di = dists_from[i]
        for j, (qy, qx) in enumerate(pois):
            dmat[i, j] = int(di[qy, qx])

    # If there are no targets, just return start->goal
    T = len(targets)
    if T == 0:
        d = dmat[0, 1]
        return None if d >= INF else int(d)

    # DP over targets: index mapping
    # indices: 0=start, 1..T=targets, T+1=goal
    start_idx = 0
    goal_idx = T + 1
    full_mask = (1 << T) - 1

    # dp[mask][i] = min distance to visit 'mask' targets and end at target i (1..T)
    dp = np.full((1 << T, T), INF, dtype=np.int32)
    # init from start to each target
    for i in range(T):
        d = dmat[start_idx, 1 + i]
        if d < INF:
            dp[1 << i, i] = d

    for mask in range(1 << T):
        for i in range(T):
            if not (mask & (1 << i)):
                continue
            cur = dp[mask, i]
            if cur >= INF:
                continue
            # try going to next target j
            for j in range(T):
                if mask & (1 << j):
                    continue
                d = dmat[1 + i, 1 + j]
                if d < INF:
                    nm = mask | (1 << j)
                    nd = cur + d
                    if nd < dp[nm, j]:
                        dp[nm, j] = nd

    # complete to goal
    best = INF
    for i in range(T):
        to_goal = dmat[1 + i, goal_idx]
        if to_goal < INF and dp[full_mask, i] < INF:
            best = min(best, int(dp[full_mask, i]) + int(to_goal))

    return None if best >= INF else int(best)


# ----------------------------
# Environment (No Gym)
# ----------------------------
@dataclass
class GridEnvConfig:
    width: int = 30
    height: int = 30
    moves8: bool = True
    max_steps: int = 400
    n_targets: int = 4
    obstacle_ratio: float = 0.10
    start: Tuple[int, int] = (0, 0)
    goal: Tuple[int, int] = (29, 29)
    R_target: float = 5.0
    R_goal: float = 10.0
    lambda_step: float = 0.1


class GridEnv:
    def __init__(self, cfg: GridEnvConfig, seed: int = 42):
        self.cfg = cfg
        set_seed(seed)
        self.W = cfg.width
        self.H = cfg.height
        self.moves8 = cfg.moves8
        self.max_steps = cfg.max_steps

        # Build grid: 0 empty, 1 obstacle
        self.grid = np.zeros((self.H, self.W), dtype=np.uint8)

        # Place obstacles (we'll finalize after we pick targets to avoid collisions)
        if cfg.obstacle_ratio > 0:
            n_obs = int(self.W * self.H * cfg.obstacle_ratio)
            obs_positions = set()
            # Sample with possible duplicates; set() dedups
            while len(obs_positions) < n_obs:
                y = np.random.randint(0, self.H)
                x = np.random.randint(0, self.W)
                obs_positions.add((y, x))
            self.obs_positions = obs_positions
        else:
            self.obs_positions = set()

        # Place targets (distinct positions not equal to start/goal)
        # Ensure start/goal are in-bounds
        self.start = _sanitize_pos(cfg.start, self.H, self.W)
        self.goal = _sanitize_pos(cfg.goal, self.H, self.W)
        self.targets = self._sample_targets(cfg.n_targets)

        # Finalize obstacle grid (not on start/goal/targets)
        for (y, x) in self.obs_positions:
            if (y, x) != self.start and (y, x) != self.goal and (y, x) not in self.targets:
                self.grid[y, x] = 1

        # Precompute action set
        if self.moves8:
            self.actions = [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]
        else:
            self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # State variables
        self.reset()

    def _sample_targets(self, n: int) -> List[Tuple[int, int]]:
        # Sample unique target positions not colliding with start/goal
        targets = set()
        tries = 0
        max_tries = 100000
        while len(targets) < n and tries < max_tries:
            y = np.random.randint(0, self.H)
            x = np.random.randint(0, self.W)
            if (y, x) != self.start and (y, x) != self.goal:
                targets.add((y, x))
            tries += 1
        return list(targets)

    def reset(self) -> Tuple[int, int, int]:
        self.pos = [self.start[0], self.start[1]]
        self.tmask = 0  # bitmask for visited targets
        self.steps = 0
        self.done = False
        return self._obs()

    def _obs(self) -> Tuple[int, int, int]:
        return (self.pos[0], self.pos[1], self.tmask)

    def step(self, a_idx: int):
        if self.done:
            raise RuntimeError("Call reset() before stepping a finished episode.")

        dy, dx = self.actions[a_idx]
        ny = clamp(self.pos[0] + dy, 0, self.H - 1)
        nx = clamp(self.pos[1] + dx, 0, self.W - 1)

        # obstacle check: if obstacle, stay (or small penalty could be added)
        if self.grid[ny, nx] == 1:
            ny, nx = self.pos[0], self.pos[1]

        self.pos = [ny, nx]
        self.steps += 1

        reward = -self.cfg.lambda_step  # step penalty

        # Target check
        for i, (ty, tx) in enumerate(self.targets):
            bit = 1 << i
            if ny == ty and nx == tx and (self.tmask & bit) == 0:
                self.tmask |= bit
                reward += self.cfg.R_target

        # Goal check
        terminated = False
        if (ny, nx) == self.goal:
            visited_count = bin(self.tmask).count("1")
            reward += self.cfg.R_goal * visited_count
            terminated = True

        truncated = False
        if self.steps >= self.max_steps:
            truncated = True

        self.done = terminated or truncated
        return self._obs(), reward, terminated, truncated, {}

    # Rendering helper (for matplotlib)
    def render_canvas(self, path: str | None = None, agent_trail=None, title: str | None = None):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(self.grid == 1, cmap="Greys", vmin=0, vmax=1)
        # Targets
        if len(self.targets) > 0:
            ty = [t[0] for t in self.targets]
            tx = [t[1] for t in self.targets]
            ax.scatter(tx, ty, marker="*", s=120, c="#FFD700", edgecolors="k", linewidths=0.5)
        # Start/Goal
        ax.scatter([self.start[1]], [self.start[0]], marker="s", s=80, c="cyan", edgecolors="k")
        ax.scatter([self.goal[1]], [self.goal[0]], marker="s", s=80, c="yellow", edgecolors="k")
        # Agent trail
        if agent_trail:
            trail_y = [p[0] for p in agent_trail]
            trail_x = [p[1] for p in agent_trail]
            ax.plot(trail_x, trail_y, linewidth=2)
            ax.scatter([trail_x[-1]], [trail_y[-1]], marker="o", s=60, c="red", edgecolors="k")
        ax.set_xlim(-0.5, self.W - 0.5)
        ax.set_ylim(self.H - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        if title:
            ax.set_title(title)
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=150)
        return fig, ax


# ----------------------------
# Tabular Agent
# ----------------------------
@dataclass
class RLConfig:
    gamma: float = 0.99
    epsilon_schedule: List[float] = field(default_factory=lambda: [0.9, 0.5, 0.1, 0.01])
    batch_size: int = 2000  # steps per batch
    max_phases: int = 4
    max_global_steps: int = 400_000
    lr_decay_k: float = 0.8  # alpha_s = exp(-k * N(s))
    conv_mode: str = "percent"  # "percent" or "abs"
    conv_avg_th: float = 0.05  # if percent: 5% of |V| mean magnitude; else absolute
    conv_std_th: float = 0.05
    conv_min_batches: int = 3


class TabularAgent:
    def __init__(self, env: GridEnv, cfg: RLConfig, seed: int = 42):
        set_seed(seed)
        self.env = env
        self.cfg = cfg
        self.gamma = cfg.gamma
        self.eps_sched = cfg.epsilon_schedule
        self.batch_size = cfg.batch_size
        self.lr_k = cfg.lr_decay_k

        self._build_state_encoding()

        self.V = np.zeros(self.num_states, dtype=np.float32)
        self.N = np.zeros(self.num_states, dtype=np.uint32)

        self.global_steps = 0
        self.phase_idx = 0
        self.delta_buffer: List[float] = []  # batch-wise ΔV stats for current phase

    def _build_state_encoding(self):
        self.SZ = self.env.W * self.env.H
        self.T = len(self.env.targets)  # target bits
        self.MASKS = 1 << self.T
        self.num_states = self.SZ * self.MASKS

    def enc_state(self, obs: Tuple[int, int, int]) -> int:
        y, x, m = obs
        p = y * self.env.W + x
        return m * self.SZ + p

    def _next_mask_if_target(self, y: int, x: int, mask: int) -> int:
        m2 = mask
        for i, (ty, tx) in enumerate(self.env.targets):
            bit = 1 << i
            if y == ty and x == tx and (m2 & bit) == 0:
                m2 |= bit
        return m2

    def act_epsilon_greedy(self, obs: Tuple[int, int, int], eps: float) -> int:
        # epsilon-random
        if np.random.rand() < eps:
            return np.random.randint(0, len(self.env.actions))

        # greedy w.r.t. V(s') with correct mask update if a target is collected
        best_a = 0
        best_v = -1e18
        y, x, m = obs
        for ai, (dy, dx) in enumerate(self.env.actions):
            ny = clamp(y + dy, 0, self.env.H - 1)
            nx = clamp(x + dx, 0, self.env.W - 1)
            # obstacle => no move
            if self.env.grid[ny, nx] == 1:
                ny, nx = y, x
            m2 = self._next_mask_if_target(ny, nx, m)
            sp = (ny, nx, m2)
            sidx = self.enc_state(sp)
            v = self.V[sidx]
            if v > best_v:
                best_v = v
                best_a = ai
        return best_a

    def _alpha(self, sidx: int) -> float:
        # alpha_s = exp(-k * N(s))
        return math.exp(-self.lr_k * float(self.N[sidx]))

    def _delta_stat_thresholds(self):
        # percent vs abs thresholds
        if self.cfg.conv_mode == "percent":
            scale = float(np.mean(np.abs(self.V)) + 1e-9)
            avg_th = self.cfg.conv_avg_th * scale
            std_th = self.cfg.conv_std_th * scale
        else:
            avg_th = self.cfg.conv_avg_th
            std_th = self.cfg.conv_std_th
        return avg_th, std_th

    def train_one_phase(
        self,
        eps: float,
        max_global_steps: int,
        viz_cb=None,
        episode_viz_cb=None,
        stop_when_optimal: bool = False,
        optimal_length: int | None = None,
    ):
        """Train until convergence or budget/time limit is reached.
        Returns (converged: bool, steps_used: int, batches: int)
        """
        self.delta_buffer = []
        batches = 0
        steps = 0
        # internal flag to signal optimal path reached if requested
        self._optimal_reached = False
        while self.global_steps < max_global_steps:
            # One batch of interactions/updates
            steps_in_batch, abs_dv_sum = 0, 0.0
            ep_idx_in_batch = 0
            while steps_in_batch < self.batch_size and self.global_steps < max_global_steps:
                obs = self.env.reset()
                done = False
                train_trail = [tuple(self.env.pos)]
                ep_steps = 0
                while not done and steps_in_batch < self.batch_size and self.global_steps < max_global_steps:
                    a = self.act_epsilon_greedy(obs, eps)
                    next_obs, r, term, trunc, _ = self.env.step(a)

                    sidx = self.enc_state(obs)
                    spidx = self.enc_state(next_obs)
                    td = r + self.gamma * self.V[spidx] - self.V[sidx]
                    alpha = self._alpha(sidx)
                    dV = alpha * td
                    self.V[sidx] += dV
                    self.N[sidx] += 1

                    abs_dv_sum += abs(float(dV))
                    obs = next_obs
                    steps_in_batch += 1
                    ep_steps += 1
                    train_trail.append(tuple(self.env.pos))
                    self.global_steps += 1
                    if term or trunc:
                        break

                # Episode ended (terminated or truncated or batch/global limit)
                if episode_viz_cb is not None and ep_steps > 0:
                    try:
                        # Compute quick metrics for the finished training episode
                        visited = set()
                        tset = set(self.env.targets)
                        for (yy, xx) in train_trail:
                            if (yy, xx) in tset:
                                visited.add((yy, xx))
                        ep_metrics = {
                            "ep_idx_in_batch": ep_idx_in_batch,
                            "ep_steps": ep_steps,
                            "batch_steps": steps_in_batch,
                            "visited_count": len(visited),
                            "terminated": bool(term),
                            "truncated": bool(trunc),
                        }
                        episode_viz_cb(ep_metrics, train_trail)
                    except Exception:
                        pass
                ep_idx_in_batch += 1

            batches += 1
            steps += steps_in_batch
            batch_abs_dv_mean = (abs_dv_sum / steps_in_batch) if steps_in_batch > 0 else 0.0
            self.delta_buffer.append(batch_abs_dv_mean)

            # Optional: evaluate current greedy policy after each batch
            if viz_cb is not None or stop_when_optimal:
                total_r, trail, term, trunc = self.evaluate_episode(eps=0.0)

                # Compute visited targets in trail
                visited = set()
                tset = set(self.env.targets)
                for (yy, xx) in trail:
                    if (yy, xx) in tset:
                        visited.add((yy, xx))
                visited_count = len(visited)
                path_len = max(0, len(trail) - 1)

                if viz_cb is not None:
                    try:
                        viz_cb(
                            batches,
                            {
                                "total_r": float(total_r),
                                "path_len": int(path_len),
                                "visited_count": int(visited_count),
                                "terminated": bool(term),
                                "truncated": bool(trunc),
                            },
                            trail,
                        )
                    except Exception:
                        # Visualization should never crash training
                        pass

                if stop_when_optimal and optimal_length is not None:
                    # Stop early if optimal-length path that visits all targets and reaches goal is achieved
                    if term and (visited_count == len(self.env.targets)) and (path_len == int(optimal_length)):
                        self._optimal_reached = True
                        return True, steps, batches

            # Convergence check (need at least conv_min_batches)
            if len(self.delta_buffer) >= self.cfg.conv_min_batches:
                recent = np.array(self.delta_buffer[-self.cfg.conv_min_batches :])
                avg_th, std_th = self._delta_stat_thresholds()
                if (float(recent.mean()) <= avg_th) and (float(recent.std()) <= std_th):
                    return True, steps, batches
        return False, steps, batches

    def train(
        self,
        progress_cb=None,
        viz_cb=None,
        episode_viz_cb=None,
        stop_when_optimal: bool = False,
        optimal_length: int | None = None,
    ):
        """Train across phases according to epsilon schedule."""
        max_phases = self.cfg.max_phases
        converged_all = False
        history = []
        # Allow training up to max_phases even if epsilon schedule is shorter:
        # repeat the last epsilon value.
        for pi in range(max_phases):
            eps = float(self.eps_sched[pi]) if pi < len(self.eps_sched) else float(self.eps_sched[-1])
            if progress_cb:
                progress_cb(f"Phase {pi+1}/{max_phases}, eps={eps:.3f} ...")
            conv, steps_used, batches = self.train_one_phase(
                eps,
                self.cfg.max_global_steps,
                viz_cb=viz_cb,
                episode_viz_cb=episode_viz_cb,
                stop_when_optimal=stop_when_optimal,
                optimal_length=optimal_length,
            )
            history.append(
                {
                    "phase": pi,
                    "epsilon": eps,
                    "converged": conv,
                    "steps_used": steps_used,
                    "batches": batches,
                    "global_steps": int(self.global_steps),
                    "recent_delta": self.delta_buffer[-self.cfg.conv_min_batches :]
                    if len(self.delta_buffer) >= self.cfg.conv_min_batches
                    else self.delta_buffer,
                }
            )
            # Stop rules
            if getattr(self, "_optimal_reached", False):
                break
            if not conv:
                break
            self.phase_idx += 1
        else:
            converged_all = True
        return history, converged_all

    def evaluate_episode(self, eps: float = 0.0, max_steps: int | None = None, render_trail: bool = False):
        if max_steps is None:
            max_steps = self.env.max_steps
        obs = self.env.reset()
        done = False
        total_r = 0.0
        trail = [tuple(self.env.pos)]
        term, trunc = False, False
        while not done:
            a = self.act_epsilon_greedy(obs, eps)
            obs, r, term, trunc, _ = self.env.step(a)
            total_r += float(r)
            trail.append(tuple(self.env.pos))
            if term or trunc or len(trail) >= max_steps:
                done = True
        return total_r, trail, term, trunc

    def save(self, path: str):
        np.savez_compressed(path, V=self.V, N=self.N, W=self.env.W, H=self.env.H, T=self.T)

    def load(self, path: str):
        data = np.load(path, allow_pickle=True)
        V = data["V"]
        N = data["N"]
        W = int(data.get("W", self.env.W))
        H = int(data.get("H", self.env.H))
        T = int(data.get("T", len(self.env.targets)))
        # Validate shape matches current env
        sz_expected = self.env.W * self.env.H * (1 << len(self.env.targets))
        if V.size != sz_expected or N.size != sz_expected or W != self.env.W or H != self.env.H or T != len(self.env.targets):
            raise ValueError(
                "Loaded model dimensions do not match current environment. "
                f"Model(W={W}, H={H}, T={T}) vs Env(W={self.env.W}, H={self.env.H}, T={len(self.env.targets)})."
            )
        self.V = V.astype(np.float32, copy=False)
        self.N = N.astype(np.uint32, copy=False)
        self._build_state_encoding()


# ----------------------------
# Streamlit UI (behind main guard)
# ----------------------------
def run_streamlit_app():
    import streamlit as st

    st.set_page_config(page_title="Single Drone RL (Tabular, No Gym)", layout="wide")

    st.title("🚁 Single Surveillance Drone Path Optimization (Tabular RL, No Gym)")

    with st.sidebar:
        st.header("Configuration")

        # Environment controls
        width = st.number_input("Grid width", 10, 100, 30, 1)
        height = st.number_input("Grid height", 10, 100, 30, 1)
        moves8 = st.selectbox("Move set", ["8-neighbors", "4-neighbors"], index=0) == "8-neighbors"
        max_steps_env = st.number_input("Episode max steps", 10, 5000, 400, 10)
        n_targets = st.number_input("Number of targets (bitmask)", 1, 10, 4, 1)
        obstacle_ratio = st.slider("Obstacle ratio", 0.0, 0.6, 0.10, 0.01)
        start_xy = st.text_input("Start (y,x)", "0,0")
        goal_xy = st.text_input("Goal (y,x)", f"{height-1},{width-1}")
        R_target = st.number_input("Reward per target", 0.0, 100.0, 5.0, 0.1)
        R_goal = st.number_input("Goal reward multiplier (×visited targets)", 0.0, 100.0, 10.0, 0.5)
        lambda_step = st.number_input("Per-step penalty", 0.0, 5.0, 0.1, 0.01)
        seed = st.number_input("Random seed", 0, 999999, 42, 1)

        # RL controls
        eps_str = st.text_input("Epsilon schedule (comma-separated)", "0.9,0.5,0.1,0.01")
        batch_size = st.number_input("Batch size (steps)", 100, 100000, 2000, 100)
        max_phases = st.number_input("Max phases", 1, 20, 4, 1)
        max_global_steps = st.number_input("Max global steps", 1000, 5_000_000, 400_000, 1000)
        lr_k = st.number_input("Learning rate decay k", 0.01, 5.0, 0.8, 0.01)
        conv_mode = st.selectbox("Convergence mode", ["percent", "abs"], index=0)
        conv_avg_th = st.number_input("Convergence avg threshold", 0.000001, 1.0, 0.05, 0.000001, format="%.6f")
        conv_std_th = st.number_input("Convergence std threshold", 0.000001, 1.0, 0.05, 0.000001, format="%.6f")
        conv_min_batches = st.number_input("Convergence min batches", 1, 100, 3, 1)

        # Runtime controls
        step_delay = st.slider("Visualization delay per step (sec)", 0.0, 0.25, 0.03, 0.01)
        eval_eps = st.slider("Eval epsilon (for demo)", 0.0, 1.0, 0.0, 0.01)

        st.markdown("---")
        reset_btn = st.button("♻️ Reset Environment/Agent")
        train_btn = st.button("🚀 Train (all phases or until budget)")
        eval_btn = st.button("▶️ Run 1 Episode (visualize)")
        save_btn = st.button("💾 Save Model")
        load_btn = st.button("📂 Load Model")

        st.markdown("---")
        st.subheader("Training Visualization")
        live_viz_training = st.checkbox("Live visualize training episodes (during batch)", value=True)
        visualize_training = st.checkbox("Visualize after each batch (greedy episode)", value=False)
        train_viz_delay = st.slider("Train viz step delay (sec)", 0.0, 0.2, 0.01, 0.01)
        stop_when_optimal = st.checkbox("Stop when optimal path found", value=True)

    # Parse positions
    def parse_pair(s: str, default: Tuple[int, int]) -> Tuple[int, int]:
        try:
            y, x = s.split(",")
            return int(y.strip()), int(x.strip())
        except Exception:
            return default

    start_pos = _sanitize_pos(parse_pair(start_xy, (0, 0)), height, width)
    goal_pos = _sanitize_pos(parse_pair(goal_xy, (height - 1, width - 1)), height, width)

    # Build configs
    env_cfg = GridEnvConfig(
        width=width,
        height=height,
        moves8=moves8,
        max_steps=max_steps_env,
        n_targets=n_targets,
        obstacle_ratio=obstacle_ratio,
        start=start_pos,
        goal=goal_pos,
        R_target=R_target,
        R_goal=R_goal,
        lambda_step=lambda_step,
    )
    eps_list = [float(x.strip()) for x in eps_str.split(",") if x.strip()]
    rl_cfg = RLConfig(
        gamma=0.99,
        epsilon_schedule=eps_list,
        batch_size=batch_size,
        max_phases=max_phases,
        max_global_steps=max_global_steps,
        lr_decay_k=lr_k,
        conv_mode=conv_mode,
        conv_avg_th=conv_avg_th,
        conv_std_th=conv_std_th,
        conv_min_batches=conv_min_batches,
    )

    # Session state
    if "env" not in st.session_state or reset_btn:
        st.session_state.env = GridEnv(env_cfg, seed=seed)
        st.session_state.agent = TabularAgent(st.session_state.env, rl_cfg, seed=seed)
        st.session_state.history = []
        st.session_state.last_episode = None

    def rebuild_if_needed():
        e: GridEnv = st.session_state.env
        # If any env config field changes, rebuild env + agent
        need_env_rebuild = e.cfg != env_cfg
        if need_env_rebuild:
            st.session_state.env = GridEnv(env_cfg, seed=seed)
            st.session_state.agent = TabularAgent(st.session_state.env, rl_cfg, seed=seed)
            st.session_state.history = []
            st.session_state.last_episode = None
        else:
            # Only RL knobs changed: update agent config and dependent attributes
            agent: TabularAgent = st.session_state.agent
            agent.cfg = rl_cfg
            agent.gamma = rl_cfg.gamma
            agent.eps_sched = rl_cfg.epsilon_schedule
            agent.batch_size = rl_cfg.batch_size
            agent.lr_k = rl_cfg.lr_decay_k

    # Layout
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Environment Preview")
        fig, _ = st.session_state.env.render_canvas(title="Initial Layout")
        st.pyplot(fig)
        plt.close(fig)
        st.caption("■ Cyan: Start, ■ Yellow: Goal, ★ Targets, ■ Dark: Obstacles")

    with col_right:
        a: TabularAgent = st.session_state.agent
        st.subheader("Agent / Training Stats")
        st.metric("Global Steps", f"{a.global_steps:,}")
        st.metric("Phases Completed", f"{a.phase_idx}/{a.cfg.max_phases}")
        # Optimal path length (computed w.r.t. current env)
        try:
            opt_len = compute_optimal_path_length(st.session_state.env)
        except Exception:
            opt_len = None
        if opt_len is not None:
            st.metric("Optimal Path Length", f"{opt_len} steps")
        else:
            st.metric("Optimal Path Length", "unreachable/unknown")
        st.write("Recent ΔV per batch (last 10):")
        if len(a.delta_buffer) > 0:
            st.line_chart(a.delta_buffer[-10:])
        else:
            st.info("No training yet. Click 'Train' to start.")

    # Buttons handlers
    status = st.empty()
    log_box = st.empty()

    if train_btn:
        rebuild_if_needed()
        a: TabularAgent = st.session_state.agent

        def cb(msg: str):
            status.info(msg)

        # Visualization placeholders and progress
        train_viz_placeholder = st.empty()
        train_progress = st.progress(0, text="Batch progress")
        path_len_log = []

        # Compute optimal path length for stopping criterion
        try:
            opt_len = compute_optimal_path_length(st.session_state.env)
        except Exception:
            opt_len = None

        def episode_viz_cb(ep_metrics: dict, trail):
            # Update per-episode training visualization and progress
            if live_viz_training:
                for i in range(1, len(trail) + 1):
                    env = st.session_state.env
                    env.reset()
                    env.pos = list(trail[i - 1])
                    fig, _ = env.render_canvas(agent_trail=trail[:i], title=f"Training ep {ep_metrics.get('ep_idx_in_batch', 0)} step {i}/{len(trail)} (batch {a.phase_idx+1})")
                    train_viz_placeholder.pyplot(fig)
                    plt.close(fig)
                    import time as _time

                    _time.sleep(train_viz_delay)
            # Update progress bar based on accumulated batch steps
            bs = int(ep_metrics.get("batch_steps", 0))
            denom = max(1, int(a.batch_size))
            train_progress.progress(min(bs / denom, 1.0), text=f"Batch progress: {bs}/{denom} steps")

        def viz_cb(batch_idx: int, metrics: dict, trail):
            # Update per-batch greedy episode visualization
            path_len_log.append(metrics.get("path_len", 0))
            if visualize_training:
                # Animate the greedy trail quickly
                for i in range(1, len(trail) + 1):
                    env = st.session_state.env
                    env.reset()
                    env.pos = list(trail[i - 1])
                    fig, _ = env.render_canvas(agent_trail=trail[:i], title=f"Batch {batch_idx}: Greedy Episode {i}/{len(trail)}")
                    train_viz_placeholder.pyplot(fig)
                    plt.close(fig)
                    import time as _time

                    _time.sleep(train_viz_delay)

        hist, conv_all = a.train(
            progress_cb=cb,
            viz_cb=viz_cb,
            episode_viz_cb=episode_viz_cb,
            stop_when_optimal=(stop_when_optimal and opt_len is not None),
            optimal_length=opt_len,
        )
        st.session_state.history.extend(hist)
        if conv_all:
            status.success("✅ Training completed: all phases converged within budget.")
        else:
            if getattr(a, "_optimal_reached", False):
                status.success("🏁 Training stopped: optimal path reached.")
            else:
                status.warning("⚠️ Training stopped (phase non-convergence or budget exhausted).")

        with log_box.container():
            st.write("Training history (last entries):")
            for h in st.session_state.history[-5:]:
                st.json(
                    {
                        "phase": int(h["phase"]),
                        "epsilon": float(h["epsilon"]),
                        "converged": bool(h["converged"]),
                        "steps_used": int(h["steps_used"]),
                        "batches": int(h["batches"]),
                        "global_steps": int(h["global_steps"]),
                        "recent_delta": [
                            float(x)
                            for x in (
                                h["recent_delta"]
                                if isinstance(h["recent_delta"], (list, np.ndarray))
                                else [h["recent_delta"]]
                            )
                        ],
                    }
                )
            if path_len_log:
                st.write("Greedy path length per batch (recent):")
                st.line_chart(path_len_log[-50:])

    if eval_btn:
        rebuild_if_needed()
        env: GridEnv = st.session_state.env
        agent: TabularAgent = st.session_state.agent

        trail_placeholder = st.empty()
        total_r, trail, term, trunc = agent.evaluate_episode(eps=eval_eps, max_steps=env.max_steps, render_trail=True)

        # Animate
        for i in range(1, len(trail) + 1):
            env.reset()
            env.pos = list(trail[i - 1])
            fig, _ = env.render_canvas(agent_trail=trail[:i], title=f"Episode Step {i}/{len(trail)}")
            trail_placeholder.pyplot(fig)
            plt.close(fig)
            import time as _time

            _time.sleep(step_delay)

        st.success(f"Episode finished. Total Reward={total_r:.2f}, terminated={term}, truncated={trunc}")
        st.session_state.last_episode = (total_r, trail, term, trunc)

    if save_btn:
        path = "drone_tabular_model.npz"
        st.session_state.agent.save(path)
        st.success(f"Saved model to {path}")

    if load_btn:
        path = st.text_input("Enter model path to load", "drone_tabular_model.npz")
        if st.button("Load Now"):
            try:
                st.session_state.agent.load(path)
                st.success(f"Loaded model from {path}")
            except Exception as e:
                st.error(f"Failed to load: {e}")

    # ----------------------------
    # FAQ / Controls explanation
    # ----------------------------
    st.markdown("---")
    st.markdown("### ❓ 학습은 언제까지 지속되는가? 내가 조정할 수 있을까?")
    st.markdown(
        """
        - **학습 종료 조건은 3가지** 중 하나로 달성되면 멈춥니다.
          1) **수렴(Convergence)**: 배치 단위 ΔV(값함수 변화량)의 **최근 평균과 표준편차**가 임계값 이하일 때, 해당 ε-페이즈가 종료됩니다. 모든 페이즈가 수렴하면 학습이 끝납니다.  
             - 사이드바에서 **`Convergence mode/avg_th/std_th/min_batches`**로 조정 가능  
          2) **최대 글로벌 스텝**: `Max global steps` 한도에 도달하면 중단  
          3) **최대 페이즈 수**: `Max phases`를 모두 소진하면 중단
        - **격자 크기, 타깃 수, 장애물 비율, 보상/패널티, ε 스케줄, 배치 크기, 최대 스텝/페이즈 등은 모두 사이드바에서 즉시 변경 가능**합니다.  
        - **`Run 1 Episode`** 버튼으로 현재 정책 상태를 **실시간 시각화**(실패/충돌·최대 스텝 초과에 의한 **truncated** 포함)할 수 있습니다.
        """
    )


if __name__ == "__main__":
    # Only executed when running as a script (e.g., via streamlit run app.py)
    run_streamlit_app()
