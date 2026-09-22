"""
Residual RL training loop with warm-start support.

If qtable.npy exists, training continues from it. Otherwise it
starts fresh. Each invocation is meant to be short (5000 episodes);
run it many times across play sessions to accumulate experience.

Run:  python train.py
Saves: qtable.npy
"""

import os
import random
import numpy as np

from physics_env import (
    AirHockeyWorld, TABLE_WIDTH, TABLE_HEIGHT,
    ROBOT_MIN_X, ROBOT_MAX_X, ROBOT_MIN_Y, ROBOT_MAX_Y,
    PUCK_RADIUS, PADDLE_RADIUS,
)
from sim_opponent import ScriptedOpponent
from strategy import _classical_plan


# ---------- discretization (must match qpolicy.py) ----------

def discretize(world):
    px, py = world.puck_body.position
    vx, vy = world.puck_body.velocity
    ry = world.robot_body.position.y

    bx = min(7, max(0, int(px / TABLE_WIDTH * 8)))
    by = min(7, max(0, int(py / TABLE_HEIGHT * 8)))
    sx = 1 if vx > 0 else 0
    ryb = min(4, max(0, int(ry / TABLE_HEIGHT * 5)))
    return (bx, by, sx, ryb)


def state_index(s):
    bx, by, sx, ryb = s
    return ((bx * 8 + by) * 2 + sx) * 5 + ryb


N_STATES = 8 * 8 * 2 * 5
N_ACTIONS = 9
ACTIONS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
           (-1, -1), (-1, 1), (1, -1), (1, 1)]
RESIDUAL_STEP = 15


# ---------- reward ----------

def compute_reward(world, goal):
    if goal == 'robot':
        return 10.0
    if goal == 'player':
        return -10.0

    px, py = world.puck_body.position
    rx, ry = world.robot_body.position
    dist = ((px - rx) ** 2 + (py - ry) ** 2) ** 0.5
    if dist < PUCK_RADIUS + PADDLE_RADIUS + 10:
        return 0.02
    return -0.001


# ---------- episode ----------

class _StateView:
    pass


def run_episode(world, opponent, qtable, alpha, gamma, epsilon,
                max_steps=1200, dt=1 / 60):
    world.serve(toward='robot')

    state = discretize(world)
    total_reward = 0.0

    for _ in range(max_steps):
        sv = _StateView()
        sv.puck_pos = tuple(world.puck_body.position)
        sv.puck_vel = tuple(world.puck_body.velocity)
        sv.robot_pos = tuple(world.robot_body.position)

        base_x, base_y = _classical_plan(sv)

        if random.random() < epsilon:
            action_idx = random.randrange(N_ACTIONS)
        else:
            action_idx = int(np.argmax(qtable[state_index(state)]))

        dx, dy = ACTIONS[action_idx]
        rx = base_x + dx * RESIDUAL_STEP
        ry = base_y + dy * RESIDUAL_STEP
        rx = max(ROBOT_MIN_X, min(rx, ROBOT_MAX_X))
        ry = max(ROBOT_MIN_Y, min(ry, ROBOT_MAX_Y))
        world.set_robot_target(rx, ry, dt)

        class _S:
            pass
        s = _S()
        s.puck_pos = tuple(world.puck_body.position)
        s.puck_vel = tuple(world.puck_body.velocity)
        raw_target = opponent.act(s, dt)
        ox, oy = world.player_body.position
        tx = ox + (raw_target[0] - ox)
        ty = oy + (raw_target[1] - oy)
        world.set_player_target(tx, ty, dt)

        world.step(dt)

        next_state = discretize(world)
        goal = world.check_goal()
        reward = compute_reward(world, goal)

        si = state_index(state)
        ni = state_index(next_state)
        best_next = float(np.max(qtable[ni]))
        qtable[si, action_idx] += alpha * (
            reward + gamma * best_next - qtable[si, action_idx])

        total_reward += reward
        state = next_state

        if goal is not None:
            break

    return total_reward


# ---------- training driver ----------

def train(episodes=5000, alpha=0.1, gamma=0.99,
          eps_start=0.3, eps_end=0.05):
    world = AirHockeyWorld()
    opponent = ScriptedOpponent()

    # warm-start
    if os.path.exists("qtable.npy"):
        qtable = np.load("qtable.npy")
        if qtable.shape != (N_STATES, N_ACTIONS):
            print(f"existing qtable shape {qtable.shape} mismatch — starting fresh")
            qtable = np.zeros((N_STATES, N_ACTIONS))
        else:
            print("warm-start: continuing from existing qtable.npy")
    else:
        qtable = np.zeros((N_STATES, N_ACTIONS))
        print("cold-start: no existing qtable.npy")

    wins = 0
    losses = 0
    timeouts = 0

    for ep in range(episodes):
        frac = ep / episodes
        epsilon = eps_start + frac * (eps_end - eps_start)

        r = run_episode(world, opponent, qtable, alpha, gamma, epsilon)

        if r >= 5.0:
            wins += 1
        elif r <= -5.0:
            losses += 1
        else:
            timeouts += 1

        if (ep + 1) % 500 == 0:
            total = wins + losses + timeouts
            print(f"ep {ep+1:>6}  "
                  f"win_rate={wins/total:.2f}  "
                  f"loss_rate={losses/total:.2f}  "
                  f"timeout_rate={timeouts/total:.2f}  "
                  f"eps={epsilon:.2f}")
            wins = 0
            losses = 0
            timeouts = 0

    np.save("qtable.npy", qtable)
    print("saved qtable.npy")

    policy = qtable.argmax(axis=1)
    counts = np.bincount(policy, minlength=N_ACTIONS)
    print("\naction distribution in final policy:")
    for i, c in enumerate(counts):
        dx, dy = ACTIONS[i]
        print(f"  ({dx:+d},{dy:+d}) : {c} states")
    pct = 100 * counts[0] / counts.sum()
    print(f"\naction 0 (no correction) chosen in {pct:.1f}% of states")


if __name__ == "__main__":
    train()