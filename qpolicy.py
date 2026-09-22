"""
Runtime residual Q-table lookup.

Actions are small corrections to the classical target.
Action 0 = (0,0) means "trust classical exactly".
"""

import os
import numpy as np

from physics_env import (
    TABLE_WIDTH, TABLE_HEIGHT,
    ROBOT_MIN_X, ROBOT_MAX_X, ROBOT_MIN_Y, ROBOT_MAX_Y,
)


# 9 residual actions. Action 0 = no correction.
ACTIONS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
           (-1, -1), (-1, 1), (1, -1), (1, 1)]
N_ACTIONS = 9

# must match train.py
N_STATES = 8 * 8 * 2 * 5         # 640
RESIDUAL_STEP = 15               # pixels per residual action


def _clamp(v, lo, hi):
    return max(lo, min(v, hi))


def _discretize(state):
    px, py = state.puck_pos
    vx, vy = state.puck_vel
    ry = state.robot_pos[1]

    bx = min(7, max(0, int(px / TABLE_WIDTH * 8)))
    by = min(7, max(0, int(py / TABLE_HEIGHT * 8)))
    sx = 1 if vx > 0 else 0
    ryb = min(4, max(0, int(ry / TABLE_HEIGHT * 5)))
    return (bx, by, sx, ryb)


def _state_index(s):
    bx, by, sx, ryb = s
    return ((bx * 8 + by) * 2 + sx) * 5 + ryb


class QPolicy:
    def __init__(self, path="qtable.npy"):
        if not os.path.exists(path):
            self.qtable = None
            return
        try:
            table = np.load(path)
        except (OSError, ValueError) as e:
            print(f"qtable load failed: {e}")
            self.qtable = None
            return

        if table.shape != (N_STATES, N_ACTIONS):
            print(f"warning: qtable shape {table.shape} "
                  f"!= expected ({N_STATES}, {N_ACTIONS}). Ignoring.")
            self.qtable = None
            return

        self.qtable = table

    def is_loaded(self):
        return self.qtable is not None

    def choose_residual(self, state, base_target):
        if self.qtable is None:
            return None

        s = _state_index(_discretize(state))
        action_idx = int(np.argmax(self.qtable[s]))
        dx, dy = ACTIONS[action_idx]

        tx = base_target[0] + dx * RESIDUAL_STEP
        ty = base_target[1] + dy * RESIDUAL_STEP
        tx = _clamp(tx, ROBOT_MIN_X, ROBOT_MAX_X)
        ty = _clamp(ty, ROBOT_MIN_Y, ROBOT_MAX_Y)
        return (tx, ty)