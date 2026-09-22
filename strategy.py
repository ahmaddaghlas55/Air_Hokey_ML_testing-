"""
Robot strategy.

Two rule-based behaviors plus optional residual Q-learning:

  ATTACK  — when the puck is on the robot's side, position behind
            it and drive forward to push it toward the player's goal.
  DEFEND  — when the puck is on the player's side or moving toward
            the robot, predict where it will cross the defensive line
            and go there.

Residual Q-learning (if qtable.npy exists) applies small corrections
to whichever rule is active. Delete qtable.npy to disable it.
"""

import random

from physics_env import (
    TABLE_WIDTH, TABLE_HEIGHT,
    PUCK_RADIUS, PADDLE_RADIUS,
    ROBOT_MIN_X, ROBOT_MAX_X, ROBOT_MIN_Y, ROBOT_MAX_Y,
    ROBOT_HOME,
)
from predictor import position_at_time
from difficulty import PRESETS


# how far behind the puck the attack target sits
ATTACK_STANDOFF = PUCK_RADIUS + PADDLE_RADIUS -10

# if the puck is on the robot's side and moving slower than this,
# switch to attack (px/s)
ATTACK_SPEED_THRESHOLD = 150


def _clamp(v, lo, hi):
    return max(lo, min(v, hi))


def _attack_target(px, py):
    """
    Stand to the right of the puck (behind it, relative to the goal
    we're attacking). Then a straight leftward push sends the puck
    toward the player's goal.
    """
    tx = px + ATTACK_STANDOFF
    ty = py
    tx = _clamp(tx, ROBOT_MIN_X, ROBOT_MAX_X)
    ty = _clamp(ty, ROBOT_MIN_Y, ROBOT_MAX_Y)
    return (tx, ty)


def _defend_target(px, py, vx, vy):
    """
    Predict where the puck crosses the robot's defensive line and
    return that intercept point.
    """
    if vx <= 10:
        return ROBOT_HOME

    t_hit = (ROBOT_MIN_X - px) / vx
    if t_hit < 0 or t_hit > 2.0:
        return ROBOT_HOME

    ix, iy = position_at_time(px, py, vx, vy, t_hit)
    ix = _clamp(ix, ROBOT_MIN_X, ROBOT_MAX_X)
    iy = _clamp(iy, ROBOT_MIN_Y, ROBOT_MAX_Y)
    return (ix, iy)


def _classical_plan(state):
    """
    Top-level rule-based behavior. Returns a target (x, y).
    """
    px, py = state.puck_pos
    vx, vy = state.puck_vel

    on_robot_side = px > TABLE_WIDTH / 2

    if on_robot_side:
        # if the puck is heading quickly away, don't chase it — defend
        if vx > ATTACK_SPEED_THRESHOLD:
            return _defend_target(px, py, vx, vy)
        # otherwise, attack: get behind the puck and drive it left
        return _attack_target(px, py)

    # puck on player's side — pure defense
    return _defend_target(px, py, vx, vy)


class DefendStrategy:

    def __init__(self, params=None):
        self.params = params or PRESETS["medium"]
        self._committed_target = ROBOT_HOME
        self._time_until_replan = 0.0

        try:
            from qpolicy import QPolicy
            self.qpolicy = QPolicy()
        except Exception as e:
            print(f"qpolicy unavailable, using rule-based only: {e}")
            self.qpolicy = None

    def set_difficulty(self, params):
        self.params = params

    def update(self, state, dt):
        self._time_until_replan -= dt

        if self._time_until_replan <= 0:
            self._committed_target = self._plan(state)
            self._time_until_replan = self.params.reaction_delay

        return self._committed_target

    def _plan(self, state):
        # 1. rule-based target
        base_x, base_y = _classical_plan(state)

        # 2. optional residual correction
        if self.qpolicy is not None and self.qpolicy.is_loaded():
            adjusted = self.qpolicy.choose_residual(
                state, (base_x, base_y))
            if adjusted is not None:
                return adjusted

        # 3. difficulty noise
        if self.params.noise_sigma > 0:
            base_x += random.gauss(0, self.params.noise_sigma)
            base_y += random.gauss(0, self.params.noise_sigma)
            base_x = _clamp(base_x, ROBOT_MIN_X, ROBOT_MAX_X)
            base_y = _clamp(base_y, ROBOT_MIN_Y, ROBOT_MAX_Y)

        return (base_x, base_y)