"""
Scripted opponent that mimics the player from player_model.json.
Used to train the Q-learning robot headless (no pygame).
"""

import random
from player_model import load_model


class ScriptedOpponent:
    """
    Plays like the logged player. Given puck position/velocity,
    returns a target (x, y) for the opponent paddle each frame.
    """

    def __init__(self, model=None):
        self.model = model or load_model()
        if self.model is None or self.model.get("n_hits", 0) < 5:
            # no data yet — fall back to a neutral random-ish player
            self.avg_speed = 500.0
            self.attack_bias = 0.6
            self.corner_bias = 0.2
        else:
            self.avg_speed = self.model["avg_speed"]
            self.attack_bias = self.model["attack_bias"]
            self.corner_bias = self.model["corner_bias"]

        self.home = (100, 250)

    def act(self, state, dt):
        """
        Simple heuristic: move toward puck when it's on our side,
        else drift home. Add noise so it isn't a perfect mirror.
        """
        px, py = state.puck_pos
        vx, vy = state.puck_vel

        # opponent only acts on the left half
        if px > 500:
            return self.home

        # chase the puck at ~our modeled speed
        target = (px, py)

        # occasionally go for a "corner shot" setup if the model says
        # this player likes corners
        if random.random() < self.corner_bias * 0.3:
            corner_y = 60 if py < 250 else 440
            target = (px, corner_y)

        return target