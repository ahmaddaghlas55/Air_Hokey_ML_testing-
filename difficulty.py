"""
Map a player_model.json into concrete robot parameters.
This is the adaptive-difficulty layer: it decides how hard
the robot should try, and translates that into numbers the
strategy can consume.

Later, this is what a hardware deployment reads to configure
the real robot — same interface, no code change.
"""

from dataclasses import dataclass


@dataclass
class DifficultyParams:
    reaction_delay: float       # seconds before robot commits to a target
    paddle_max_speed: float     # px/s
    prediction_horizon: float   # seconds to look ahead
    noise_sigma: float          # px of Gaussian noise on target
    name: str


PRESETS = {
    "easy":       DifficultyParams(0.35, 400, 0.3, 30, "easy"),
    "medium":     DifficultyParams(0.20, 700, 0.5, 15, "medium"),
    "hard":       DifficultyParams(0.10, 1000, 0.8, 6, "hard"),
    "impossible": DifficultyParams(0.02, 1600, 1.2, 0, "impossible"),
}


def _player_is_winning(player_score, robot_score, model):
    """
    Estimate whether the player is dominating.
    Combines match score with model signals (fast shots, corner-heavy).
    """
    # if the player is winning the match, they're clearly doing well
    if player_score - robot_score >= 2:
        return True
    if player_score - robot_score <= -2:
        return False

    # otherwise use the model
    if model is None:
        return False
    skill = 0.0
    skill += min(1.0, model.get("avg_speed", 0) / 900)          # fast shot
    skill += model.get("corner_bias", 0) * 0.5                  # corner hits
    skill += abs(model.get("attack_bias", 0.5) - 0.5) * 1.0     # decisive
    return skill > 0.7


def pick_difficulty(player_score, robot_score, model):
    """
    Choose a difficulty level. The rules are intentionally simple:
    they're the thing you tune by playing.
    """
    if model is None or model.get("n_hits", 0) < 5:
        return PRESETS["easy"]       # warm-up until we know the player

    if _player_is_winning(player_score, robot_score, model):
        # ratchet up
        if player_score - robot_score >= 4:
            return PRESETS["impossible"]
        return PRESETS["hard"]

    if player_score - robot_score <= -3:
        return PRESETS["easy"]

    return PRESETS["medium"]