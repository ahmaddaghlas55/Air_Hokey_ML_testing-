"""
Air hockey simulation — entry point.
Move mouse to play. First to 7 goals wins. Press R to restart.
"""

import json
import time

from sim_backend import SimBackend
from strategy import DefendStrategy


HIT_LOG_PATH = "opponent_hits.jsonl"
TIME_SCALE = 0.6      # 1.0 = real speed, 0.5 = half speed


def log_player_hit(hit):
    with open(HIT_LOG_PATH, "a") as f:
        f.write(json.dumps(hit) + "\n")


def main():
    backend = SimBackend(on_player_hit=log_player_hit)
    strategy = DefendStrategy()

    last = time.time()

    while backend.is_running():
        now = time.time()
        dt = now - last
        last = now
        if dt <= 0 or dt > 1 / 30:
            dt = 1 / 60
        sim_dt = dt * TIME_SCALE

        state = backend.read_state()
        tx, ty = strategy.update(state, sim_dt)
        backend.set_robot_target(tx, ty)

        backend.step(sim_dt)


if __name__ == "__main__":
    main()