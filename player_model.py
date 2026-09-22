"""
Build a statistical model of the player from opponent_hits.jsonl.

Only the most recent MAX_HITS are used, so the model follows the
player's evolving style rather than averaging over all history.

Tolerates both old field names (opponent_*) and new ones (player_*).
"""

import json
import math
import os

HIT_LOG = "opponent_hits.jsonl"
MODEL_PATH = "player_model.json"

ZONES_X = 5
ZONES_Y = 5
MAX_HITS = 500          # sliding window: only the most recent N hits

try:
    from physics_env import TABLE_WIDTH, TABLE_HEIGHT
except ImportError:
    TABLE_WIDTH = 1000
    TABLE_HEIGHT = 500


# ---------- helpers ----------

def _zone_x(x):
    return min(ZONES_X - 1, int(x / (TABLE_WIDTH / 2) * ZONES_X))


def _zone_y(y):
    return min(ZONES_Y - 1, int(y / TABLE_HEIGHT * ZONES_Y))


def _extract_pos(hit):
    pos = hit.get("player_pos") or hit.get("opponent_pos")
    if pos is None:
        return None
    return float(pos[0]), float(pos[1])


def _extract_vel(hit):
    vel = hit.get("player_vel") or hit.get("opponent_vel")
    if vel is None:
        return 0.0, 0.0
    return float(vel[0]), float(vel[1])


def load_hits(path=HIT_LOG, max_hits=MAX_HITS):
    """Load the last `max_hits` valid hits from the log."""
    hits = []
    if not os.path.exists(path):
        return hits
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                hits.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return hits[-max_hits:]


# ---------- core ----------

def build_model(hits):
    model = {
        "n_hits": 0,
        "x_zone_counts": [0] * ZONES_X,
        "y_zone_counts": [0] * ZONES_Y,
        "avg_speed": 0.0,
        "avg_angle_deg": 0.0,
        "attack_bias": 0.0,
        "corner_bias": 0.0,
        "reaction_gaps_ms": [],
    }

    if not hits:
        return model

    speeds = []
    angles = []
    attack = 0
    corners = 0
    times = []
    valid = 0

    for h in hits:
        pos = _extract_pos(h)
        if pos is None:
            continue
        px, py = pos

        puck_vel = h.get("puck_vel_after")
        if puck_vel is None or len(puck_vel) != 2:
            continue
        vx, vy = float(puck_vel[0]), float(puck_vel[1])

        speed = math.hypot(vx, vy)
        if speed > 3000:      # skip physics-glitch records
            continue

        model["x_zone_counts"][_zone_x(px)] += 1
        model["y_zone_counts"][_zone_y(py)] += 1

        speeds.append(speed)
        angles.append(math.degrees(math.atan2(vy, vx)))

        if vx > 0:
            attack += 1

        for cx, cy in ((0, 0), (0, TABLE_HEIGHT),
                       (TABLE_WIDTH / 2, 0), (TABLE_WIDTH / 2, TABLE_HEIGHT)):
            if math.hypot(px - cx, py - cy) < 120:
                corners += 1
                break

        t = h.get("t")
        if t is not None:
            times.append(float(t))

        valid += 1

    model["n_hits"] = valid
    if valid == 0:
        return model

    model["avg_speed"] = sum(speeds) / valid
    model["avg_angle_deg"] = sum(angles) / valid
    model["attack_bias"] = attack / valid
    model["corner_bias"] = corners / valid

    times.sort()
    gaps = [(times[i + 1] - times[i]) * 1000.0 for i in range(len(times) - 1)]
    gaps = [g for g in gaps if 100.0 <= g <= 10000.0]
    model["reaction_gaps_ms"] = gaps[:200]

    return model


# ---------- persistence ----------

def save_model(model, path=MODEL_PATH):
    with open(path, "w") as f:
        json.dump(model, f, indent=2)


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def rebuild():
    hits = load_hits()
    model = build_model(hits)
    save_model(model)
    return model


# ---------- CLI ----------

if __name__ == "__main__":
    m = rebuild()
    print(f"model built from {m['n_hits']} hits (last {MAX_HITS} window)")
    print(f"  avg speed:    {m['avg_speed']:.0f} px/s")
    print(f"  avg angle:    {m['avg_angle_deg']:.1f} deg")
    print(f"  attack bias:  {m['attack_bias']:.2f}  (1.0 = always toward robot)")
    print(f"  corner bias:  {m['corner_bias']:.2f}  (1.0 = always from a corner)")
    print(f"  x zones:      {m['x_zone_counts']}")
    print(f"  y zones:      {m['y_zone_counts']}")
    if m["reaction_gaps_ms"]:
        avg_gap = sum(m["reaction_gaps_ms"]) / len(m["reaction_gaps_ms"])
        print(f"  avg gap:      {avg_gap:.0f} ms")