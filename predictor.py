"""
Simple trajectory prediction with one wall bounce.
All units are pixels and pixels/sec, matching physics_env.
"""

from physics_env import (
    TABLE_WIDTH, TABLE_HEIGHT, WALL_THICKNESS,
    PUCK_RADIUS,
)

# playable bounds for the puck's CENTER (keeps it off walls)
PUCK_MIN_X = WALL_THICKNESS + PUCK_RADIUS
PUCK_MAX_X = TABLE_WIDTH - WALL_THICKNESS - PUCK_RADIUS
PUCK_MIN_Y = WALL_THICKNESS + PUCK_RADIUS
PUCK_MAX_Y = TABLE_HEIGHT - WALL_THICKNESS - PUCK_RADIUS

PERP_BOUNCE = 0.85   # perpendicular component retained on wall hit
PAR_BOUNCE = 0.95    # parallel component (tangential) retained
EPS = 1e-6


def _first_wall_hit(x, y, vx, vy):
    """Return (hx, hy, wall) for the first wall the puck hits."""
    if abs(vx) < EPS and abs(vy) < EPS:
        return x, y, None

    tx = None
    if vx > EPS:
        tx = (PUCK_MAX_X - x) / vx
    elif vx < -EPS:
        tx = (PUCK_MIN_X - x) / vx

    ty = None
    if vy > EPS:
        ty = (PUCK_MAX_Y - y) / vy
    elif vy < -EPS:
        ty = (PUCK_MIN_Y - y) / vy

    candidates = [(t, 'x') for t in (tx,) if t is not None and t >= 0]
    candidates += [(t, 'y') for t in (ty,) if t is not None and t >= 0]
    if not candidates:
        return x, y, None

    t, axis = min(candidates, key=lambda c: c[0])
    if axis == 'x':
        return (PUCK_MAX_X if vx > 0 else PUCK_MIN_X,
                y + vy * t,
                'r' if vx > 0 else 'l')
    else:
        return (x + vx * t,
                PUCK_MAX_Y if vy > 0 else PUCK_MIN_Y,
                'b' if vy > 0 else 't')


def predict_path(x, y, vx, vy):
    """
    Returns (B, C) where:
      B = (hx, hy, wall)   first wall hit
      C = (hx, hy, wall)   second wall hit after bounce
    Velocity after first bounce uses parallel/perpendicular coefficients.
    """
    B = _first_wall_hit(x, y, vx, vy)
    if B[2] is None:
        return B, B

    if B[2] in ('t', 'b'):
        bvx = vx * PAR_BOUNCE
        bvy = -vy * PERP_BOUNCE
    else:  # 'l' or 'r'
        bvx = -vx * PERP_BOUNCE
        bvy = vy * PAR_BOUNCE

    C = _first_wall_hit(B[0], B[1], bvx, bvy)
    return B, C


def position_at_time(x, y, vx, vy, t):
    """Where will the puck be at time t (one bounce max)? Returns (px, py)."""
    B = _first_wall_hit(x, y, vx, vy)
    if B[2] is None:
        return (x + vx * t, y + vy * t)

    # time to reach B
    if abs(vx) > abs(vy):
        tB = (B[0] - x) / vx if abs(vx) > EPS else 0.0
    else:
        tB = (B[1] - y) / vy if abs(vy) > EPS else 0.0

    if t <= tB:
        return (x + vx * t, y + vy * t)

    if B[2] in ('t', 'b'):
        bvx, bvy = vx * PAR_BOUNCE, -vy * PERP_BOUNCE
    else:
        bvx, bvy = -vx * PERP_BOUNCE, vy * PAR_BOUNCE

    dt = t - tB
    return (B[0] + bvx * dt, B[1] + bvy * dt)