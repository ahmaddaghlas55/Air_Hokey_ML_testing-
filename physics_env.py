"""
Physics world. Pure pymunk — no pygame here, no rendering.
Coordinates: origin top-left, +x right, +y down (matches pygame).

Paddle control note: _drive_kinematic sets a body's VELOCITY toward a
target. Pymunk integrates that velocity during space.step(), which is
how the paddle actually moves. Call set_*_target() EVERY FRAME with a
fresh target — do not call it once and let the body drift.
"""

import pymunk

# --- table geometry (pixels) ---
TABLE_WIDTH = 1000
TABLE_HEIGHT = 500
WALL_THICKNESS = 10

# --- goal geometry ---
GOAL_HALF_HEIGHT = 80
GOAL_Y_TOP = TABLE_HEIGHT / 2 - GOAL_HALF_HEIGHT
GOAL_Y_BOT = TABLE_HEIGHT / 2 + GOAL_HALF_HEIGHT

# --- body radii (pixels) ---
PUCK_RADIUS = 15
PADDLE_RADIUS = 20

# --- physics parameters ---
PUCK_MASS = 1.0
PUCK_ELASTICITY = 0.95
PUCK_FRICTION = 0.02
WALL_ELASTICITY = 0.90
PADDLE_ELASTICITY = 0.90

# --- motion limits (pixels/sec) ---
PADDLE_MAX_SPEED = 2000.0
ROBOT_MAX_SPEED = 1200.0

# --- playable bounds (center of each paddle) ---
PLAYER_MIN_X = WALL_THICKNESS + PADDLE_RADIUS
PLAYER_MAX_X = TABLE_WIDTH / 2 - PADDLE_RADIUS
PLAYER_MIN_Y = WALL_THICKNESS + PADDLE_RADIUS
PLAYER_MAX_Y = TABLE_HEIGHT - WALL_THICKNESS - PADDLE_RADIUS

ROBOT_MIN_X = TABLE_WIDTH / 2 + PADDLE_RADIUS
ROBOT_MAX_X = TABLE_WIDTH - WALL_THICKNESS - PADDLE_RADIUS
ROBOT_MIN_Y = WALL_THICKNESS + PADDLE_RADIUS
ROBOT_MAX_Y = TABLE_HEIGHT - WALL_THICKNESS - PADDLE_RADIUS

# robot home = middle of far right edge
ROBOT_HOME = (ROBOT_MAX_X, TABLE_HEIGHT / 2)


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


class AirHockeyWorld:

    def __init__(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = 0.999

        self._add_walls()

        # --- puck ---
        self.puck_body = pymunk.Body(
            PUCK_MASS,
            pymunk.moment_for_circle(PUCK_MASS, 0, PUCK_RADIUS))
        self.puck_body.position = (TABLE_WIDTH / 2, TABLE_HEIGHT / 2)
        self.puck_body.velocity = (-300, 100)
        self.puck_shape = pymunk.Circle(self.puck_body, PUCK_RADIUS)
        self.puck_shape.elasticity = PUCK_ELASTICITY
        self.puck_shape.friction = PUCK_FRICTION
        self.space.add(self.puck_body, self.puck_shape)

        # --- player paddle (left half) ---
        self.player_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.player_body.position = (PLAYER_MIN_X + 60, TABLE_HEIGHT / 2)
        self.player_shape = pymunk.Circle(self.player_body, PADDLE_RADIUS)
        self.player_shape.elasticity = PADDLE_ELASTICITY
        self.player_shape.friction = 0.3
        self.space.add(self.player_body, self.player_shape)

        # --- robot paddle (right half) ---
        self.robot_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.robot_body.position = ROBOT_HOME
        self.robot_shape = pymunk.Circle(self.robot_body, PADDLE_RADIUS)
        self.robot_shape.elasticity = PADDLE_ELASTICITY
        self.robot_shape.friction = 0.3
        self.space.add(self.robot_body, self.robot_shape)

    # ---------- walls ----------

    def _add_walls(self):
        static = self.space.static_body
        segs = [
            ((0, 0), (TABLE_WIDTH, 0)),
            ((TABLE_WIDTH, TABLE_HEIGHT), (0, TABLE_HEIGHT)),
            ((0, 0), (0, GOAL_Y_TOP)),
            ((0, GOAL_Y_BOT), (0, TABLE_HEIGHT)),
            ((TABLE_WIDTH, 0), (TABLE_WIDTH, GOAL_Y_TOP)),
            ((TABLE_WIDTH, GOAL_Y_BOT), (TABLE_WIDTH, TABLE_HEIGHT)),
        ]
        for a, b in segs:
            seg = pymunk.Segment(static, a, b, WALL_THICKNESS)
            seg.elasticity = WALL_ELASTICITY
            seg.friction = 0.1
            self.space.add(seg)

    # ---------- paddle driving ----------

    def _drive_kinematic(self, body, target, dt, max_speed):
        """
        Move a kinematic body toward `target` by setting its velocity.
        Pymunk integrates the velocity during space.step().

        Important: call this EVERY FRAME with a fresh target. If you
        call it once and then step many frames, the paddle keeps moving
        at the stale velocity and overshoots wildly.

        Velocity is capped at max_speed, but we also cap the per-frame
        displacement at `dist` so the paddle can't overshoot the target.
        """
        tx = clamp(target[0], 0, TABLE_WIDTH)
        ty = clamp(target[1], 0, TABLE_HEIGHT)

        px, py = body.position
        dx, dy = tx - px, ty - py
        dist = (dx * dx + dy * dy) ** 0.5

        if dist < 0.5:
            body.velocity = (0, 0)
            return

        # required velocity to reach target in this dt
        vx = dx / dt
        vy = dy / dt
        speed = (vx * vx + vy * vy) ** 0.5

        # never exceed the paddle's max speed
        if speed > max_speed:
            scale = max_speed / speed
            vx *= scale
            vy *= scale

        # never move further than `dist` in one frame (prevents overshoot
        # when max_speed * dt > dist)
        max_disp = dist
        disp = (vx * dt) ** 2 + (vy * dt) ** 2
        disp = disp ** 0.5
        if disp > max_disp and disp > 0:
            scale = max_disp / disp
            vx *= scale
            vy *= scale

        body.velocity = (vx, vy)

    def set_player_target(self, x, y, dt):
        x = clamp(x, PLAYER_MIN_X, PLAYER_MAX_X)
        y = clamp(y, PLAYER_MIN_Y, PLAYER_MAX_Y)
        self._drive_kinematic(self.player_body, (x, y), dt, PADDLE_MAX_SPEED)

    def set_robot_target(self, x, y, dt):
        x = clamp(x, ROBOT_MIN_X, ROBOT_MAX_X)
        y = clamp(y, ROBOT_MIN_Y, ROBOT_MAX_Y)
        self._drive_kinematic(self.robot_body, (x, y), dt, ROBOT_MAX_SPEED)

    # ---------- physics ----------

    def step(self, dt):
        self.space.step(dt)

    # ---------- game events ----------

    def check_goal(self):
        px, py = self.puck_body.position
        if not (GOAL_Y_TOP <= py <= GOAL_Y_BOT):
            return None
        if px < 0:
            return 'robot'
        if px > TABLE_WIDTH:
            return 'player'
        return None

    def serve(self, toward):
        self.puck_body.position = (TABLE_WIDTH / 2, TABLE_HEIGHT / 2)
        if toward == 'player':
            self.puck_body.velocity = (-250, 80)
        elif toward == 'robot':
            self.puck_body.velocity = (250, -80)
        else:
            self.puck_body.velocity = (0, 0)
        self.player_body.position = (PLAYER_MIN_X + 60, TABLE_HEIGHT / 2)
        self.player_body.velocity = (0, 0)
        self.robot_body.position = ROBOT_HOME
        self.robot_body.velocity = (0, 0)

    def reset_puck(self, vx=-300, vy=100):
        self.puck_body.position = (TABLE_WIDTH / 2, TABLE_HEIGHT / 2)
        self.puck_body.velocity = (vx, vy)