"""
pygame rendering + input. Reads mouse, drives player paddle.
Calls set_robot_target with the strategy's desired position.
Handles scoring, serve pauses, winner, and R-to-restart.
"""

import time
import pygame

from interface import TableBackend, TableState
from physics_env import (
    AirHockeyWorld, TABLE_WIDTH, TABLE_HEIGHT,
    PUCK_RADIUS, PADDLE_RADIUS, WALL_THICKNESS,
    PLAYER_MIN_X, PLAYER_MAX_X, PLAYER_MIN_Y, PLAYER_MAX_Y,
    ROBOT_HOME, GOAL_Y_TOP, GOAL_Y_BOT,
)
from predictor import predict_path


WIN_SCORE = 7
SERVE_PAUSE = 1.5     # seconds


class SimBackend(TableBackend):

    def __init__(self, on_player_hit=None):
        pygame.init()
        pygame.font.init()

        self.screen = pygame.display.set_mode((TABLE_WIDTH, TABLE_HEIGHT))
        pygame.display.set_caption("Air hockey - mouse")
        self.clock = pygame.time.Clock()

        self.world = AirHockeyWorld()
        self._running = True

        self.on_player_hit = on_player_hit
        self._last_player_dist = None
        self._robot_target = ROBOT_HOME

        # scoring / serve state
        self.player_score = 0
        self.robot_score = 0
        self.serve_cooldown = 0.0
        self._last_scorer = None
        self.winner = None

        # fonts
        self.font_small = pygame.font.SysFont(None, 40)
        self.font_big = pygame.font.SysFont(None, 80)

    # ---------- interface ----------

    def read_state(self):
        w = self.world
        return TableState(
            puck_pos=tuple(w.puck_body.position),
            puck_vel=tuple(w.puck_body.velocity),
            robot_pos=tuple(w.robot_body.position),
            robot_vel=tuple(w.robot_body.velocity),
            opponent_pos=tuple(w.player_body.position),
            opponent_vel=tuple(w.player_body.velocity),
        )

    def set_robot_target(self, x, y):
        self._robot_target = (x, y)

    def is_running(self):
        return self._running

    # ---------- main step ----------

    def step(self, dt):
        # --- input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._running = False
                elif event.key == pygame.K_r:
                    self._restart_match()

        # --- game over: freeze ---
        if self.winner is not None:
            self._draw()
            return

        # --- serve pause: count down, then serve ---
        if self.serve_cooldown > 0:
            self.serve_cooldown -= dt
            if self.serve_cooldown <= 0:
                toward = 'player' if self._last_scorer == 'robot' else 'robot'
                self.world.serve(toward)
            self._draw()
            return

        # --- normal frame ---
        mx, my = pygame.mouse.get_pos()
        self.world.set_player_target(mx, my, dt)

        rx, ry = self._robot_target
        self.world.set_robot_target(rx, ry, dt)

        self._log_player_hit_if_any()
        self.world.step(dt)

        # --- goal check ---
        goal = self.world.check_goal()
        if goal is not None:
            self._handle_goal(goal)

        self._draw()

    # ---------- game logic ----------

    def _handle_goal(self, scorer):
        if scorer == 'player':
            self.player_score += 1
        else:
            self.robot_score += 1
        self._last_scorer = scorer

        if self.player_score >= WIN_SCORE:
            self.winner = 'player'
        elif self.robot_score >= WIN_SCORE:
            self.winner = 'robot'
        else:
            self.serve_cooldown = SERVE_PAUSE
            # hide puck off-screen during the pause
            self.world.puck_body.position = (-500, -500)
            self.world.puck_body.velocity = (0, 0)

    def _restart_match(self):
        self.player_score = 0
        self.robot_score = 0
        self.winner = None
        self.serve_cooldown = 0.0
        self._last_scorer = None
        self.world.serve('player')

    # ---------- logging ----------

    def _log_player_hit_if_any(self):
        if self.on_player_hit is None:
            return
        w = self.world
        dx = w.puck_body.position.x - w.player_body.position.x
        dy = w.puck_body.position.y - w.player_body.position.y
        dist = (dx * dx + dy * dy) ** 0.5
        rng = PUCK_RADIUS + PADDLE_RADIUS + 2
        just = (dist <= rng and self._last_player_dist is not None
                and self._last_player_dist > rng)
        if just:
            self.on_player_hit({
                "t": time.time(),
                "puck_pos": tuple(w.puck_body.position),
                "puck_vel_after": tuple(w.puck_body.velocity),
                "player_pos": tuple(w.player_body.position),
                "player_vel": tuple(w.player_body.velocity),
            })
        self._last_player_dist = dist

    # ---------- rendering ----------

    def _draw(self):
        s = self.screen
        s.fill((15, 20, 30))

        # player half slightly darker
        pygame.draw.rect(s, (22, 28, 40),
                         (0, 0, TABLE_WIDTH // 2, TABLE_HEIGHT))

        # center divider
        pygame.draw.line(s, (90, 95, 110),
                         (TABLE_WIDTH // 2, 0),
                         (TABLE_WIDTH // 2, TABLE_HEIGHT), 2)

        # border
        pygame.draw.rect(s, (70, 80, 100),
                         (0, 0, TABLE_WIDTH, TABLE_HEIGHT),
                         WALL_THICKNESS)

        # goal zones — green on left (robot's goal), blue on right (your goal)
        pygame.draw.rect(s, (60, 160, 80),
                         (0, GOAL_Y_TOP, WALL_THICKNESS,
                          GOAL_Y_BOT - GOAL_Y_TOP))
        pygame.draw.rect(s, (90, 120, 220),
                         (TABLE_WIDTH - WALL_THICKNESS, GOAL_Y_TOP,
                          WALL_THICKNESS, GOAL_Y_BOT - GOAL_Y_TOP))

        w = self.world

        # puck
        px, py = w.puck_body.position
        if px >= 0 and px <= TABLE_WIDTH and py >= 0 and py <= TABLE_HEIGHT:
            pygame.draw.circle(s, (230, 230, 230),
                               (int(px), int(py)), PUCK_RADIUS)

            # trajectory preview (one wall bounce)
            vx, vy = w.puck_body.velocity
            if abs(vx) + abs(vy) > 5:
                B, C = predict_path(px, py, vx, vy)
                if B[2] is not None:
                    pygame.draw.line(s, (255, 220, 80),
                                     (px, py), (B[0], B[1]), 2)
                    if C[2] is not None:
                        pygame.draw.line(s, (255, 220, 80),
                                         (B[0], B[1]), (C[0], C[1]), 2)

        # robot paddle
        rx, ry = w.robot_body.position
        pygame.draw.circle(s, (90, 160, 250),
                           (int(rx), int(ry)), PADDLE_RADIUS)

        # player paddle
        ox, oy = w.player_body.position
        pygame.draw.circle(s, (240, 120, 90),
                           (int(ox), int(oy)), PADDLE_RADIUS)

        # crosshair at mouse
        mx, my = pygame.mouse.get_pos()
        pygame.draw.line(s, (255, 255, 255), (mx - 10, my), (mx + 10, my), 1)
        pygame.draw.line(s, (255, 255, 255), (mx, my - 10), (mx, my + 10), 1)

        # HUD: score
        score_text = self.font_small.render(
            f"YOU  {self.player_score}   -   {self.robot_score}  ROBOT",
            True, (220, 220, 220))
        s.blit(score_text,
               (TABLE_WIDTH // 2 - score_text.get_width() // 2, 8))

        # HUD: serve pause
        if self.serve_cooldown > 0:
            msg = self.font_small.render("GOAL!", True, (255, 200, 100))
            s.blit(msg, (TABLE_WIDTH // 2 - msg.get_width() // 2, 60))

        # HUD: winner
        if self.winner is not None:
            label = "PLAYER WINS" if self.winner == 'player' else "ROBOT WINS"
            msg = self.font_big.render(label, True, (255, 220, 120))
            s.blit(msg, (TABLE_WIDTH // 2 - msg.get_width() // 2,
                         TABLE_HEIGHT // 2 - msg.get_height() // 2 - 20))
            hint = self.font_small.render("press R to restart",
                                          True, (200, 200, 200))
            s.blit(hint, (TABLE_WIDTH // 2 - hint.get_width() // 2,
                          TABLE_HEIGHT // 2 + 40))

        pygame.display.flip()