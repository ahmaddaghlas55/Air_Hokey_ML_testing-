"""
Abstraction contract. Strategy code talks to this, not to pygame or motors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TableState:
    puck_pos: tuple           # (x, y)
    puck_vel: tuple           # (vx, vy)
    robot_pos: tuple          # (x, y)
    robot_vel: tuple          # (vx, vy)
    opponent_pos: tuple       # (x, y)
    opponent_vel: tuple       # (vx, vy)


class TableBackend(ABC):

    @abstractmethod
    def read_state(self) -> TableState:
        ...

    @abstractmethod
    def set_robot_target(self, x: float, y: float) -> None:
        ...

    @abstractmethod
    def step(self, dt: float) -> None:
        ...

    @abstractmethod
    def is_running(self) -> bool:
        ...