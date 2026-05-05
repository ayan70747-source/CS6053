"""Rational agent model for warehouse MAPF."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class AgentState(str, Enum):
    IDLE = "IDLE"
    MOVING_TO_PICK = "MOVING_TO_PICK"
    MOVING_TO_DELIVERY = "MOVING_TO_DELIVERY"


@dataclass
class RationalAgent:
    agent_id: int
    start_pos: Tuple[int, int]
    current_pos: Tuple[int, int] = field(init=False)
    state: AgentState = field(default=AgentState.IDLE)
    pick_target: Optional[Tuple[int, int]] = field(default=None)
    delivery_target: Optional[Tuple[int, int]] = field(default=None)
    planned_path: List[Tuple[int, int]] = field(default_factory=list)
    path_to_pick: List[Tuple[int, int]] = field(default_factory=list)
    path_to_delivery: List[Tuple[int, int]] = field(default_factory=list)
    total_steps_taken: int = field(default=0)

    def __post_init__(self) -> None:
        self.current_pos = self.start_pos

    def assign_task(self, pick_target: Tuple[int, int], delivery_target: Tuple[int, int]) -> None:
        self.pick_target = pick_target
        self.delivery_target = delivery_target
        self.state = AgentState.MOVING_TO_PICK

    def set_paths(
        self,
        to_pick: List[Tuple[int, int]],
        to_delivery: List[Tuple[int, int]],
    ) -> None:
        self.path_to_pick = to_pick
        self.path_to_delivery = to_delivery
        self.planned_path = to_pick + to_delivery[1:] if to_delivery else to_pick

    def update_state_from_position(self) -> None:
        if self.pick_target and self.current_pos == self.pick_target and self.state == AgentState.MOVING_TO_PICK:
            self.state = AgentState.MOVING_TO_DELIVERY
        if self.delivery_target and self.current_pos == self.delivery_target:
            self.state = AgentState.IDLE
