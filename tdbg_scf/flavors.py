"""Explicit spin/valley flavor conventions for correlated calculations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Flavor:
    index: int
    valley: int
    spin: int
    name: str


FLAVORS: Tuple[Flavor, ...] = (
    Flavor(0, +1, +1, "K_up"),
    Flavor(1, -1, +1, "Kp_up"),
    Flavor(2, +1, -1, "K_down"),
    Flavor(3, -1, -1, "Kp_down"),
)


def flavor_names() -> list[str]:
    return [f.name for f in FLAVORS]
