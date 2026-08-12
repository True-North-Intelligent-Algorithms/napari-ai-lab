"""How ready a segmenter is to run, in more than two states.

A segmenter used to answer one question -- are my dependencies importable --
and the UI drew that green or red. Running ops in their own environments adds
a third possibility that is neither: the op will run, but somewhere else, and
possibly only after building an environment that takes minutes and gigabytes.

Telling those apart matters, because they mean different things to whoever is
about to press the button:

    AVAILABLE   runs here, now
    READY       runs now, in another environment that already exists
    WILL_BUILD  runs, but the first use builds an environment first
    UNAVAILABLE cannot run; install something, or choose another segmenter

See docs/spec/0003-optional-dependencies.md and
docs/spec/0004-first-scikit-ops-segmenter.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class State(Enum):
    """The four states, in the order a user would prefer them."""

    AVAILABLE = "available"
    READY = "ready"
    WILL_BUILD = "will_build"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class Availability:
    """A state and something to show the user.

    Truthy for everything except UNAVAILABLE, so the older boolean question --
    "can this run at all" -- still works on it.
    """

    state: State
    message: str = ""

    def __bool__(self) -> bool:
        return self.state is not State.UNAVAILABLE


def available(message: str = "Dependencies available") -> Availability:
    return Availability(State.AVAILABLE, message)


def ready(message: str) -> Availability:
    return Availability(State.READY, message)


def will_build(message: str) -> Availability:
    return Availability(State.WILL_BUILD, message)


def unavailable(
    message: str = "Dependencies not available — install required packages to use this segmenter",
) -> Availability:
    return Availability(State.UNAVAILABLE, message)
