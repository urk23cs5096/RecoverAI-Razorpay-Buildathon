"""Agent lifecycle states and transition definitions."""

from enum import Enum
from typing import Set, Dict


class AgentState(str, Enum):
    DETECTED = "DETECTED"
    DIAGNOSED = "DIAGNOSED"
    DECIDED = "DECIDED"
    ACTION_SCHEDULED = "ACTION_SCHEDULED"
    ACTION_EXECUTED = "ACTION_EXECUTED"
    VERIFIED = "VERIFIED"
    RECOVERED = "RECOVERED"
    FAILED_TERMINAL = "FAILED_TERMINAL"
    ESCALATED = "ESCALATED"


# Valid state machine transitions
ALLOWED_STATE_TRANSITIONS: Dict[AgentState, Set[AgentState]] = {
    AgentState.DETECTED: {
        AgentState.DIAGNOSED,
        AgentState.DECIDED,
        AgentState.FAILED_TERMINAL,
        AgentState.ESCALATED,
    },
    AgentState.DIAGNOSED: {
        AgentState.DECIDED,
        AgentState.ACTION_SCHEDULED,
        AgentState.ACTION_EXECUTED,
        AgentState.ESCALATED,
        AgentState.FAILED_TERMINAL,
    },
    AgentState.DECIDED: {
        AgentState.ACTION_SCHEDULED,
        AgentState.ACTION_EXECUTED,
        AgentState.ESCALATED,
        AgentState.FAILED_TERMINAL,
    },
    AgentState.ACTION_SCHEDULED: {
        AgentState.ACTION_EXECUTED,
        AgentState.VERIFIED,
        AgentState.RECOVERED,
        AgentState.FAILED_TERMINAL,
        AgentState.ESCALATED,
    },
    AgentState.ACTION_EXECUTED: {
        AgentState.VERIFIED,
        AgentState.RECOVERED,
        AgentState.FAILED_TERMINAL,
        AgentState.ESCALATED,
    },
    AgentState.VERIFIED: {
        AgentState.RECOVERED,
        AgentState.FAILED_TERMINAL,
        AgentState.ESCALATED,
        AgentState.DETECTED,
    },
    AgentState.RECOVERED: set(),  # Terminal
    AgentState.FAILED_TERMINAL: set(),  # Terminal
    AgentState.ESCALATED: set(),  # Terminal (handed to human)
}
