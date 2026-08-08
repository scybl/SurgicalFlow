"""Workflow schema for Cholec80-style surgical phase and tool labels."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Phase:
    phase_id: int
    name: str
    group: str
    importance: float = 1.0


PHASES = (
    Phase(1, "Preparation", "access", 1.00),
    Phase(2, "CalotTriangleDissection", "dissection", 1.35),
    Phase(3, "ClippingCutting", "sealing", 1.45),
    Phase(4, "GallbladderDissection", "dissection", 1.25),
    Phase(5, "GallbladderPackaging", "extraction_cleanup", 1.10),
    Phase(6, "CleaningCoagulation", "extraction_cleanup", 1.20),
    Phase(7, "GallbladderRetraction", "dissection", 1.20),
)

TOOLS = (
    "Grasper",
    "Bipolar",
    "Hook",
    "Scissors",
    "Clipper",
    "Irrigator",
    "SpecimenBag",
)

PHASE2ID = {phase.name: phase.phase_id for phase in PHASES}
ZERO_BASED_PHASE2ID = {phase.name: phase.phase_id - 1 for phase in PHASES}
ID2PHASE = {phase.phase_id: phase.name for phase in PHASES}
PHASE_IMPORTANCE = {phase.phase_id: phase.importance for phase in PHASES}
PHASE_GROUPS = tuple(dict.fromkeys(phase.group for phase in PHASES))
PHASE_GROUP2ID = {group: index for index, group in enumerate(PHASE_GROUPS)}
PHASE_ID_TO_GROUP_ID = {
    phase.phase_id: PHASE_GROUP2ID[phase.group] for phase in PHASES
}
PHASE_GROUP_MEMBERS = tuple(
    tuple(phase.phase_id for phase in PHASES if phase.group == group)
    for group in PHASE_GROUPS
)

NUM_PHASES = len(PHASES)
NUM_TOOLS = len(TOOLS)


def phase_group_id(phase_id: int) -> int:
    """Return the coarse workflow group id for a phase id."""
    if phase_id not in PHASE_ID_TO_GROUP_ID:
        raise ValueError(f"Unknown phase id: {phase_id}")
    return PHASE_ID_TO_GROUP_ID[phase_id]


def phase_transition_distance(source_phase: int, target_phase: int) -> int:
    """Return ordinal workflow distance between two phase ids."""
    if source_phase not in ID2PHASE:
        raise ValueError(f"Unknown source phase id: {source_phase}")
    if target_phase not in ID2PHASE:
        raise ValueError(f"Unknown target phase id: {target_phase}")
    return abs(source_phase - target_phase)


def balanced_class_weights(
    labels: Iterable[int],
    class_ids: Iterable[int],
    *,
    smoothing: float = 1.0,
    max_weight: float = 5.0,
) -> dict[int, float]:
    """Return inverse-frequency class weights normalized around 1.0."""
    class_ids = tuple(class_ids)
    counts = Counter(labels)
    total = sum(counts[class_id] for class_id in class_ids)
    if total == 0:
        return {class_id: 1.0 for class_id in class_ids}

    raw_weights = {
        class_id: total / (len(class_ids) * (counts[class_id] + smoothing))
        for class_id in class_ids
    }
    clipped = {
        class_id: min(weight, max_weight)
        for class_id, weight in raw_weights.items()
    }
    mean_weight = sum(clipped.values()) / len(clipped)
    return {
        class_id: weight / mean_weight
        for class_id, weight in clipped.items()
    }
