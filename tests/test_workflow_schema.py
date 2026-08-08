import pytest

from workflow_schema import (
    NUM_PHASES,
    NUM_TOOLS,
    PHASE2ID,
    PHASE_GROUP_MEMBERS,
    balanced_class_weights,
    phase_group_id,
    phase_transition_distance,
)


def test_phase_and_tool_schema_sizes_match_cholec80_labels():
    assert NUM_PHASES == 7
    assert NUM_TOOLS == 7
    assert PHASE2ID["Preparation"] == 1
    assert PHASE2ID["GallbladderRetraction"] == 7


def test_phase_groups_cover_each_phase_once():
    members = sorted(phase_id for group in PHASE_GROUP_MEMBERS for phase_id in group)

    assert members == list(range(1, NUM_PHASES + 1))
    assert phase_group_id(PHASE2ID["CalotTriangleDissection"]) == phase_group_id(
        PHASE2ID["GallbladderDissection"]
    )
    assert phase_group_id(PHASE2ID["ClippingCutting"]) != phase_group_id(
        PHASE2ID["Preparation"]
    )


def test_phase_transition_distance_uses_workflow_order():
    assert phase_transition_distance(1, 1) == 0
    assert phase_transition_distance(1, 7) == 6
    assert phase_transition_distance(4, 2) == 2

    with pytest.raises(ValueError):
        phase_transition_distance(0, 1)


def test_balanced_class_weights_emphasize_rare_labels():
    weights = balanced_class_weights([1, 1, 1, 2], class_ids=[1, 2])

    assert weights[2] > weights[1]
    assert pytest.approx(sum(weights.values()) / len(weights)) == 1.0
