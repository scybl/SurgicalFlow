"""Workflow-aware training losses for SurgicalFlow models."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from workflow_schema import (
    NUM_PHASES,
    NUM_TOOLS,
    PHASE_GROUP_MEMBERS,
    PHASE_ID_TO_GROUP_ID,
    PHASE_IMPORTANCE,
    balanced_class_weights,
)


def phase_class_weight_tensor(samples: Iterable[dict], device):
    """Build class-balanced phase weights with workflow importance factors."""
    import torch

    labels = [sample["phase"] for sample in samples if sample.get("phase", 0) > 0]
    balanced = balanced_class_weights(labels, range(1, NUM_PHASES + 1))
    values = [0.0]
    for phase_id in range(1, NUM_PHASES + 1):
        values.append(balanced[phase_id] * PHASE_IMPORTANCE[phase_id])

    phase_mean = sum(values[1:]) / NUM_PHASES
    values = [0.0] + [value / phase_mean for value in values[1:]]
    return torch.tensor(values, dtype=torch.float32, device=device)


def phase_group_targets(phase_gt):
    """Map fine phase ids to coarse workflow group ids."""
    import torch

    mapping = torch.zeros(NUM_PHASES + 1, dtype=torch.long, device=phase_gt.device)
    for phase_id, group_id in PHASE_ID_TO_GROUP_ID.items():
        mapping[phase_id] = group_id
    return mapping[phase_gt.long()]


def phase_group_logits(phase_logits):
    """Aggregate fine phase logits into coarse workflow group logits."""
    import torch

    grouped_logits = []
    for members in PHASE_GROUP_MEMBERS:
        member_logits = phase_logits[:, list(members)]
        grouped_logits.append(torch.logsumexp(member_logits, dim=1))
    return torch.stack(grouped_logits, dim=1)


def phase_group_loss(phase_logits, phase_gt):
    """Coarse workflow-group loss derived from fine phase logits."""
    import torch.nn.functional as F

    valid = phase_gt != 0
    if valid.sum().item() == 0:
        return phase_logits.sum() * 0.0

    return F.cross_entropy(
        phase_group_logits(phase_logits[valid]),
        phase_group_targets(phase_gt[valid]),
    )


def phase_order_loss(phase_logits, phase_gt):
    """Penalize phase predictions by ordinal distance in the workflow."""
    import torch
    import torch.nn.functional as F

    valid = phase_gt != 0
    if valid.sum().item() == 0:
        return phase_logits.sum() * 0.0

    probs = torch.softmax(phase_logits[valid, 1:], dim=1)
    phase_indices = torch.arange(1, NUM_PHASES + 1, device=phase_logits.device)
    expected_phase = (probs * phase_indices.float()).sum(dim=1)
    return F.smooth_l1_loss(expected_phase, phase_gt[valid].float())


def timeline_horizon_weights(
    stage_order,
    cur_stage_position,
    *,
    current_stage_weight: float = 1.5,
    future_decay: float = 0.85,
):
    """Weight timeline loss by current and near-future workflow positions."""
    import torch

    batch_size, num_positions = stage_order.shape
    positions = torch.arange(num_positions, device=stage_order.device).unsqueeze(0)
    current_position = (cur_stage_position.long() - 1).clamp(min=0).unsqueeze(1)
    horizon = (positions - current_position).clamp(min=0).float()
    weights = future_decay ** horizon
    weights = weights * (stage_order > 0).float()

    current_mask = positions == current_position
    weights = torch.where(
        current_mask,
        weights * current_stage_weight,
        weights,
    )
    normalizer = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
    return weights * (num_positions / normalizer)


def weighted_timeline_loss(
    prediction,
    target,
    stage_order,
    cur_stage_position,
    *,
    current_stage_weight: float = 1.5,
    future_decay: float = 0.85,
):
    """SmoothL1 timeline loss with workflow-position weights."""
    import torch.nn.functional as F

    per_element = F.smooth_l1_loss(prediction, target, reduction="none")
    weights = timeline_horizon_weights(
        stage_order,
        cur_stage_position,
        current_stage_weight=current_stage_weight,
        future_decay=future_decay,
    )
    return (per_element * weights).sum() / weights.sum().clamp(min=1.0)


def tool_pos_weight_tensor(samples: Iterable[dict], device):
    """Build positive-class weights for multi-label tool prediction."""
    import torch

    counts = Counter()
    total = 0
    for sample in samples:
        tool_label = sample.get("tool")
        if tool_label is None:
            continue
        total += 1
        for index, value in enumerate(tool_label):
            if float(value) > 0:
                counts[index] += 1

    if total == 0:
        return torch.ones(NUM_TOOLS, dtype=torch.float32, device=device)

    values = []
    for index in range(NUM_TOOLS):
        positive = counts[index]
        negative = total - positive
        if positive == 0:
            values.append(1.0)
        else:
            values.append(max(0.25, min(8.0, negative / positive)))

    return torch.tensor(values, dtype=torch.float32, device=device)
