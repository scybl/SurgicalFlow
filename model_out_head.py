import torch
import torch.nn as nn
import torch.nn.functional as F


class FutureTimelineModel(nn.Module):
    """
    Pure regression model (NO structural constraints inside forward)

    Input:
        cur_stage_idx : [B]     (1~7)
        ratio_input   : [B,1]   (remaining ratio of current stage)
        stage_order   : [B,7]   (0/1 mask)
        all_time      : [B,7]   (total time of each stage)

    Output:
        raw_future    : [B,7]   (unconstrained timeline prediction)
    """

    def __init__(
        self,
        hidden_dim=128,
        num_phases=7
    ):
        super().__init__()

        self.num_phases = num_phases

        # -------------------------------------------------
        # Feature encoder
        # -------------------------------------------------

        # input dimension:
        # cur_stage (1) + ratio (1) + stage_order (7) + all_time (7)
        in_dim = 1 + 1 + num_phases + num_phases

        self.mlp = nn.Sequential(

            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, num_phases)
        )

    def forward(
        self,
        cur_stage_idx,
        ratio_input,
        stage_order,
        all_time
    ):
        """
        Forward WITHOUT any constraint

        Only learn mapping:
        (stage context + ratio + time prior) -> raw timeline
        """

        # ---------------- normalize inputs ----------------

        # stage id normalize to [0,1]
        cur_stage_norm = cur_stage_idx.float().unsqueeze(1) / self.num_phases

        # concat features
        x = torch.cat(
            [
                cur_stage_norm,     # [B,1]
                ratio_input,        # [B,1]
                stage_order,        # [B,7]
                all_time            # [B,7]
            ],
            dim=1
        )

        # ---------------- regression ----------------

        raw_future = self.mlp(x)

        return raw_future


class ToolPredictionModel(nn.Module):
    """
    Pure MLP tool prediction model (same design style as FutureTimelineModel)

    Input:
        cur_stage_idx : [B]     (1~7)
        phase_remain  : [B,1]   (remaining time or ratio)

    Output:
        tool_logits   : [B,7]   (tool presence logits)
    """

    def __init__(
        self,
        hidden_dim=128,
        num_phases=7,
        num_tools=7
    ):
        super().__init__()

        self.num_phases = num_phases
        self.num_tools = num_tools

        # input dimension:
        # stage id (1) + remaining time (1)
        in_dim = 1 + 1

        self.mlp = nn.Sequential(

            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, num_tools)
        )

    def forward(
        self,
        cur_stage_idx,
        phase_remain
    ):
        """
        Forward WITHOUT structural constraints

        Mapping:
        (current phase + remaining time) -> tool presence
        """

        # -------- normalize inputs --------

        # stage id normalize to [0,1]
        cur_stage_norm = cur_stage_idx.float().unsqueeze(1) / self.num_phases

        remain = phase_remain.unsqueeze(1)

        # -------- concat --------

        x = torch.cat(
            [
                cur_stage_norm,   # [B,1]
                remain            # [B,1]
            ],
            dim=1
        )

        # -------- prediction --------

        tool_logits = self.mlp(x)

        return tool_logits