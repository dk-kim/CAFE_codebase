# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_code: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_code: This is the relative weight of the L2 error of the membership code in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_code = cost_code
        assert cost_class != 0 or cost_code != 0, "all costs cant be 0"

    # membership cost
    def _get_cost_code(self, sim, targets_membership):
        cost_code = torch.cdist(sim, targets_membership, p=2)
        return cost_code

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries, num_boxes = outputs["membership"].shape

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_activities"].flatten(0, 1).softmax(-1)  # [bs * t * num_queries, num_classes]

        sim = outputs["membership"]

        if "dummy_idx" in targets[0].keys():
            sim_dummy = []
            for batch_idx in range(bs):
                dummy_idx = targets[batch_idx]["dummy_idx"].squeeze()
                sim_batch = sim[batch_idx] * dummy_idx
                sim_dummy.append(sim_batch.unsqueeze(0))
            sim = torch.cat(sim_dummy, dim=0)

        sim = sim.flatten(0, 1)

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["activities"] for v in targets], dim=1).reshape(-1)
        targets_membership = torch.cat([v["members"] for v in targets], dim=1).reshape(-1, num_boxes)

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]
        # If cost code is similarity, it should be minus sign
        cost_code = self._get_cost_code(sim, targets_membership)

        # Final cost matrix
        C = self.cost_class * cost_class + self.cost_code * cost_code
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(k) for v in targets for k in v["activities"]]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_group_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_group_class, cost_code=args.set_cost_membership)
