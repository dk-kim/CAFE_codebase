# ------------------------------------------------------------------------
# Modified from HOTR (https://github.com/kakaobrain/HOTR)
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
import torch.nn.functional as F
import copy
import numpy as np

from torch import nn

from util import box_ops
from util.misc import (accuracy, get_world_size, is_dist_avail_and_initialized)


class SetCriterion(nn.Module):
    def __init__(self, num_classes, weight_dict, eos_coef, losses, group_losses=None,
                 group_matcher=None, args=None):
        """ Create the criterion.
        Parameters:
        num_classes: number of object categories, omitting the special no-object category
        weight_dict: dict containing as key the names of the losses and as values their relative weight.
        eos_coef: relative classification weight applied to the no-group activity class category
        losses: list of all the losses to be applied. See get_loss for list of available losses.
        group_losses: list of all the group losses to be applied. See get_group_loss for list of available group losses.
        group_matcher: module able to compute a matching between targets and predictions
        """
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.losses = losses
        self.eos_coef = eos_coef

        self.group_losses = group_losses
        self.group_matcher = group_matcher

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

        empty_group_weight = torch.ones(self.num_classes + 1)
        empty_group_weight[-1] = args.group_eos_coef
        self.register_buffer('empty_group_weight', empty_group_weight)

        self.num_boxes = args.num_boxes

        # option
        self.temperature = args.temperature

    #######################################################################################################################
    # * Individual Losses
    #######################################################################################################################
    def loss_labels(self, outputs, targets, num_boxes, log=True):
        """Individual action classification loss (NLL)"""
        assert 'pred_actions' in outputs
        src_logits = outputs['pred_actions']
        target_classes = torch.cat([v["actions"] for v in targets], dim=0)

        loss_ce = 0.0

        src_logits_log = None
        tgt_classes_log = None

        for batch_idx in range(src_logits.shape[0]):
            dummy_idx = targets[batch_idx]["dummy_idx"].squeeze()
            non_dummy_idx = dummy_idx.nonzero(as_tuple=True)
            src_logit = src_logits[batch_idx][non_dummy_idx].unsqueeze(0)
            target_class = target_classes[batch_idx][non_dummy_idx].unsqueeze(0)
            loss_ce += F.cross_entropy(src_logit.transpose(1, 2), target_class, self.empty_weight)

            if src_logits_log is None:
                src_logits_log = src_logit
                tgt_classes_log = target_class
            else:
                src_logits_log = torch.cat([src_logits_log.squeeze(), src_logit.squeeze()], dim=0)
                tgt_classes_log = torch.cat([tgt_classes_log.squeeze(), target_class.squeeze()], dim=0)

        loss_ce /= src_logits.shape[0]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits_log, tgt_classes_log)[0]

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, num_boxes):
        pred_logits = outputs['pred_actions']
        device = pred_logits.device
        # tgt_lengths = torch.as_tensor([len(v["actions"]) for v in targets], device=device)
        tgt_lengths = torch.as_tensor([len(k) for v in targets for k in v["actions"]], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    #######################################################################################################################
    # * Group Losses
    #######################################################################################################################
    def loss_group_labels(self, outputs, targets, group_indices, log=True):
        """ Group activity classification loss (NLL)"""
        assert 'pred_activities' in outputs
        src_logits = outputs['pred_activities']

        idx = self._get_src_permutation_idx(group_indices)
        flatten_targets = [u for t in targets for u in t["activities"]]
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(flatten_targets, group_indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64,
                                    device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_group_weight)
        losses = {'loss_group_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['group_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    @torch.no_grad()
    def loss_group_cardinality(self, outputs, targets, group_indices):
        pred_logits = outputs['pred_activities']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(k) for v in targets for k in v["activities"]], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'group_cardinality_error': card_err}
        return losses

    def loss_group_code(self, outputs, targets, group_indices, log=True):
        """Membership loss"""
        sim = outputs['membership']

        idx = self._get_src_permutation_idx(group_indices)

        # Binary cross entropy loss
        flatten_targets = [u for t in targets for u in t["members"]]

        # target_members_o = torch.cat([t[J] for t, (_, J) in zip(flatten_targets, group_indices)]).type(torch.FloatTensor).to(sim.device)
        target_members_o = torch.cat([t[J] for t, (_, J) in zip(flatten_targets, group_indices)])
        target_members = torch.full(sim.shape, 0.0, dtype=torch.float, device=sim.device)
        target_members[idx] = target_members_o

        loss_membership = 0.0
        for batch_idx in range(sim.shape[0]):
            dummy_idx = targets[batch_idx]["dummy_idx"].squeeze()
            non_dummy_idx = dummy_idx.nonzero(as_tuple=True)
            sim_batch = sim[batch_idx].transpose(0, 1)[non_dummy_idx].transpose(0, 1).unsqueeze(0)

            target_members_batch = target_members[batch_idx].transpose(0, 1)[non_dummy_idx].transpose(0, 1).unsqueeze(0)

            loss_membership += F.binary_cross_entropy(sim_batch, target_members_batch)
        loss_membership /= sim.shape[0]

        losses = {'loss_group_code': loss_membership}
        return losses

    def loss_group_consistency(self, outputs, targets, group_indices):
        """Group consistency loss"""
        actor_embeds = outputs['actor_embeddings']

        consistency_loss = 0.0

        for batch_idx in range(actor_embeds.shape[0]):
            membership = targets[batch_idx]["membership"][0]
            actor_embed = actor_embeds[batch_idx]                           # [n, f]

            cos = nn.CosineSimilarity(dim=-1)
            sim = cos(actor_embed.unsqueeze(1), actor_embed.unsqueeze(0)) / self.temperature

            dummy_idx = targets[batch_idx]["dummy_idx"].squeeze()
            non_dummy_idx = dummy_idx.nonzero(as_tuple=True)

            N = len(non_dummy_idx[0])

            non_dummy_membership = membership[non_dummy_idx]

            group_count = 0

            for actor_idx in range(N):
                group_id = non_dummy_membership[actor_idx]

                if group_id != -1:
                    positive_idx = (non_dummy_membership == group_id).nonzero(as_tuple=True)
                    positive_idx = list(positive_idx[0])
                    positive_idx.remove(actor_idx)
                    positive_idx = [tuple(positive_idx)]
                    positive_samples = sim[actor_idx][positive_idx]

                    negative_idx = (non_dummy_membership != group_id).nonzero(as_tuple=True)
                    negative_samples = sim[actor_idx][negative_idx]

                    nominator = torch.exp(positive_samples)
                    denominator = torch.exp(torch.cat((positive_samples, negative_samples)))
                    loss_partial = -torch.log(torch.sum(nominator) / torch.sum(denominator))
                    group_count += 1

                    consistency_loss += loss_partial

            consistency_loss /= group_count

        consistency_loss /= actor_embeds.shape[0]
        losses = {'loss_consistency': consistency_loss}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    # *****************************************************************************
    # >>> DETR Losses
    def get_loss(self, loss, outputs, targets, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, num_boxes, **kwargs)

    # >>> Group Losses
    def get_group_loss(self, loss, outputs, targets, group_indices, **kwargs):
        loss_map = {
            'group_labels': self.loss_group_labels,
            'group_cardinality': self.loss_group_cardinality,
            'group_code': self.loss_group_code,
            'group_consistency': self.loss_group_consistency,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, group_indices, **kwargs)

    # *****************************************************************************

    def forward(self, outputs, targets, log=True):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        num_boxes = sum(len(u) for t in targets for u in t["actions"])
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        sim = outputs['membership']
        bs, num_queries, num_clip_boxes = sim.shape

        for tgt in targets:
            tgt["dummy_idx"] = torch.ones_like(tgt["actions"], dtype=int)
            for box_idx in range(num_clip_boxes):
                if bool(tgt["actions"][0, box_idx] == self.num_classes + 1):
                    tgt["dummy_idx"][0, box_idx] = 0

        input_targets = [copy.deepcopy(target) for target in targets]
        group_indices = self.group_matcher(outputs_without_aux, input_targets)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, num_boxes))

        # Group activity detection losses
        for loss in self.group_losses:
            losses.update(self.get_group_loss(loss, outputs, targets, group_indices))

        return losses
