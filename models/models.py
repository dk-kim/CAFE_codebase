import torch
import torch.nn as nn
import torch.nn.functional as F

from roi_align.roi_align import RoIAlign

from .backbone import build_backbone
from .group_transformer import build_group_transformer
from .feed_forward import MLP


class GADTR(nn.Module):
    def __init__(self, args):
        super(GADTR, self).__init__()

        self.dataset = args.dataset
        self.num_class = args.num_class
        self.num_frame = args.num_frame
        self.num_boxes = args.num_boxes

        self.hidden_dim = args.hidden_dim
        self.backbone = build_backbone(args)

        # RoI Align
        self.crop_size = args.crop_size
        self.roi_align = RoIAlign(crop_height=self.crop_size, crop_width=self.crop_size)
        self.fc_emb = nn.Linear(self.crop_size*self.crop_size*self.backbone.num_channels, self.hidden_dim)
        self.drop_emb = nn.Dropout(p=args.drop_rate)

        # Actor embedding
        self.input_proj = nn.Conv2d(self.backbone.num_channels, self.hidden_dim, kernel_size=1)
        self.box_pos_emb = MLP(4, self.hidden_dim, self.hidden_dim, 3)

        # Individual action classification head
        self.class_emb = nn.Linear(self.hidden_dim, self.num_class + 1)

        # Group Transformer
        self.group_transformer = build_group_transformer(args)
        self.num_group_tokens = args.num_group_tokens
        self.group_query_emb = nn.Embedding(self.num_group_tokens * self.num_frame, self.hidden_dim)
        
        # Group activity classfication head
        self.group_emb = nn.Linear(self.hidden_dim, self.num_class + 1)
        
        # Distance mask threshold
        self.distance_threshold = args.distance_threshold

        # Membership prediction heads
        self.actor_match_emb = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.group_match_emb = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.relu = F.relu

        for name, m in self.named_modules():
            if 'backbone' not in name and 'group_transformer' not in name:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def calculate_pairwise_distnace(self, boxes):
        bs = boxes.shape[0]

        rx = boxes.pow(2).sum(dim=2).reshape((bs, -1, 1))
        ry = boxes.pow(2).sum(dim=2).reshape((bs, -1, 1))

        dist = rx - 2.0 * boxes.matmul(boxes.transpose(1, 2)) + ry.transpose(1, 2)

        return torch.sqrt(dist)

    def forward(self, x, boxes, dummy_mask):
        """
        :param x: [B, T, 3, H, W]
        :param boxes: [B, T, N, 4]
        :param dummy_mask: [B, N]
        :return:
        """
        bs, t, _, h, w = x.shape
        n = boxes.shape[2]

        boxes = torch.reshape(boxes, (-1, 4))                                           # [b x t x n, 4]
        boxes_flat = boxes.clone().detach()
        boxes_idx = [i * torch.ones(n, dtype=torch.int) for i in range(bs * t)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes.device)
        boxes_idx_flat = torch.reshape(boxes_idx, (bs * t * n, ))                       # [b x t x n]

        features, pos = self.backbone(x)
        _, c, oh, ow = features.shape                                                   # [b x t, d, oh, ow]

        src = self.input_proj(features)
        src = torch.reshape(src, (bs, t, -1, oh, ow))                                   # [b, t, c, oh, ow]

        # calculate distance & distance mask
        boxes_center = boxes.clone().detach()
        boxes_center = torch.reshape(boxes_center[:, :2], (-1, n, 2))
        boxes_distance = self.calculate_pairwise_distnace(boxes_center)

        distance_mask = (boxes_distance > self.distance_threshold)

        # ignore dummy boxes (padded boxes to match the number of actors)
        dummy_mask = dummy_mask.unsqueeze(1).repeat(1, t, 1).reshape(-1, n)
        actor_dummy_mask = (~dummy_mask.unsqueeze(2)).float() @ (~dummy_mask.unsqueeze(1)).float()
        dummy_diag = (dummy_mask.unsqueeze(2).float() @ dummy_mask.unsqueeze(1).float()).nonzero(as_tuple=True)
        actor_mask = ~(actor_dummy_mask.bool())
        actor_mask[dummy_diag] = False
        actor_mask = distance_mask + actor_mask
        group_dummy_mask = dummy_mask

        boxes_flat[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * ow
        boxes_flat[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * oh
        boxes_flat[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * ow
        boxes_flat[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * oh

        boxes_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False

        # extract actor features
        actor_features = self.roi_align(features, boxes_flat, boxes_idx_flat)
        actor_features = torch.reshape(actor_features, (bs * t * n, -1))
        actor_features = self.fc_emb(actor_features)
        actor_features = F.relu(actor_features)
        actor_features = self.drop_emb(actor_features)
        actor_features = actor_features.reshape(bs, t, n, self.hidden_dim)

        # add positional information to box features
        box_pos_emb = self.box_pos_emb(boxes)
        box_pos_emb = torch.reshape(box_pos_emb, (bs, t, n, -1))                        # [b, t, n, c]
        actor_features = actor_features + box_pos_emb

        # group transformer
        hs, actor_att, feature_att = self.group_transformer(src, actor_mask, group_dummy_mask,
                                                            self.group_query_emb.weight, pos, actor_features)
        # [1, bs * t, n + k, f'], [1, bs * t, k, n], [1, bs * t, n + k, oh x ow]   M: # group tokens, K: # boxes

        actor_hs = hs[0, :, :n]
        group_hs = hs[0, :, n:]

        actor_hs = actor_hs.reshape(bs, t, n, -1)
        actor_hs = actor_features + actor_hs

        # normalize
        inst_repr = F.normalize(actor_hs.reshape(bs, t, n, -1).mean(dim=1), p=2, dim=2)
        group_repr = F.normalize(group_hs.reshape(bs, t, self.num_group_tokens, -1).mean(dim=1), p=2, dim=2)

        # prediction heads
        outputs_class = self.class_emb(actor_hs)

        outputs_group_class = self.group_emb(group_hs)

        outputs_actor_emb = self.actor_match_emb(inst_repr)
        outputs_group_emb = self.group_match_emb(group_repr)

        membership = torch.bmm(outputs_group_emb, outputs_actor_emb.transpose(1, 2))
        membership = F.softmax(membership, dim=1)

        out = {
            "pred_actions": outputs_class.reshape(bs, t, self.num_boxes, self.num_class + 1).mean(dim=1),
            "pred_activities": outputs_group_class.reshape(bs, t, self.num_group_tokens, self.num_class + 1).mean(dim=1),
            "membership": membership.reshape(bs, self.num_group_tokens, self.num_boxes),
            "actor_embeddings": F.normalize(actor_hs.reshape(bs, t, n, -1).mean(dim=1), p=2, dim=2),
        }

        return out
