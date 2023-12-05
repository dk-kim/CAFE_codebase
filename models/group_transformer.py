# ------------------------------------------------------------------------
# Modified from HOTR (https://github.com/kakaobrain/HOTR)
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_actor=14,
                 dim_feedforward=2048, dropout=0.1, activation="relu", return_intermediate_dec=False):
        super().__init__()

        decoder_layer = TransformerDecoderLayer(d_model, nhead, num_actor, dim_feedforward, dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_encoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, actor_mask, group_dummy_mask, group_embed, pos_embed, actor_embed):
        bs, t, c, h, w = src.shape
        _, _, n, _ = actor_embed.shape
        src = src.reshape(bs * t, c, h, w).flatten(2).permute(2, 0, 1)              # [h x w, bs x t, c]
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        group_embed = group_embed.reshape(-1, t, c)
        group_embed = group_embed.unsqueeze(1).repeat(1, bs, 1, 1).reshape(-1, bs * t, c)
        actor_embed = actor_embed.reshape(bs * t, -1, c).permute(1, 0, 2)           # [n, bs x t, c]
        query_embed = torch.cat([actor_embed, group_embed], dim=0)                  # [n + k, bs x t, c]
        tgt = torch.zeros_like(query_embed)                                         # [n + k, bs x t, c]
        if actor_mask is not None:
            actor_mask = actor_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1).reshape(-1, n, n)
        hs, actor_att, feature_att = self.decoder(tgt, src, attn_mask=actor_mask,
                                                  tgt_key_padding_mask=group_dummy_mask,
                                                  pos=pos_embed, query_pos=query_embed)

        return hs.transpose(1, 2), actor_att, feature_att


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        actor_att = None
        feature_att = None

        intermediate = []
        intermediate_actor_att = []
        intermediate_feature_att = []

        for layer in self.layers:
            output, actor_att, feature_att = layer(output, memory, tgt_mask=tgt_mask,
                                                   memory_mask=memory_mask,
                                                   tgt_key_padding_mask=tgt_key_padding_mask,
                                                   memory_key_padding_mask=memory_key_padding_mask,
                                                   attn_mask=attn_mask,
                                                   pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                intermediate_actor_att.append(actor_att)
                intermediate_feature_att.append(feature_att)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_actor_att), torch.stack(intermediate_feature_att)

        if actor_att is not None:
            actor_att = actor_att.unsqueeze(0)
        if feature_att is not None:
            feature_att = feature_att.unsqueeze(0)

        return output.unsqueeze(0), actor_att, feature_att


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, num_actor, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        self.multihead_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.self_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.num_actor = num_actor

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     attn_mask=None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        q = k = self.with_pos_embed(tgt, query_pos)

        actor_q = q[:self.num_actor, :, :]
        group_q = q[self.num_actor:, :, :]
        actor_k = k[:self.num_actor, :, :]
        group_k = k[self.num_actor:, :, :]

        tgt_actor = tgt[:self.num_actor, :, :]
        tgt_group = tgt[self.num_actor:, :, :]

        # actor-actor, group-group self-attention
        tgt2_actor, actor_att = self.self_attn1(actor_q, actor_k, value=tgt_actor, attn_mask=attn_mask)
        tgt2_group, _ = self.self_attn2(group_q, group_k, value=tgt_group)
        tgt2 = torch.cat([tgt2_actor, tgt2_group], dim=0)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # actor-group attention
        tgt_actor = tgt[:self.num_actor, :, :]
        tgt_group = tgt[self.num_actor:, :, :]

        tgt2_group, group_att = self.multihead_attn1(query=tgt_group, key=tgt_actor, value=tgt_actor,
                                                     key_padding_mask=tgt_key_padding_mask)

        tgt2 = torch.cat([tgt_actor, tgt2_group], dim=0)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # actor-feature, group-feature cross-attention
        tgt2, feature_att = self.multihead_attn2(query=self.with_pos_embed(tgt, query_pos),
                                                 key=self.with_pos_embed(memory, pos), value=memory)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm4(tgt)

        return tgt, group_att, feature_att

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        return self.forward_post(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask,
                                 memory_key_padding_mask, attn_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_group_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.drop_rate,
        nhead=args.gar_nheads,
        dim_feedforward=args.gar_ffn_dim,
        num_encoder_layers=args.gar_enc_layers,
        return_intermediate_dec=False,
        num_actor=args.num_boxes,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
