import torch
from torch import nn
from torch.nn.init import xavier_uniform_, constant_, normal_

from adet.modeling.ops.modules.ms_deform_attn import MSDeformAttn
from .trans_utils import _get_clones, get_reference_points, with_pos_embed
from .feed_forward import get_ffn


class ClarityFormerTransformerEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 ffn_type="default", num_feature_levels=4, enc_n_points=4):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = TransformerEncoderLayer(d_model, dim_feedforward,
                                                dropout, ffn_type,
                                                num_feature_levels, nhead, enc_n_points)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    # def forward(self, srcs, pos_embeds):
    #     # prepare input for encoder
    #     src_flatten = []
    #     lvl_pos_embed_flatten = []
    #     spatial_shapes = []
    #     for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
    #         bs, c, h, w = src.shape
    #         spatial_shape = (h, w)
    #         spatial_shapes.append(spatial_shape)
    #         src = src.flatten(2).transpose(1, 2)
    #         pos_embed = pos_embed.flatten(2).transpose(1, 2)
    #         lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
    #         lvl_pos_embed_flatten.append(lvl_pos_embed)
    #         src_flatten.append(src)
    #     src_flatten = torch.cat(src_flatten, 1)
    #     lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
    #     spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
    #     level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))

    #     # encoder
    #     memory = self.encoder(src_flatten, spatial_shapes, level_start_index, lvl_pos_embed_flatten)

    #     return memory, level_start_index
    def forward(self, srcs, pos_embeds):
        # prepare input for encoder
        src_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
        src_flatten = torch.cat(src_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # encoder
        # *** 淇敼杩欓噷锛氭帴鏀?reference_points 鍜?all_layer_offsets ***
        memory, reference_points, all_layer_offsets, all_layer_weights = self.encoder(src_flatten, spatial_shapes, level_start_index, lvl_pos_embed_flatten)

        # *** 淇敼杩欓噷锛氳繑鍥炴墍鏈夐渶瑕佺敤浜庡彲瑙嗗寲鐨勪俊鎭?***
        return memory, level_start_index, spatial_shapes, reference_points, all_layer_offsets, all_layer_weights


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, ffn_type="default",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.ffn = get_ffn(d_model, ffn_type)

    # def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
    #     # self attention
    #     src2 = self.self_attn(with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
    #     src = src + self.dropout1(src2)
    #     src = self.norm1(src)   # (bs, w*h, dim)

    #     # ffn
    #     src = self.ffn(src, spatial_shapes, level_start_index)

    #     return src
    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        # *** 淇敼杩欓噷锛氭帴鏀?offsets ***
        src2, offsets, weights = self.self_attn(with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.ffn(src, spatial_shapes, level_start_index)

        # *** 淇敼杩欓噷锛氳繑鍥?offsets ***
        return src, offsets, weights

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    # def forward(self, src, spatial_shapes, level_start_index, pos=None):
    #     output = src
    #     batch_size = src.shape[0]
    #     reference_points = get_reference_points(spatial_shapes, batch_size, device=src.device)
    #     for _, layer in enumerate(self.layers):
    #         output = layer(output, pos, reference_points, spatial_shapes, level_start_index)

    #     return output
    def forward(self, src, spatial_shapes, level_start_index, pos=None):
        output = src
        batch_size = src.shape[0]
        reference_points = get_reference_points(spatial_shapes, batch_size, device=src.device)
        
        # *** 鏂板锛氱敤浜庡瓨鍌ㄦ瘡涓€灞傜殑 offsets ***
        all_layer_offsets = []
        all_layer_weights = []

        for _, layer in enumerate(self.layers):
            # *** 淇敼杩欓噷锛氭帴鏀?offsets ***
            output, offsets, weights = layer(output, pos, reference_points, spatial_shapes, level_start_index)
            all_layer_offsets.append(offsets)
            all_layer_weights.append(weights)

        # *** 淇敼杩欓噷锛氳繑鍥?output, reference_points, 鍜屾墍鏈夊眰鐨?offsets ***
        return output, reference_points, all_layer_offsets, all_layer_weights

def build_transformer_encoder(cfg):
    return ClarityFormerTransformerEncoder(
        d_model=cfg.MODEL.CLARITY_FORMER.HIDDEN_DIM,
        nhead=cfg.MODEL.CLARITY_FORMER.NHEAD,
        num_encoder_layers=cfg.MODEL.CLARITY_FORMER.ENC_LAYERS,
        dim_feedforward=cfg.MODEL.CLARITY_FORMER.DIM_FEEDFORWARD,
        dropout=0.1,
        ffn_type=cfg.MODEL.CLARITY_FORMER.FFN,
        num_feature_levels=len(cfg.MODEL.CLARITY_FORMER.FEAT_INSTANCE_STRIDES),
        enc_n_points=cfg.MODEL.CLARITY_FORMER.ENC_POINTS)



