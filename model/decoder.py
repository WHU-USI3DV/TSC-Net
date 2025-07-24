import math

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from typing import Any, Optional, Tuple, Type

class VisualEncoder(nn.Module):
    def __init__(self, num_layers, dim_feedforward=768, in_chans=1024, multi_scale=False, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.multi_scale = multi_scale

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(**kwargs) for _ in range(self.num_layers)]
        )
        self.upsample = nn.ConvTranspose2d(
                in_channels=in_chans,
                out_channels=in_chans,
                kernel_size=2,
                stride=2,
                padding=0)
        self.proj = nn.Conv2d(in_chans, dim_feedforward, 1, 1, 0)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.3)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def generate_position_encoding(self, seq_len, d_model, device):
        position = torch.arange(0, 1, 1.0 / seq_len, dtype=torch.float32, device=device).unsqueeze(1) * (1024 ** 2)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pos_encoding = torch.zeros((seq_len, d_model), dtype=torch.float32, device=device)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        return pos_encoding.unsqueeze(0)

    @staticmethod
    def get_reference_points(spatial_shapes, device):
        reference_points_list = []
        for _, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / H_
            ref_x = ref_x.reshape(-1)[None] / W_
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None]
        return reference_points


    def forward(self, features, pos=None):
        device = features[-1].device
        B = features[-1].shape[0]
        x = features[-1]
        x = self.proj(self.upsample(x))
        x = x.reshape(B, self.dim_feedforward, -1).permute(0, 2, 1)
        if pos is None:
            pos = self.generate_position_encoding(x.shape[1], self.dim_feedforward, x.device)
        for _, layer in enumerate(self.layers):
            x = layer(x, None, pos)
        return x


class DepthEncoder(nn.Module):

    def __init__(self, num_layers, dim_feedforward, in_chans=256, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(**kwargs) for _ in range(self.num_layers)]
        )
        self.project_0 =  nn.Conv2d(in_chans, self.dim_feedforward, 1, 1, 0)
        self.project_1 =  nn.Conv2d(in_chans, self.dim_feedforward, 1, 1, 0)
        self.project_2 =  nn.Conv2d(in_chans, self.dim_feedforward, 1, 1, 0)
        self.project_3 =  nn.Conv2d(in_chans, self.dim_feedforward, 1, 1, 0)

        self.div_term = torch.exp(torch.arange(0, 32, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / 32))

        self.depth_embed = nn.Sequential(
            nn.Conv2d(32, 32, 7, 7, 0),
            nn.Conv2d(32, self.dim_feedforward, 1, 1, 0)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.3)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def generate_position_encoding(self, seq_len, d_model, device):
        position = torch.arange(0, 1, 1.0 / seq_len, dtype=torch.float32, device=device).unsqueeze(1) * (1024 ** 2)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pos_encoding = torch.zeros((seq_len, d_model), dtype=torch.float32, device=device)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        return pos_encoding.unsqueeze(0)

    def get_3D_position(self, H, W, depth, device):
        if self.P is None:
            f = 1024 / (2 * math.tan(1024 * math.pi / 360))
            M = torch.tensor([
                [f, 0, 1024 / 2.0],
                [0, -f, 1024 / 2.0],
                [0, 0, 1,],
            ], device=device)
            W_old, H_old = 1024.0, 1024.0
            s_x = W / W_old
            s_y = H / H_old
            M[0, 0] *= s_x  # f_x
            M[1, 1] *= s_y  # f_y
            M[0, 2] *= s_x  # c_x
            M[1, 2] *= s_y
            y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
            y, x = y.unsqueeze(0).contiguous(), x.unsqueeze(0).contiguous()
            z = torch.ones_like(y, device=device)
            p = torch.cat((x, y, z), dim=0).float()
            intrinsics_inv = torch.inverse(M)
            p = p.reshape(3, -1)
            self.P = torch.matmul(intrinsics_inv, p) # 3 * N
            self.P = self.P.unsqueeze(0)

        B = depth.shape[0]
        pnts = self.P * depth.reshape([B, 1, -1]) # B，3, N
        embed = self.pos2posemb3d(pnts.permute(0, 2, 1))

        return embed

    def pos2posemb3d(self, pos, temperature=10000):
        num_pos_feats = self.embed_dims
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        pos_x = pos[..., 0, None] / dim_t
        pos_y = pos[..., 1, None] / dim_t
        pos_z = pos[..., 2, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
        posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
        return posemb

    def forward(self, depth_feature, depth, normal, conf, pos=None):
        B, _, H, W = depth.shape
        device = depth.device
        tar_sz = depth_feature[2].shape[-2:]

        src_0 = nn.functional.interpolate(self.project_0(depth_feature[0]), tar_sz, mode='bilinear', align_corners=True)
        src_1 = nn.functional.interpolate(self.project_1(depth_feature[1]), tar_sz, mode='bilinear', align_corners=True)
        src_2 = self.project_2(depth_feature[2])
        src_3 = nn.functional.interpolate(self.project_3(depth_feature[3]), tar_sz, mode='bilinear', align_corners=True)
        output = (src_0 + src_1 + src_2 + src_3) / 4.0

        div_term = self.div_term.clone().view(1, 16, 1, 1).to(device)
        depth_pos = torch.zeros((B, 32, H, W), dtype=torch.float32, device=device)
        depth_pos[:, 0::2] = torch.sin(depth * div_term)
        depth_pos[:, 1::2] = torch.cos(depth * div_term)
        depth_pos = self.depth_embed(depth_pos)

        output = output.reshape(B, self.dim_feedforward, -1).permute(0, 2, 1)
        depth_pos = depth_pos.reshape(B, self.dim_feedforward, -1).permute(0, 2, 1)

        if pos is None:
            pos = self.generate_position_encoding(depth_pos.shape[1], self.dim_feedforward, depth_pos.device)

        for layer in self.layers:
            output = layer(output, None, pos)

        return output + depth_pos

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", **kwargs):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.3)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, src_key_padding_mask, pos):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "leakyrelu":
        return F.leaky_relu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.3)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.leaky_relu(layer(x), 0.2, True) if i < self.num_layers - 1 else layer(x)
        return x
class ObjectDecoder(nn.Module):
    def __init__(self, num_layers=3, dim_feedforward=768, multi_scale=False, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [ObjectLayer(**kwargs) for _ in range(self.num_layers)]
        )

    def forward(self, v_f, d_f, q_pos, tgt, key_padding_mask, reference_points=None,spatial_shapes=None, level_start_index=None):
        # reference_points: B, N, 2

        for layer in self.layers:
            tgt = layer(v_f, d_f, q_pos, tgt, key_padding_mask, reference_points, spatial_shapes, level_start_index)

        return tgt

class ObjectLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=256,
                 dropout=0.1, activation="relu",
                 n_heads=8, n_levels=4, n_points=8, multi_scale=False, **kwargs):
        super().__init__()
        self.multi_scale = multi_scale

        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # depth cross attention
        self.cross_attn_depth = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout_depth = nn.Dropout(dropout)
        self.norm_depth = nn.LayerNorm(d_model)
        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        # Decoder Self-Attention
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.nhead = n_heads
        self.apply(self._init_weights)

    def _init_weights(self, m):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.3)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def generate_position_encoding(self, seq_len, d_model, device):
        position = torch.arange(0, 1, 1.0 / seq_len, dtype=torch.float32, device=device).unsqueeze(1) * (1024 ** 2)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pos_encoding = torch.zeros((seq_len, d_model), dtype=torch.float32, device=device)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        return pos_encoding.unsqueeze(0)


    def forward(self, visual_features, depth_features, pos, query, key_padding_mask, reference_points=None, spatial_shapes=None, level_start_index=None):
        # B, N, C
        # B, N, C
        # B, num_query, C
        bs = depth_features.shape[0]
        tgt = query

        # depth cross
        tgt2 = self.cross_attn_depth(self.with_pos_embed(tgt, pos).transpose(0, 1), depth_features.transpose(0, 1), depth_features.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout_depth(tgt2)
        tgt = self.norm_depth(tgt)

        # B, N, C
        # self attention
        q = k = self.with_pos_embed(tgt, pos)
        q_content = self.sa_qcontent_proj(q)
        q_pos = self.sa_qpos_proj(q)
        k_content = self.sa_kcontent_proj(k)
        k_pos = self.sa_kpos_proj(k)
        v = self.sa_v_proj(tgt)
        q = q_content + q_pos
        k = k_content + k_pos
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = tgt.transpose(0, 1)
        tgt2 = self.self_attn(q, k, v, key_padding_mask=key_padding_mask)[0].transpose(0, 1)
        # 做self attention的时候不使用没有值的key
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.cross_attn(self.with_pos_embed(tgt, pos).transpose(0, 1), visual_features.transpose(0, 1), visual_features.transpose(0, 1))[0].transpose(0, 1)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.div_term = torch.exp(torch.arange(0, self.embed_dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / 32))

        self.project = nn.Linear(self.embed_dim * 3, self.embed_dim)

        point_embeddings = [nn.Embedding(1, self.embed_dim) for _ in range(3)]
        # 来标注锚点，w, h
        self.point_embeddings = nn.ModuleList(point_embeddings)

    def _embed_boxes(self, boxes: torch.Tensor, corner_embedding ) -> torch.Tensor:
        """Embeds box prompts."""
        B, N, _ = boxes.shape
        device = boxes.device
        div_term = self.div_term.clone().view(1, 1, self.embed_dim // 2).to(device)
        width_embedding = torch.zeros((B, N, self.embed_dim), dtype=torch.float32, device=device)
        height_embedding = torch.zeros((B, N, self.embed_dim), dtype=torch.float32, device=device)
        width_embedding[..., 0::2] = torch.sin(boxes[..., 2:3] * div_term)
        width_embedding[..., 1::2] = torch.cos(boxes[..., 2:3] * div_term)
        height_embedding[..., 0::2] = torch.sin(boxes[..., 3:4] * div_term)
        height_embedding[..., 1::2] = torch.cos(boxes[..., 3:4] * div_term)
        corner_embedding += self.point_embeddings[0].weight
        width_embedding += self.point_embeddings[1].weight
        height_embedding += self.point_embeddings[2].weight
        return self.project(torch.cat((corner_embedding, width_embedding, height_embedding), dim=-1))

    def _get_batch_size(
        self,
        boxes: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if boxes is not None:
            return boxes.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        boxes: Optional[torch.Tensor],
        pos,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(boxes)
        box_embeddings = self._embed_boxes(boxes, pos)

        return box_embeddings


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x