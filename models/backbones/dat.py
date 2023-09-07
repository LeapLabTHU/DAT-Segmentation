# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .loading import load_checkpoint
from mmseg.utils import get_root_logger
from mmcv.runner import auto_fp16
from timm.models.layers import DropPath, to_2tuple
from mmseg.models.builder import BACKBONES
from ..utils.dat_blocks import *
from ..utils.nat import NeighborhoodAttention2D
from ..utils.slide import SlideAttention

import torch.utils.checkpoint as cp

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
        self.fp16_enabled = False
    
    @auto_fp16(apply_to=('x', ))
    def forward(self, x):
        B, C, H, W = x.size()
        gamma = self.gamma[None, :, None, None].expand(B, C, H, W)
        return x * gamma


class TransformerStage(nn.Module):

    def __init__(self, fmap_size, window_size, ns_per_pt,
                 dim_in, dim_embed, depths, stage_spec, n_groups, 
                 use_pe, sr_ratio,
                 heads, heads_q, stride,
                 offset_range_factor,
                 local_orf, local_kv_size,
                 dwc_pe, no_off, fixed_pe,
                 attn_drop, proj_drop, expansion, drop, drop_path_rate, 
                 use_dwc_mlp, ksize, nat_ksize,
                 k_qna, nq_qna, qna_activation, deform_groups, 
                 layer_scale_value, 
                 use_lpu, use_cmt_mlp, log_cpb, 
                 stage_i, 
                 use_checkpoint):

        super().__init__()
        self.fp16_enabled = False
        self.use_checkpoint = use_checkpoint
        fmap_size = to_2tuple(fmap_size)
        local_kv_size = to_2tuple(local_kv_size)
        self.depths = depths
        hc = dim_embed // heads
        assert dim_embed == heads * hc
        self.proj = nn.Conv2d(dim_in, dim_embed, 1, 1, 0) if dim_in != dim_embed else nn.Identity()
        self.use_lpu = use_lpu
        self.stage_spec = stage_spec

        self.ln_cnvnxt = nn.ModuleDict(
            {str(d): LayerNormProxy(dim_embed) for d in range(depths) if stage_spec[d] == 'X'}
        )
        self.layer_norms = nn.ModuleList(
            [LayerNormProxy(dim_embed) if stage_spec[d // 2] != 'X' else nn.Identity() for d in range(2 * depths)]
        )

        mlp_fn = TransformerMLP
        if use_dwc_mlp:
            if use_cmt_mlp:
                mlp_fn = TransformerMLPWithConv_CMT
            else:
                mlp_fn = TransformerMLPWithConv

        self.mlps = nn.ModuleList(
            [ 
                mlp_fn(dim_embed, expansion, drop) for _ in range(depths)
            ]
        )
        self.attns = nn.ModuleList()
        self.drop_path = nn.ModuleList()
        self.layer_scales = nn.ModuleList(
            [
                LayerScale(dim_embed, init_values=layer_scale_value) if layer_scale_value > 0.0 else nn.Identity() 
                for _ in range(2 * depths)
            ]
        )
        self.local_perception_units = nn.ModuleList(
            [
                nn.Conv2d(dim_embed, dim_embed, kernel_size=3, stride=1, padding=1, groups=dim_embed) if use_lpu else nn.Identity()
                for _ in range(depths)
            ]
        )
        for i in range(depths):
            if stage_spec[i] == 'L':
                self.attns.append(
                    LocalAttention(dim_embed, heads, window_size, attn_drop, proj_drop)
                )
            elif stage_spec[i] == 'D':
                self.attns.append(
                    DAttentionBaseline(
                        fmap_size, 
                        fmap_size, 
                        heads, 
                        hc, 
                        n_groups, 
                        attn_drop, 
                        proj_drop, 
                        stride,
                        offset_range_factor,
                        use_pe,
                        dwc_pe, 
                        no_off,
                        fixed_pe,
                        ksize,
                        log_cpb,
                        stage_i
                    )
                )
            elif stage_spec[i] == 'S':
                shift_size = math.ceil(window_size / 2)
                self.attns.append(
                    ShiftWindowAttention(dim_embed, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size)
                )
            elif stage_spec[i] == 'N':
                self.attns.append(
                    NeighborhoodAttention2D(dim_embed, nat_ksize, heads, attn_drop, proj_drop)
                )
            elif stage_spec[i] == 'A':
                self.attns.append(
                    LDABaseline(fmap_size, local_kv_size, heads, hc, n_groups, use_pe, no_off, local_orf)
                )
            elif stage_spec[i] == 'P':
                self.attns.append(
                    PyramidAttention(dim_embed, heads, attn_drop, proj_drop, sr_ratio)
                )
            elif self.stage_spec[i] == 'X':
                self.attns.append(
                    nn.Conv2d(dim_embed, dim_embed, kernel_size=window_size, padding=window_size // 2, groups=dim_embed)
                )
            elif self.stage_spec[i] == 'E':
                self.attns.append(
                    SlideAttention(dim_embed, heads, 3)
                )
            else:
                raise NotImplementedError(f'Spec: {stage_spec[i]} is not supported.')
            
            self.drop_path.append(DropPath(drop_path_rate[i]) if drop_path_rate[i] > 0.0 else nn.Identity())
    
    @auto_fp16(apply_to=('x', ))
    def _inner_forward(self, x):

        # assert x.dtype == torch.float16, f"AMP failed!, dtype={x.dtype}"
        x = self.proj(x)

        # positions = []
        # references = []
        for d in range(self.depths):

            if self.use_lpu:
                x0 = x
                x = self.local_perception_units[d](x.contiguous())
                x = x + x0
            
            if self.stage_spec[d] == 'X':
                x0 = x
                x = self.attns[d](x.contiguous())
                x = self.mlps[d](self.ln_cnvnxt[str(d)](x))
                x = self.drop_path[d](x) + x0
            else:
                x0 = x
                x, pos, ref = self.attns[d](self.layer_norms[2 * d](x))
                x = self.layer_scales[2 * d](x)
                x = self.drop_path[d](x) + x0
                x0 = x
                x = self.mlps[d](self.layer_norms[2 * d + 1](x))
                x = self.layer_scales[2 * d + 1](x)
                x = self.drop_path[d](x) + x0
            # positions.append(pos)
            # references.append(ref)

        # return x, positions, references
        # assert x.dtype == torch.float16, f"AMP failed!, dtype={x.dtype}"
        return x

    @auto_fp16(apply_to=('x', ))
    def forward(self, x):
        if self.training and x.requires_grad and self.use_checkpoint:
            return cp.checkpoint(self._inner_forward, x)
        else:
            return self._inner_forward(x)


@BACKBONES.register_module()
class DAT(nn.Module):

    def __init__(self, img_size=224, patch_size=4, num_classes=1000, expansion=4,
                 dim_stem=96, dims=[96, 192, 384, 768], depths=[2, 2, 6, 2], 
                 heads=[3, 6, 12, 24], heads_q=[6, 12, 24, 48],
                 window_sizes=[7, 7, 7, 7],
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, 
                 strides=[-1,-1,-1,-1],
                 offset_range_factor=[1, 2, 3, 4],
                 local_orf=[-1, -1, -1, -1],
                 local_kv_sizes=[-1, -1, -1, -1],
                 offset_pes=[False, False, False, False],
                 stage_spec=[['L', 'D'], ['L', 'D'], ['L', 'D', 'L', 'D', 'L', 'D'], ['L', 'D']], 
                 groups=[-1, -1, 3, 6],
                 use_pes=[False, False, False, False], 
                 dwc_pes=[False, False, False, False],
                 sr_ratios=[8, 4, 2, 1], 
                 lower_lr_kvs={},
                 fixed_pes=[False, False, False, False],
                 no_offs=[False, False, False, False],
                 ns_per_pts=[4, 4, 4, 4],
                 use_dwc_mlps=[False, False, False, False],
                 use_conv_patches=False,
                 ksizes=[9, 7, 5, 3],
                 ksize_qnas=[3, 3, 3, 3],
                 nqs=[2, 2, 2, 2],
                 qna_activation='exp',
                 deform_groups=[0, 0, 0, 0],
                 nat_ksizes=[3,3,3,3],
                 layer_scale_values=[-1,-1,-1,-1],
                 use_lpus=[False, False, False, False],
                 use_cmt_mlps=[False, False, False, False],
                 log_cpb=[False, False, False, False],
                 out_indices=(0, 1, 2, 3),
                 use_checkpoint=True,
                 init_cfg=dict(type='Pretrained', checkpoint=None),
                 **kwargs):
        super().__init__()
        
        self.fp16_enabled = False
        self.out_indices = out_indices
        
        self.log_cpb = log_cpb[0]
        self.dwc_pe = dwc_pes[0]
        self.slide = stage_spec[0][0] == "E"

        self.patch_proj = nn.Sequential(
            nn.Conv2d(3, dim_stem // 2, 3, patch_size // 2, 1),
            LayerNormProxy(dim_stem // 2),
            nn.GELU(),
            nn.Conv2d(dim_stem // 2, dim_stem, 3, patch_size // 2, 1),
            LayerNormProxy(dim_stem)
        ) if use_conv_patches else nn.Sequential(
            nn.Conv2d(3, dim_stem, patch_size, patch_size, 0),
            LayerNormProxy(dim_stem)
        )

        img_size = img_size // patch_size
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stages = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(4):
            dim1 = dim_stem if i == 0 else dims[i - 1] * 2
            dim2 = dims[i]
            self.stages.append(
                TransformerStage(
                    img_size, window_sizes[i], ns_per_pts[i],
                    dim1, dim2, depths[i],
                    stage_spec[i], groups[i], use_pes[i],
                    sr_ratios[i], heads[i], heads_q[i], strides[i],
                    offset_range_factor[i],
                    local_orf[i], local_kv_sizes[i],
                    dwc_pes[i], no_offs[i], fixed_pes[i],
                    attn_drop_rate, drop_rate, expansion, drop_rate,
                    dpr[sum(depths[:i]):sum(depths[:i + 1])],
                    use_dwc_mlps[i],
                    ksizes[i], nat_ksizes[i],
                    ksize_qnas[i],
                    nqs[i],
                    qna_activation,
                    deform_groups[i],
                    layer_scale_values[i],
                    use_lpus[i],
                    use_cmt_mlps[i],
                    log_cpb[i],
                    i, use_checkpoint
                )
            )
            if i in self.out_indices:
                self.norms.append(
                    LayerNormProxy(dim2)
                )
            else:
                self.norms.append(nn.Identity())
            img_size = img_size // 2

        self.down_projs = nn.ModuleList()
        for i in range(3):
            self.down_projs.append(
                nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1, bias=False),
                    LayerNormProxy(dims[i + 1])
                ) if use_conv_patches else nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 2, 2, 0, bias=False),
                    LayerNormProxy(dims[i + 1])
                )
            )

        self.lower_lr_kvs = lower_lr_kvs
        self.init_cfg = init_cfg
        self.reset_parameters()

    def reset_parameters(self):

        for m in self.parameters():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def init_weights(self):
        
        pretrained = self.init_cfg.get('checkpoint', None)
        
        if pretrained is None:
            return
        
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
            logger.info("DAT bakcbone has been loaded successfully!")

    @auto_fp16(apply_to=('x', ))
    def forward(self, x):
        
        x = self.patch_proj(x)
        
        outs = []
        for i in range(4):
            
            x = self.stages[i](x)
            
            y = self.norms[i](x)
            outs.append(y.contiguous())
            if i < 3:
                x = self.down_projs[i](x)

        return outs
