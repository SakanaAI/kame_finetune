from copy import deepcopy

import torch
import torch.nn as nn
from kame.models import LMModel


def extend_moshi_modules_for_user_stream(lm: LMModel) -> LMModel:
    """
    Extend the depth transformer's modules to model user stream.
    1. depformer_in * 2
    2. depformer_emb * 2 + 1
    3. depformer (layers)
        3.1 self_attn.in_proj_weight * 2
        3.2 self_attn.out_proj * 2
        3.3 gating * 2
    4. linears * 2
    """
    lm_us = deepcopy(lm)

    # 1. depformer_in
    lm_us.depformer_in.extend(deepcopy(lm.depformer_in))

    # 2. depformer_emb
    # depformer_emb doesn't have any embedding to encode the moshi's last acoustic token
    # (i.e., embedding for predicting user's semantic token) because moshi's semantic
    # token is predicted from text token, so we just use the first embedding in depformer_emb,
    # which is for predicting first acoustic token
    lm_us.depformer_emb.append(deepcopy(lm.depformer_emb[0]))
    lm_us.depformer_emb.extend(deepcopy(lm.depformer_emb))

    # 3. depformer (layers)
    for layer in lm_us.depformer.layers:
        # 3.1 self_attn.in_proj
        layer.self_attn.in_proj_weight.data = layer.self_attn.in_proj_weight.data.repeat(2, 1)

        sa = layer.self_attn
        sa.in_projs.extend(deepcopy(sa.in_projs))

        # 3.2 self_attn.out_proj
        new_linear = torch.nn.Linear(
            in_features=layer.self_attn.out_proj.in_features,
            out_features=layer.self_attn.out_proj.out_features * 2,
            bias=False if layer.self_attn.out_proj.bias is None else True,
        )
        new_linear.load_state_dict(
            {
                "weight": layer.self_attn.out_proj.weight.repeat(2, 1),
            }
        )
        layer.self_attn.out_proj = new_linear
        sa.out_projs.extend(deepcopy(sa.out_projs))

        # 3.3 gating
        layer.gating.extend(deepcopy(layer.gating))

    # 4. linears
    lm_us.linears.extend(deepcopy(lm_us.linears))

    return lm_us


def remove_moshi_modules_for_user_stream(lm_us: LMModel) -> LMModel:
    """
    Remove the depth transformer's modules to model user stream.
    Reverse of `extend_moshi_modules_for_user_stream()`.
    """
    lm = deepcopy(lm_us)
    device = next(lm.parameters()).device
    dtype = next(lm.parameters()).dtype

    # depformer_in
    lm.depformer_in = lm.depformer_in[:8]
    # depformer_emb
    lm.depformer_emb = lm.depformer_emb[:7]
    # depformer.layers
    for layer in lm.depformer.layers:
        # self_attn.in_proj_weight
        layer.self_attn.in_proj_weight.data = layer.self_attn.in_proj_weight.data[
            : layer.self_attn.in_proj_weight.shape[0] // 2
        ]
        # self_attn.out_proj
        out_proj = nn.Linear(
            in_features=layer.self_attn.out_proj.in_features,
            out_features=layer.self_attn.out_proj.out_features // 2,
            bias=False,
            device=device,
            dtype=dtype,
        )
        out_proj.load_state_dict(
            {
                "weight": layer.self_attn.out_proj.weight[
                    : layer.self_attn.out_proj.out_features // 2
                ]
            }
        )
        layer.self_attn.out_proj = out_proj
        # gating
        layer.gating = layer.gating[:8]
    # linears
    lm.linears = lm.linears[:8]

    return lm
