import json
import os
import re
from collections import OrderedDict

import torch
from kame.models import LMModel
from kame.modules.gating import (
    gating_forward_kernel,
)
from kame.modules.transformer import (
    StreamingTransformer,
    create_sin_embedding,
)
from safetensors.torch import load_file, save_file

from models.oracle_embedding_utils import (
    backfill_oracle_embedding_from_text,
    validate_oracle_embedding_checkpoint_load,
)


def expose_linear_weights_for_zero3(
    moshi_lm: LMModel,
) -> None:
    """Exposes linear layer weights at their parent modules for DeepSpeed Zero-3 compatibility.

    In DeepSpeed Zero-3, weights of child modules cannot be accessed directly, so we need to
    restructure the model by exposing weights from linear layers at their
    parent modules while removing the original linear modules to maintain compatibility.

    Target modules:
    - `transformer.layers[*].gating.linear_in`
    - `transformer.layers[*].gating.linear_out`
    - `depformer.layers[*].gating[*].linear_in`
    - `depformer.layers[*].gating[*].linear_out`
    - `depformer.layers[*].self_attn.out_proj`
    """
    if isinstance(moshi_lm.transformer, StreamingTransformer):
        for layer in moshi_lm.transformer.layers:
            layer.gating.linear_in_weight = layer.gating.linear_in.weight
            layer.gating.linear_out_weight = layer.gating.linear_out.weight
            del layer.gating.linear_in, layer.gating.linear_out
    if isinstance(moshi_lm.depformer, StreamingTransformer):
        for layer in moshi_lm.depformer.layers:
            for gating in layer.gating:
                gating.linear_in_weight = gating.linear_in.weight
                gating.linear_out_weight = gating.linear_out.weight
                del gating.linear_in, gating.linear_out


def activation_gating_forward(self, x: torch.Tensor):
    """
    Forward patch used after exposing gating linear weights for DeepSpeed ZeRO-3 compatibility.
    """
    return gating_forward_kernel(
        # use the exposed linear weights
        self.linear_in_weight,
        self.linear_out_weight,
        self.activation,
        x,
    )


def transformer_forward(self, x: torch.Tensor, *args, **kwargs):
    """
    Forward patch used by the finetuning wrapper to support activation checkpointing.
    """
    _, T, C = x.shape

    dtype_input = x.dtype
    state = self._streaming_state
    if state is None:
        offsets = torch.zeros(1, dtype=torch.long, device=x.device)
    else:
        offsets = state.offsets

    if self.positional_embedding in {"sin", "sin_rope"}:
        positions = torch.arange(T, device=x.device).view(1, -1, 1)
        positions = positions + offsets.view(-1, 1, 1)
        pos_emb = create_sin_embedding(positions, C, max_period=self.max_period, dtype=x.dtype)
        x = x + self.positional_scale * pos_emb

    for layer in self.layers:
        if self.activation_checkpointing:
            x = self.checkpointing_func(layer, x)
        else:
            x = layer(x)  # , *args, **kwargs)

    if state is not None:
        state.offsets[:] = torch.where(state.exec_mask, state.offsets + T, state.offsets)
    return x.to(dtype_input)


def restore_linear_weights_from_exposed_state_dict(
    moshi_lm_for_ft_state_dict: OrderedDict,
) -> OrderedDict:
    """
    Restore linear layer weights from the exposed state dict for DeepSpeed Zero-3 compatibility.

    Target parameters:
    - `transformer.layers[*].gating.linear_in_weight`
    - `transformer.layers[*].gating.linear_out_weight`
    - `depformer.layers[*].gating[*].linear_in_weight`
    - `depformer.layers[*].gating[*].linear_out_weight`
    - `depformer.layers[*].self_attn.out_proj_weight`
    """

    gating_linear_in_pattern = re.compile(r"transformer\.layers\.\d+\.gating\.linear_in_weight")
    gating_linear_out_pattern = re.compile(r"transformer\.layers\.\d+\.gating\.linear_out_weight")
    depformer_gating_linear_in_pattern = re.compile(
        r"depformer\.layers\.\d+\.gating\.\d+\.linear_in_weight"
    )
    depformer_gating_linear_out_pattern = re.compile(
        r"depformer\.layers\.\d+\.gating\.\d+\.linear_out_weight"
    )

    new_state_dict = OrderedDict()
    for key in moshi_lm_for_ft_state_dict.keys():
        if gating_linear_in_pattern.match(key):
            new_key = key.replace("linear_in_weight", "linear_in.weight")
        elif gating_linear_out_pattern.match(key):
            new_key = key.replace("linear_out_weight", "linear_out.weight")
        elif depformer_gating_linear_in_pattern.match(key):
            new_key = key.replace("linear_in_weight", "linear_in.weight")
        elif depformer_gating_linear_out_pattern.match(key):
            new_key = key.replace("linear_out_weight", "linear_out.weight")
        else:
            new_key = key
        if new_key != key:
            print(f"{key} -> {new_key}")
        new_state_dict[new_key] = moshi_lm_for_ft_state_dict[key]

    return new_state_dict


def _copy_module_state_(source_module: torch.nn.Module, target_module: torch.nn.Module) -> None:
    source_params = dict(source_module.named_parameters())
    target_params = dict(target_module.named_parameters())
    if source_params.keys() != target_params.keys():
        raise ValueError(
            "Source and target modules do not share the same parameter structure: "
            f"{sorted(source_params.keys())} != {sorted(target_params.keys())}"
        )

    source_buffers = dict(source_module.named_buffers())
    target_buffers = dict(target_module.named_buffers())
    if source_buffers.keys() != target_buffers.keys():
        raise ValueError(
            "Source and target modules do not share the same buffer structure: "
            f"{sorted(source_buffers.keys())} != {sorted(target_buffers.keys())}"
        )

    with torch.no_grad():
        for name, target_param in target_params.items():
            target_param.copy_(source_params[name])
        for name, target_buffer in target_buffers.items():
            target_buffer.copy_(source_buffers[name])


class MoshiForFinetuning(LMModel):
    """
    Moshi language model for finetuning.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # DeepSpeed Zero-3 compatibility
        ## 1. Expose linear layer weights
        expose_linear_weights_for_zero3(self)
        ## 2. Apply patches for forward functions
        for layer in self.transformer.layers:
            layer.gating.forward = activation_gating_forward.__get__(layer.gating)
        for layer in self.depformer.layers:
            for gating in layer.gating:
                gating.forward = activation_gating_forward.__get__(gating)

        # Implement activation checkpointing
        self.transformer.activation_checkpointing = False
        self.transformer.forward = transformer_forward.__get__(self.transformer)
        self.depformer.activation_checkpointing = False
        self.depformer.forward = transformer_forward.__get__(self.depformer)

    def enable_activation_checkpointing(self, checkpointing_func):
        self.transformer.activation_checkpointing = True
        self.transformer.checkpointing_func = checkpointing_func
        self.depformer.activation_checkpointing = True
        self.depformer.checkpointing_func = checkpointing_func

    def disable_activation_checkpointing(self):
        self.transformer.activation_checkpointing = False
        self.depformer.activation_checkpointing = False

    def materialize_oracle_embedding_for_checkpoint_(
        self,
        oracle_embedding_mode: str,
    ) -> None:
        # Existing inference code always reads oracle_emb. In tie mode we keep
        # training semantics tied to text_emb, then mirror that state into oracle_emb
        # before checkpoint serialization.
        if oracle_embedding_mode == "tie":
            _copy_module_state_(self.text_emb, self.oracle_emb)
        elif oracle_embedding_mode != "separate":
            raise ValueError(f"Unknown oracle embedding mode: {oracle_embedding_mode}")

    @classmethod
    def from_original_moshi_lm(
        cls,
        moshi_lm: LMModel,
        moshi_lm_kwargs: dict,
    ) -> "MoshiForFinetuning":
        """
        Initialize `MoshiForFinetuning` from the original `LMModel`.
        """
        # Expose linear layer weights for DeepSpeed Zero-3 compatibility
        expose_linear_weights_for_zero3(moshi_lm)
        state_dict = moshi_lm.state_dict()
        device = next(moshi_lm.parameters()).device
        dtype = next(moshi_lm.parameters()).dtype

        # Clear the original model to save memory
        del moshi_lm

        # Initialize the new model
        moshi_lm = cls(device=device, dtype=dtype, **moshi_lm_kwargs).to(device=device, dtype=dtype)
        incompatible = moshi_lm.load_state_dict(state_dict, strict=False)
        oracle_emb_missing = validate_oracle_embedding_checkpoint_load(
            moshi_lm,
            missing_keys=incompatible.missing_keys,
            unexpected_keys=incompatible.unexpected_keys,
            context="initializing MoshiForFinetuning from original LMModel",
        )

        if oracle_emb_missing:
            # Backward compatibility for checkpoints that do not contain oracle_emb.
            backfill_oracle_embedding_from_text(moshi_lm)

        # Store the kwargs for the later use
        moshi_lm.moshi_lm_kwargs = moshi_lm_kwargs

        return moshi_lm

    def to_original_moshi_lm(self) -> LMModel:
        """
        Convert the model to the original `LMModel`.
        """
        state_dict = self.state_dict()
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # Convert the state dict to the original format
        state_dict = restore_linear_weights_from_exposed_state_dict(state_dict)

        # Initialize the original model
        moshi_lm = LMModel(device=device, dtype=dtype, **self.moshi_lm_kwargs).to(
            device=device, dtype=dtype
        )
        moshi_lm.load_state_dict(state_dict, strict=True)

        return moshi_lm

    def save_pretrained(
        self,
        save_dir: str,
        oracle_embedding_mode: str | None = None,
    ):
        """
        Save the model to the given directory.

        If the model was trained with tie oracle embeddings, pass
        ``oracle_embedding_mode="tie"`` (or set ``self.oracle_embedding_mode``)
        so ``oracle_emb`` is materialized from ``text_emb`` before serialization.
        """
        if oracle_embedding_mode is None:
            oracle_embedding_mode = getattr(self, "oracle_embedding_mode", None)
        if oracle_embedding_mode is not None:
            self.materialize_oracle_embedding_for_checkpoint_(oracle_embedding_mode)

        os.makedirs(save_dir, exist_ok=True)
        # Save the model
        save_file(self.state_dict(), os.path.join(save_dir, "model.safetensors"))
        # Save the kwargs
        with open(os.path.join(save_dir, "moshi_lm_kwargs.json"), "w") as f:
            json.dump(self.moshi_lm_kwargs, f, indent=4)

    @classmethod
    def from_pretrained(
        cls,
        save_dir: str,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "MoshiForFinetuning":
        """
        Load the model from the given directory.
        """
        # Load the kwargs
        with open(os.path.join(save_dir, "moshi_lm_kwargs.json")) as f:
            moshi_lm_kwargs = json.load(f)
        # Initialize the model
        moshi_lm = cls(device=device, dtype=dtype, **moshi_lm_kwargs).to(device=device, dtype=dtype)

        model_path = os.path.join(save_dir, "model.safetensors")
        state_dict = load_file(model_path, device=str(device))
        incompatible = moshi_lm.load_state_dict(state_dict, strict=False)
        oracle_emb_missing = validate_oracle_embedding_checkpoint_load(
            moshi_lm,
            missing_keys=incompatible.missing_keys,
            unexpected_keys=incompatible.unexpected_keys,
            context=f"loading MoshiForFinetuning from {model_path}",
        )

        if oracle_emb_missing:
            print("[INFO] oracle_emb not found in checkpoint, initializing from text_emb")
            backfill_oracle_embedding_from_text(moshi_lm)

        moshi_lm.moshi_lm_kwargs = moshi_lm_kwargs

        return moshi_lm
