import json
from enum import Enum
from pathlib import Path
from typing import Dict

import torch
import torch.nn.init
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from safetensors.torch import load_file
from torch import Tensor, nn

from ifo.common.layers.transformer import LayerScale, PatchEmbed

# Path to the local_cls_norm constants for vit7b16 (not available in HuggingFace checkpoint)
_VIT7B16_LOCAL_CLS_NORM_PATH = Path(__file__).parent.parent / "nets/artifacts/dinov3_vit7b16_local_cls_norm.pt"


def _load_safetensors_from_hub(repo_id: str) -> Dict[str, torch.Tensor]:
    """Load safetensors checkpoint from HuggingFace Hub.

    Handles both single-file checkpoints (model.safetensors) and
    sharded checkpoints (model-00001-of-00006.safetensors, etc.).

    Args:
        repo_id: HuggingFace repository ID (e.g., "facebook/dinov3-vits16-pretrain-lvd1689m")

    Returns:
        The loaded state dictionary.
    """
    try:
        # Try single file first
        checkpoint_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        return load_file(checkpoint_path)
    except EntryNotFoundError:
        # Fall back to sharded checkpoint
        index_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors.index.json")

        with open(index_path, "r") as f:
            index = json.load(f)

        # Get unique shard files
        shard_files = sorted(set(index["weight_map"].values()))

        # Download and load all shards
        state_dict = {}
        for shard_file in shard_files:
            shard_path = hf_hub_download(repo_id=repo_id, filename=shard_file)
            shard_dict = load_file(shard_path)
            state_dict.update(shard_dict)

        return state_dict


class Weights(Enum):
    """Enumeration of available pretrained weight options for DINOv3 models."""
    LVD1689M = "LVD1689M"
    SAT493M = "SAT493M"


def _rename_huggingface_to_legacy_keys(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Rename Hugging Face to legacy keys in a state dictionary.

    Hugging Face uses a different key structure than the legacy checkpoint format.
    This function renames the keys to the legacy format.

    Undoes: convert_and_test_dinov3_checkpoint
    https://github.com/huggingface/transformers/blob/a7f29523361b2cc12e51c1f5133d95f122f6f45c/
    src/transformers/models/dinov3_vit/convert_dinov3_vit_to_hf.py#L208
    """
    # Detect if this is a SwiGLU model by checking for gate_proj keys
    # SwiGLU has: gate_proj, up_proj, down_proj -> w1, w2, w3
    # MLP has: up_proj, down_proj -> fc1, fc2
    is_swiglu = any("gate_proj" in key for key in state_dict.keys())

    keys = list(state_dict.keys())
    for key in keys:
        if key.startswith("layer."):
            new_key = key.replace("layer.", "blocks.")
            if "attention" in key:
                new_key = new_key.replace("attention", "attn")
                if "o_proj" in new_key:
                    new_key = new_key.replace("o_proj", "proj")
            elif "mlp.gate_proj" in key:
                new_key = new_key.replace("mlp.gate_proj", "mlp.w1")
            elif "mlp.up_proj" in key:
                if is_swiglu:
                    new_key = new_key.replace("mlp.up_proj", "mlp.w2")
                else:
                    new_key = new_key.replace("mlp.up_proj", "mlp.fc1")
            elif "mlp.down_proj" in key:
                if is_swiglu:
                    new_key = new_key.replace("mlp.down_proj", "mlp.w3")
                else:
                    new_key = new_key.replace("mlp.down_proj", "mlp.fc2")
            elif "layer_scale" in key:
                new_key = new_key.replace("layer_scale", "ls")
                new_key = new_key.replace("lambda1", "gamma")
        elif key.startswith("embeddings."):
            new_key = key.replace("embeddings.", "", 1)
            if "register_tokens" in key:
                new_key = new_key.replace("register_tokens", "storage_tokens")
            elif "patch_embeddings" in key:
                new_key = new_key.replace("patch_embeddings", "patch_embed.proj")
        else:
            new_key = key
        state_dict[new_key] = state_dict.pop(key)

    # Combine single q, k, v state dict weights and biases to a single qkv state dict weight and bias
    q_bias_keys = sorted([key for key in state_dict.keys() if key.endswith("q_proj.bias")])
    k_bias_keys = sorted([key for key in state_dict.keys() if key.endswith("k_proj.bias")])
    v_bias_keys = sorted([key for key in state_dict.keys() if key.endswith("v_proj.bias")])

    assert len(q_bias_keys) == len(v_bias_keys), "Number of q and v bias keys must match"
    # Key bias is optional, since it only is a shift in the attention calculation, as the softmax calculation is
    # invariant to shift, we can replace it by an arbitrary value (zero in this case), without changing the attention
    # calculation.
    if not k_bias_keys:
        k_bias_keys = [q.replace("q_proj.bias", "k_proj.bias") for q in q_bias_keys]
    for q, k, v in zip(q_bias_keys, k_bias_keys, v_bias_keys):
        q_bias = state_dict.pop(q)
        v_bias = state_dict.pop(v)
        k_bias = state_dict.pop(k, torch.zeros_like(q_bias))
        qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
        state_dict[q.replace("q_proj.bias", "qkv.bias")] = qkv_bias

    q_weight_keys = sorted([key for key in state_dict.keys() if key.endswith("q_proj.weight")])
    k_weight_keys = sorted([key for key in state_dict.keys() if key.endswith("k_proj.weight")])
    v_weight_keys = sorted([key for key in state_dict.keys() if key.endswith("v_proj.weight")])

    assert len(q_weight_keys) == len(k_weight_keys) == len(v_weight_keys), "Number of q, k and v weight keys must match"
    for q, k, v in zip(q_weight_keys, k_weight_keys, v_weight_keys):
        q_weight = state_dict.pop(q)
        v_weight = state_dict.pop(v)
        k_weight = state_dict.pop(k)
        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        state_dict[q.replace("q_proj.weight", "qkv.weight")] = qkv_weight

    # Reshape mask token: HF uses (1, 1, embed_dim), legacy uses (1, embed_dim)
    state_dict["mask_token"] = state_dict.pop("mask_token").squeeze(1)
    return state_dict


def _add_missing_keys(state_dict: Dict[str, Tensor], model: nn.Module) -> Dict[str, Tensor]:
    """Add missing keys to the state dictionary of a HuggingFace converted checkpoint.

    Args:
        state_dict (Dict[str, Tensor]): State dictionary to add missing keys to.
        model (nn.Module): Model to get the missing keys from.

    Returns:
        Dict[str, Tensor]: State dictionary with missing keys added.
    """
    # Rope periods are missing in the huggingface checkpoint, so we add them back.
    # Initial training has been carried out inf bfloat16, so we convert the periods to bfloat16 and then back to
    # the original dtype for all models except vit7b16.
    state_dict["rope_embed.periods"] = model.state_dict()["rope_embed.periods"]
    if not model.untie_global_and_local_cls_norm:
        state_dict["rope_embed.periods"] = state_dict["rope_embed.periods"].to(torch.bfloat16).to(model.rope_embed.dtype)

    # Bias mask is missing in the huggingface checkpoint, so we add it back.
    # In the original the mask is all zeros.
    for k, v in model.state_dict().items():
        if k.endswith("bias_mask"):
            state_dict[k] = torch.zeros_like(v)

    # Local CLS norm is missing in the huggingface checkpoint, so we add it back for vit7b16.
    if model.untie_global_and_local_cls_norm:
        local_cls_norm_data = torch.load(_VIT7B16_LOCAL_CLS_NORM_PATH, map_location="cpu", weights_only=True)
        state_dict["local_cls_norm.weight"] = local_cls_norm_data["local_cls_norm.weight"]
        state_dict["local_cls_norm.bias"] = local_cls_norm_data["local_cls_norm.bias"]
    return state_dict


def init_weights_vit(module: nn.Module, name: str = "") -> None:
    """Initialize weights for Vision Transformer modules.

    Applies appropriate initialization strategies for different module types:
    - Linear layers: truncated normal with std=0.02
    - LayerNorm/RMSNorm: calls reset_parameters()
    - LayerScale: calls reset_parameters()
    - PatchEmbed: calls reset_parameters()

    For Linear layers with bias_mask (used in masked attention), sets the middle
    third of the mask to 0 (corresponding to key biases).

    Args:
        module (nn.Module): Module to initialize.
        name (str): Name of the module (unused, kept for compatibility). Defaults to "".
    """
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if hasattr(module, "bias_mask") and module.bias_mask is not None:
            o = module.out_features
            module.bias_mask.fill_(1)  # type: ignore[union-attr]
            module.bias_mask[o // 3 : 2 * o // 3].fill_(0)  # type: ignore[index]
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    if isinstance(module, nn.RMSNorm):
        module.reset_parameters()