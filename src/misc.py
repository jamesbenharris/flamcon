"""
Util functions for setting up distributed training.
Credit: https://github.com/mlfoundations/open_clip/blob/main/src/training/train_utils.py
& https://huggingface.co/tiiuae/falcon-7b/blob/main/modelling_RW.py
"""

import os
import math
import torch
from torch import Size, device, Tensor, BoolTensor
from typing import Optional, Tuple, Union
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
)

def filter_state_dict_to_trainable(model,state_dict):
    for (
        name,
        p,
    ) in model.named_parameters():  # won't work for fsdp + use_orig_params=False
        if "fsdp" in name:
            continue
        if "embed" in name or isinstance(p, torch.nn.Embedding):
            continue
        if not p.requires_grad:
            if name in state_dict:
                del state_dict[name]
            else:
                print(f"WARNING: filtering but {name} not in state_dict")

    delete = [
        n
        for n in state_dict.keys()
        if ("model.transformer.h" in n)
    ]
    for name in delete:
        del state_dict[name]
    return state_dict


def save_checkpoint(model, optimizer, lr_scheduler, epoch, args, log):
    """
    Save training checkpoint with model, optimizer, and lr_scheduler state.
    """
    if args.fsdp:
        FSDP.set_state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        )
        model_state = model.state_dict()
    else:
        model_state = model.state_dict()

    if args.rank == 0:
        model_state = filter_state_dict_to_trainable(model,model_state)
        if not os.path.exists(args.run_name):
            os.makedirs(args.run_name)

        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        }

        log.info(f"Saving checkpoint to {args.run_name}/checkpoint_{epoch}.pt\n")
        torch.save(checkpoint_dict, f"{args.run_name}/checkpoint_{epoch}.pt")

        if args.delete_previous_checkpoint:
            if epoch > 0:
                if os.path.exists(f"{args.run_name}/checkpoint_{args.last_saved_epoch}.pt"):
                    os.remove(f"{args.run_name}/checkpoint_{args.last_saved_epoch}.pt")
                
                
def build_alibi_tensor(attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
        batch_size, seq_length = attention_mask.shape
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        base = torch.tensor(
            2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
        )
        powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
        slopes = torch.pow(base, powers)

        if closest_power_of_2 != num_heads:
            extra_base = torch.tensor(
                2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
            )
            num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
            extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32)
            slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

        # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
        # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
        # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
        # => the query_length dimension will then be broadcasted correctly
        # This is more or less identical to T5's relative position bias:
        # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
        arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
        alibi = slopes[..., None].bfloat16() * arange_tensor
        return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)
    
def _make_causal_mask(
        input_ids_shape: Size, device: device, past_key_values_length: int
    ) -> BoolTensor:
    batch_size, target_length = input_ids_shape
    mask = torch.empty((target_length, target_length + past_key_values_length), dtype=torch.bool, device=device)
    # ONNX doesn't support `torch.Tensor.triu` properly, thus we use this workaround
    seq_ids = torch.arange(target_length,device=device)
    mask[:, past_key_values_length:] = seq_ids[:, None] < seq_ids[None, :]

    if past_key_values_length > 0:
        mask[:, :past_key_values_length] = False

    expanded_mask = mask[None, None, :, :].expand(batch_size, 1, target_length, target_length + past_key_values_length)
    return expanded_mask


def _expand_mask(mask: Tensor, tgt_length: int) -> BoolTensor:
    batch_size, src_length = mask.shape
    tgt_length = tgt_length if tgt_length is not None else src_length

    expanded_mask = ~(mask[:, None, None, :].to(torch.bool))
    return expanded_mask.expand(batch_size, 1, tgt_length, src_length)
    
def _prepare_attn_mask(
            attention_mask: Tensor, input_shape: Tuple[int, int], past_key_values_length: int,
        ) -> BoolTensor:
            # create causal mask
            # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        combined_attention_mask = None
        device = attention_mask.device
        _, src_length = input_shape

        if src_length > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, device=device, past_key_values_length=past_key_values_length
            )

            # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask | combined_attention_mask
        )
        return combined_attention_mask