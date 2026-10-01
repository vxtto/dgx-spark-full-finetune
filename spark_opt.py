"""Throughput optimizations for full fine-tuning on the DGX Spark (GB10, sm_121).

Enabled in open_instruct/finetune.py by SPARK_OPT_CONFIG=<yaml> (hook added by
patches/finetune-spark.patch) and exercised on its own by
scripts/bench_spark_opt.py, so the benchmark measures exactly what trains.
Measurements behind every choice are in docs/results.md, section
"Throughput".

apply() does, in order:

1. Attention: pins SDPA to FlashAttention-2. For causal GQA at 4k-28k tokens it
   beat cuDNN 9.19 by 5-11% and FlexAttention by ~2x; the memory-efficient
   backend has no GQA kernel. cuDNN and mem-efficient are disabled so nothing
   switches silently; math stays as the correct-but-slow fallback.
2. Liger fused linear cross-entropy: replaces the lm_head gradient update.
   Liger 0.8.0 runs `grad_weight += mm(...).float()` into a bf16 buffer once
   per ~512-token chunk, ~10 GB of memory traffic per chunk and 9% of GPU time
   in sft_qwen3_8b_run1. Here chunks are `chunk_tokens` long and each update is
   one in-place fp32 addmm, which is also more precise.
3. FP8: torchao tensorwise Float8Linear on every decoder linear. lm_head stays
   BF16 (Liger FLCE reads its weight directly, bypassing the module).
4. torch.compile per decoder layer (dynamic shapes) with a torch._check that the
   token count is a multiple of 16: the FP8 weight-gradient GEMM contracts over
   tokens and _scaled_mm needs that dim % 16 == 0.
5. Pads every micro-batch to that multiple, after the sequence with labels -100,
   so causal attention leaves the real tokens' outputs and loss unchanged.
6. Adaptive gradient checkpointing: per micro-batch, the last n layers skip
   checkpointing, with n the largest that keeps their extra saved activations
   inside `activation_budget_gib`. Short sequences skip it on every layer.

Liger's in-layer kernels (RMSNorm, RoPE, SwiGLU) must be OFF when the model is
loaded: dynamo in torch 2.11 asserts while tracing LigerRMSNormFunction. Inductor
fuses the stock HF modules instead, at the same speed (measured).
"""

from collections import defaultdict, deque

import torch
import yaml

# Window counters, drained by log_metrics(); history is for the benchmark report.
_STATS = defaultdict(float)
_HISTORY = deque(maxlen=4096)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def apply(model, cfg):
    """Apply every optimization in `cfg` to a loaded HF causal LM, in place.

    Call after model.gradient_checkpointing_enable() and before the optimizer is
    built and accelerator.prepare() runs.
    """
    layers = model.model.layers
    multiple = cfg["pad_to_multiple_of"]

    _pin_attention(cfg["attention"])
    _patch_liger_flce(cfg["flce"])
    if cfg["fp8"]["enabled"]:
        _convert_fp8(model, cfg["fp8"], multiple)

    def tokens_divisible(module, args, kwargs):
        hidden_states = args[0] if args else kwargs["hidden_states"]
        torch._check(hidden_states.size(1) % multiple == 0)

    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
    for layer in layers:
        layer.register_forward_pre_hook(tokens_divisible, with_kwargs=True)
        if cfg["compile"]["enabled"]:
            layer.compile(dynamic=cfg["compile"]["dynamic"])

    checkpointing = any(layer.gradient_checkpointing for layer in layers)
    ckpt_cfg = cfg["checkpointing"]
    pad_token_id = model.config.pad_token_id
    if pad_token_id is None:
        pad_token_id = model.config.eos_token_id
    if isinstance(pad_token_id, list):
        pad_token_id = pad_token_id[0]

    def prepare_micro_batch(module, args, kwargs):
        if args:
            raise TypeError("spark_opt expects keyword inputs, as in model(**batch)")
        seq_len = kwargs["input_ids"].shape[1]
        pad = -seq_len % multiple
        if pad:
            kwargs = _pad_inputs(kwargs, seq_len, pad, pad_token_id)

        free = 0
        if checkpointing and module.training:
            free = _layers_without_checkpointing(len(layers), seq_len + pad, ckpt_cfg)
            for i, layer in enumerate(layers):
                layer.gradient_checkpointing = i < len(layers) - free

        _STATS["micro_batches"] += 1
        _STATS["tokens"] += seq_len
        _STATS["pad_tokens"] += pad
        _STATS["layers_not_checkpointed"] += free
        _HISTORY.append({"tokens": seq_len, "pad": pad, "layers_not_checkpointed": free})
        return args, kwargs

    model.register_forward_pre_hook(prepare_micro_batch, with_kwargs=True)
    return model


def log_metrics():
    """Per-micro-batch averages since the last call, keyed for wandb."""
    n = max(_STATS["micro_batches"], 1)
    calls = max(_STATS["flce_calls"], 1)
    metrics = {
        "spark_opt/pad_tokens_per_micro_batch": _STATS["pad_tokens"] / n,
        "spark_opt/layers_not_checkpointed": _STATS["layers_not_checkpointed"] / n,
        "spark_opt/flce_chunks_per_call": _STATS["flce_chunks"] / calls,
        "spark_opt/flce_fallback_calls": _STATS["flce_fallback_calls"],
    }
    _STATS.clear()
    return metrics


def history():
    return list(_HISTORY)


def _layers_without_checkpointing(n_layers, tokens, ckpt_cfg):
    if not ckpt_cfg["adaptive"]:
        return 0
    per_layer = tokens * ckpt_cfg["bytes_per_token_layer"] * ckpt_cfg["safety_factor"]
    return min(n_layers, int(ckpt_cfg["activation_budget_gib"] * 2**30 // per_layer))


def _pad_inputs(kwargs, seq_len, pad, pad_token_id):
    kwargs = dict(kwargs)
    for key, value in kwargs.items():
        if not (torch.is_tensor(value) and value.dim() == 2 and value.shape[1] == seq_len):
            continue
        if key == "input_ids":
            kwargs[key] = torch.nn.functional.pad(value, (0, pad), value=pad_token_id)
        elif key == "labels":
            kwargs[key] = torch.nn.functional.pad(value, (0, pad), value=-100)
        elif key == "attention_mask":
            # 1, not 0: an all-ones mask keeps HF on the is_causal path that
            # FlashAttention accepts; pads sit after every real token.
            kwargs[key] = torch.nn.functional.pad(value, (0, pad), value=1)
        elif key == "position_ids":
            tail = value[:, -1:] + torch.arange(1, pad + 1, device=value.device)
            kwargs[key] = torch.cat([value, tail], dim=1)
        else:
            raise ValueError(f"spark_opt does not know how to pad per-token input {key!r}")
    return kwargs


def _pin_attention(cfg):
    if cfg["sdpa_backend"] != "flash":
        raise ValueError(f"unsupported sdpa_backend {cfg['sdpa_backend']!r}; only 'flash' is measured")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(cfg["allow_math_fallback"])


def _convert_fp8(model, cfg, multiple):
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training

    skip = set(cfg["skip_modules"])
    convert_to_float8_training(
        model,
        config=Float8LinearConfig.from_recipe_name(cfg["recipe"]),
        module_filter_fn=lambda mod, fqn: fqn not in skip
        and mod.in_features % multiple == 0
        and mod.out_features % multiple == 0,
    )


def _patch_liger_flce(cfg):
    import triton
    from liger_kernel.ops import fused_linear_cross_entropy as flce
    from liger_kernel.ops.cross_entropy import liger_cross_entropy_kernel

    original = flce.fused_linear_cross_entropy_forward
    if getattr(original, "spark_opt", False):
        return
    chunk_tokens = cfg["chunk_tokens"]

    def forward(
        _input,
        weight,
        target,
        ce_weight=None,
        bias=None,
        ignore_index=-100,
        lse_square_scale=0.0,
        label_smoothing=0.0,
        reduction="mean",
        softcap=None,
        return_z_loss=False,
        accum_dtype=None,
        use_token_scaling=False,
        return_token_accuracy=False,
        return_predicted_tokens=False,
    ):
        unsupported = (
            ce_weight is not None
            or bias is not None
            or softcap is not None
            or lse_square_scale
            or label_smoothing
            or reduction == "none"
            or return_z_loss
            or use_token_scaling
            or return_token_accuracy
            or return_predicted_tokens
        )
        if unsupported:
            _STATS["flce_fallback_calls"] += 1
            return original(
                _input=_input, weight=weight, target=target, ce_weight=ce_weight, bias=bias,
                ignore_index=ignore_index, lse_square_scale=lse_square_scale,
                label_smoothing=label_smoothing, reduction=reduction, softcap=softcap,
                return_z_loss=return_z_loss, accum_dtype=accum_dtype,
                use_token_scaling=use_token_scaling, return_token_accuracy=return_token_accuracy,
                return_predicted_tokens=return_predicted_tokens,
            )

        n_tokens, hidden = _input.shape
        vocab = weight.shape[0]
        block_size = min(flce.MAX_FUSED_SIZE, triton.next_power_of_2(vocab))
        needs_grad = _input.requires_grad

        grad_input = torch.zeros_like(_input)
        grad_weight = None
        if needs_grad and weight.requires_grad:
            grad_weight = torch.zeros(vocab, hidden, dtype=torch.float32, device=_input.device)
        loss_1d = torch.zeros(n_tokens, dtype=torch.float32, device=_input.device)
        n_non_ignore = (target != ignore_index).sum().item()

        for start in range(0, n_tokens, chunk_tokens):
            end = min(start + chunk_tokens, n_tokens)
            x = _input[start:end]
            logits = (x @ weight.t()).contiguous()
            target_chunk = target[start:end].contiguous()
            loss_chunk = loss_1d[start:end]
            # Same kernel and arguments as Liger with every optional feature off;
            # it writes d(loss)/d(logits) into `logits` in place.
            liger_cross_entropy_kernel[(end - start,)](
                X_ptr=logits,
                X_stride=logits.stride(-2),
                Y_ptr=target_chunk,
                Y_stride=target_chunk.stride(-1),
                weight_ptr=None,
                loss_ptr=loss_chunk,
                z_loss_ptr=None,
                loss_stride=loss_chunk.stride(-1),
                token_accuracy_ptr=None,
                token_accuracy_stride=0,
                predicted_tokens_ptr=None,
                predicted_tokens_stride=0,
                n_cols=vocab,
                n_non_ignore=n_non_ignore,
                sum_non_ignore_weight=n_non_ignore,
                weight_sum=0.0,
                ignore_index=ignore_index,
                lse_square_scale=0.0,
                label_smoothing=0.0,
                reduction=reduction,
                softcap=None,
                RETURN_Z_LOSS=False,
                RETURN_TOKEN_ACCURACY=False,
                RETURN_PREDICTED_TOKENS=False,
                HAS_WEIGHT=False,
                HAS_SOFTCAPPING=False,
                HAS_GRADIENTS=needs_grad,
                BLOCK_SIZE=block_size,
                num_warps=32,
            )
            if needs_grad:
                grad_input[start:end] = logits @ weight
                if grad_weight is not None:
                    torch.addmm(grad_weight, logits.t(), x, out_dtype=torch.float32, out=grad_weight)
            _STATS["flce_chunks"] += 1
        _STATS["flce_calls"] += 1

        if grad_weight is not None:
            grad_weight = grad_weight.to(weight.dtype)
        return loss_1d.sum(), None, None, None, grad_input, grad_weight, None

    forward.spark_opt = True
    forward.original = original
    flce.fused_linear_cross_entropy_forward = forward
