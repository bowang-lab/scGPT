from typing import Optional, Tuple

import torch
from torch import Tensor, nn

try:
    from einops import rearrange
except (ImportError, OSError):
    rearrange = None

flash_attn_backend: Optional[str] = None
flash_attn_error: Optional[str] = None

try:
    # flash-attn 1.x (preferred - simpler public API)
    from flash_attn.flash_attention import FlashMHA as _FlashMHA1

    _FlashMHA2 = None
    _fa2_qkvpacked_func = None
    _fa2_varlen_qkvpacked_func = None
    _fa2_unpad_input = None
    _fa2_pad_input = None
    flash_attn_backend = "fa1"
    flash_attn_error = None
except (ImportError, OSError) as exc:
    _FlashMHA1 = None
    flash_attn_error = str(exc)
    try:
        # flash-attn 2.x (fallback - use modules.mha.MHA)
        from flash_attn.modules.mha import MHA as _FlashMHA2
        from flash_attn import (
            flash_attn_qkvpacked_func as _fa2_qkvpacked_func,
            flash_attn_varlen_qkvpacked_func as _fa2_varlen_qkvpacked_func,
        )
        from flash_attn.bert_padding import (
            unpad_input as _fa2_unpad_input,
            pad_input as _fa2_pad_input,
        )

        flash_attn_backend = "fa2"
        flash_attn_error = None
    except (ImportError, OSError) as exc:
        _FlashMHA2 = None
        _fa2_qkvpacked_func = None
        _fa2_varlen_qkvpacked_func = None
        _fa2_unpad_input = None
        _fa2_pad_input = None
        flash_attn_error = str(exc)

flash_attn_available = flash_attn_backend is not None


def get_flash_attn_info() -> str:
    """
    Return a human-readable string describing the detected flash-attn backend.
    
    Returns:
        String describing the backend (e.g., "flash-attn 1.x", "flash-attn 2.x", 
        or "not installed").
        
    Note:
        FA1 is preferred when available for its simpler API.
        FA2 is used as fallback with a simplified wrapper (no rotary, MQA/GQA, etc.).
    """
    if flash_attn_backend == "fa1":
        return "flash-attn 1.x (preferred - FlashMHA with explicit FlashAttention wrapper)"
    elif flash_attn_backend == "fa2":
        return "flash-attn 2.x (fallback - MHA module with simplified wrapper interface)"
    elif flash_attn_error is not None:
        return f"flash-attn import failed: {flash_attn_error}"
    else:
        return "flash-attn not installed"


class FlashMHA(nn.Module):
    """
    Compatibility wrapper for flash-attn 1.x and 2.x MHA classes.
    
    **API Preference Strategy:**
    - **FA1 (Preferred):** Uses the simple public API from flash_attn.flash_attention.FlashMHA.
      This provides a straightforward interface for basic use cases.
    - **FA2 (Fallback):** Uses flash_attn.modules.mha.MHA but with a simplified wrapper that
      avoids FA2-specific complexity (rotary embeddings, MQA/GQA, varlen sequences, etc.)
      unless explicitly required. This ensures compatibility when only FA2 is available.

    Both backends perform optimized attention computation.
    FA2 path intentionally mirrors FA1 architecture layout:
    - `Wqkv` projection
    - `inner_attn` core
    - `out_proj` projection

    This preserves high-level semantics and parameter naming stability across
    FA1/FA2, instead of exposing FA2's lower-level `MHA` internals directly.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        bias=True,
        batch_first=True,
        attention_dropout=0.0,
        causal=False,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        """
        Initialize FlashMHA with parameters compatible with FA1 API.
        
        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            bias: Whether to use bias in projections (default: True).
            batch_first: Whether input is (batch, seq_len, hidden) (default: True).
            attention_dropout: Dropout probability in attention (default: 0.0).
            causal: Whether to use causal masking (default: False).
            device: Device for model parameters.
            dtype: Data type for model parameters.
            **kwargs: Additional backend-specific parameters (mostly for FA2 fallback).
            
        Implementation Details:
            - **FA1 path:** Directly instantiates FlashMHA from flash_attn.flash_attention.
            - **FA2 path:** Instantiates MHA from flash_attn.modules.mha with simplified
              configuration. Constructor parameters are introspected to avoid passing
              unsupported arguments (different FA2 versions vary in their accepted kwargs).
        
        Notes on key_padding_mask:
            - Uses flash-attn convention: True=keep/valid token, False=masked token.
            - This matches FA1 unpadding utilities and FA2 MHA behavior.
            - Callers that use PyTorch Transformer masks (True=pad) should invert
                the mask before calling this wrapper.
        """
        super().__init__()
        if not batch_first:
            raise ValueError("FlashMHA wrapper currently supports batch_first=True only.")

        if not flash_attn_available:
            raise ImportError(
                "flash_attn is not installed. Install flash-attn 1.x or 2.x to use "
                "fast transformer backend."
            )

        self.batch_first = batch_first
        self.backend = flash_attn_backend
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Validate dimensions
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        factory_kwargs = {"device": device, "dtype": dtype}

        if self.backend == "fa1":
            # FA1: Direct instantiation with simple API
            self._impl = _FlashMHA1(
                embed_dim=embed_dim,
                num_heads=num_heads,
                bias=bias,
                batch_first=batch_first,
                attention_dropout=attention_dropout,
                causal=causal,
                **factory_kwargs,
                **kwargs,
            )
        else:
            # FA2: Use FA1-shaped high-level wrapper on top of FA2 kernels.
            # This keeps module semantics and parameter structure stable.
            self._impl = _FA2FlashMHA(
                embed_dim=embed_dim,
                num_heads=num_heads,
                bias=bias,
                batch_first=batch_first,
                attention_dropout=attention_dropout,
                causal=causal,
                **factory_kwargs,
                **kwargs,
            )

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass with support for key padding mask.
        
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim).
            key_padding_mask: Optional bool mask of shape (batch, seq_len).
                             True indicates valid tokens to keep.
                             False indicates masked / padding tokens.
                             This follows flash-attn convention (FA1 and FA2).
                             If your upstream mask uses PyTorch Transformer semantics
                             (True=pad), invert it before passing to FlashMHA.
            need_weights: Whether to return attention weights (rarely supported 
                         efficiently in flash attention). Default: False.
            **kwargs: Additional backend-specific arguments.
            
        Returns:
            Tuple[output, attn_weights_or_none]:
                - output: Attention output of shape (batch, seq_len, embed_dim).
                - attn_weights_or_none: None (flash attention doesn't expose weights efficiently).
                
        Notes:
              * Both backends perform padding-aware attention internally:
              * Unpad input to remove padding tokens
              * Run flash attention on the packed sequence
              * Pad output back to original shape
            - This is much more efficient than masking directly.
        """
        orig_dtype = x.dtype

        if key_padding_mask is not None and key_padding_mask.dtype != torch.bool:
            key_padding_mask = key_padding_mask.bool()

        # Both backend implementations support the same high-level forward contract.
        if self.backend == "fa1":
            # Some flash-attn 1.x builds may not expose need_weights.
            try:
                out = self._impl(
                    x,
                    key_padding_mask=key_padding_mask,
                    need_weights=need_weights,
                    **kwargs,
                )
            except TypeError as e:
                if "need_weights" in str(e):
                    out = self._impl(
                        x,
                        key_padding_mask=key_padding_mask,
                        **kwargs,
                    )
                else:
                    raise
        else:
            out = self._impl(
                x,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                **kwargs,
            )

        if isinstance(out, tuple):
            out_tensor, attn_weights = out
            if out_tensor.dtype != orig_dtype:
                out_tensor = out_tensor.to(orig_dtype)
            return out_tensor, attn_weights
        
        # Normalize output to always be (output, attn_weights_or_none)
        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        return out, None


class _FA2FlashAttention(nn.Module):
    """FA2 attention core with FA1-style behavior and mask handling.

    Supports:
    - Packed path: qkv shape (B, S, 3, H, D) when no key_padding_mask.
    - Varlen path: unpad/pad around flash_attn_varlen_qkvpacked_func when
      key_padding_mask is provided (True=keep, False=pad).
    """

    def __init__(self, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(
        self,
        qkv: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        causal: bool = False,
        cu_seqlens: Optional[Tensor] = None,
        max_s: Optional[int] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if need_weights:
            raise ValueError("flash-attn does not efficiently return attention weights")
        if qkv.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"FA2 flash path requires fp16/bf16 qkv, got {qkv.dtype}. "
                "Use torch.amp.autocast or explicit casting."
            )
        if not qkv.is_cuda:
            raise ValueError("FA2 flash path requires CUDA tensors")

        dropout_p = self.dropout_p if self.training else 0.0

        # Explicit varlen mode (already unpadded by caller).
        if cu_seqlens is not None:
            if _fa2_varlen_qkvpacked_func is None:
                raise ImportError("flash_attn_varlen_qkvpacked_func not available")
            if max_s is None:
                raise ValueError("max_s must be provided when cu_seqlens is set")
            out = _fa2_varlen_qkvpacked_func(
                qkv,
                cu_seqlens,
                max_s,
                dropout_p,
                softmax_scale=self.softmax_scale,
                causal=causal,
            )
            return out, None

        # Packed path without mask.
        if key_padding_mask is None:
            if _fa2_qkvpacked_func is None:
                raise ImportError("flash_attn_qkvpacked_func not available")
            out = _fa2_qkvpacked_func(
                qkv,
                dropout_p,
                softmax_scale=self.softmax_scale,
                causal=causal,
            )
            return out, None

        # Packed path with mask -> unpad to varlen -> flash -> pad back.
        if _fa2_unpad_input is None or _fa2_pad_input is None:
            raise ImportError("flash_attn.bert_padding (unpad_input/pad_input) not available")
        if _fa2_varlen_qkvpacked_func is None:
            raise ImportError("flash_attn_varlen_qkvpacked_func not available")
        if rearrange is None:
            raise ImportError("einops is required for FA2 varlen wrapper path")

        batch_size, seqlen, _, nheads, _ = qkv.shape
        x = rearrange(qkv, "b s three h d -> b s (three h d)")
        unpad_out = _fa2_unpad_input(x, key_padding_mask)
        if len(unpad_out) == 4:
            x_unpad, indices, cu_seqlens, max_s = unpad_out
        elif len(unpad_out) == 5:
            # Newer FA2 builds return an extra used-seqlens tensor.
            x_unpad, indices, cu_seqlens, max_s, _ = unpad_out
        else:
            raise RuntimeError(
                f"Unexpected unpad_input output tuple length: {len(unpad_out)}"
            )
        qkv_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)

        out_unpad = _fa2_varlen_qkvpacked_func(
            qkv_unpad,
            cu_seqlens,
            max_s,
            dropout_p,
            softmax_scale=self.softmax_scale,
            causal=causal,
        )
        out = rearrange(
            _fa2_pad_input(rearrange(out_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen),
            "b s (h d) -> b s h d",
            h=nheads,
        )
        return out, None


class _FA2FlashMHA(nn.Module):
    """FA1-shaped high-level MHA wrapper implemented with FA2 kernels.

    Mirrors FA1 public structure and parameter names:
    - Wqkv
    - inner_attn
    - out_proj
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        bias=True,
        batch_first=True,
        attention_dropout=0.0,
        causal=False,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        if kwargs:
            # Keep behavior explicit: this wrapper intentionally exposes a narrow
            # FA1-style API surface.
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unsupported kwargs for _FA2FlashMHA: {unexpected}")
        if not batch_first:
            raise ValueError("_FA2FlashMHA supports batch_first=True only")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.inner_attn = _FA2FlashAttention(attention_dropout=attention_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unsupported kwargs for _FA2FlashMHA.forward: {unexpected}")
        if rearrange is None:
            raise ImportError("einops is required for _FA2FlashMHA")

        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads)
        context, attn_weights = self.inner_attn(
            qkv,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            causal=self.causal,
        )
        out = self.out_proj(rearrange(context, "b s h d -> b s (h d)"))
        return out, attn_weights


def get_flash_attn_parameter_rename_rules(
    checkpoint_state_dict: dict,
    current_backend: str = None,
) -> dict:
    """
    Detect FA1 ↔ FA2 parameter naming mismatches and generate rename rules.
    
    **Issue Background:**
    Historical checkpoints may use one of two naming layouts:
    - Direct layout: `self_attn.Wqkv.weight`, `self_attn.out_proj.weight`, etc.
    - Wrapped layout: `self_attn._impl.Wqkv.weight`, `self_attn._impl.out_proj.weight`, etc.

    Current compatibility wrapper stores attention parameters under wrapped
    layout (`self_attn._impl.*`) for both FA1 and FA2 backends.
    
    This function detects which version the checkpoint uses and generates rename rules
    to convert to the current backend if needed.
    
    Args:
        checkpoint_state_dict: The checkpoint state dict to inspect.
        current_backend: Current flash-attn backend ("fa1", "fa2", or None for auto).
                        If None, uses global flash_attn_backend. If None and no
                        flash-attn available, returns empty dict.
    
    Returns:
        Dict of rename rules {old_pattern: new_pattern} for regex-based renaming
        in flexible_load_model_weights(..., rename_rules=...).
        Empty dict if no mismatch detected or conversion not needed.
        
    Example:
        >>> ckpt = torch.load("old_fa1_model.pt")
        >>> rules = get_flash_attn_parameter_rename_rules(ckpt, current_backend="fa2")
        >>> # rules = {r"self_attn\.Wqkv\.": "self_attn._impl.Wqkv.",
        >>> #          r"self_attn\.out_proj\.": "self_attn._impl.out_proj."}
    """
    if current_backend is None:
        current_backend = flash_attn_backend
    
    if current_backend is None or not checkpoint_state_dict:
        return {}
    
    # Detect which version the checkpoint uses by examining parameter names
    has_impl_wqkv = any("self_attn._impl.Wqkv" in k for k in checkpoint_state_dict.keys())
    has_direct_wqkv = any(
        "self_attn.Wqkv" in k and "self_attn._impl" not in k 
        for k in checkpoint_state_dict.keys()
    )
    
    # Current target layout (both FA1/FA2): wrapped keys under self_attn._impl.*
    if has_impl_wqkv:
        return {}
    
    # Mismatch: need conversion
    rename_rules = {}
    
    if has_direct_wqkv:
        # Convert direct layout -> wrapped layout
        rename_rules[r"self_attn\.Wqkv\."] = "self_attn._impl.Wqkv."
        rename_rules[r"self_attn\.out_proj\."] = "self_attn._impl.out_proj."
    
    return rename_rules