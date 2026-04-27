"""
Test suite for flash-attn compatibility wrapper.

This validates that the FlashMHA wrapper correctly supports both
flash-attn 1.x and 2.x APIs, especially for the CUDA 12.8 + flash-attn 2.8.x
upgrade path.
"""

import pytest
import torch
from torch import nn

from scgpt.model.flash_attn_compat import (
    FlashMHA,
    flash_attn_available,
    flash_attn_backend,
    get_flash_attn_info,
)
from scgpt.utils import load_pretrained


class _FakeWrappedSelfAttnImpl(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wqkv = nn.Linear(4, 12, bias=True)
        self.out_proj = nn.Linear(4, 4, bias=True)


class _FakeWrappedSelfAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self._impl = _FakeWrappedSelfAttnImpl()


class _FakeFlashCheckpointTarget(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _FakeWrappedSelfAttn()
        self.use_fast_transformer = True


class _FakeTorchCheckpointTarget(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=4,
            num_heads=1,
            batch_first=True,
        )
        self.use_fast_transformer = False


def _amp_dtype_for_device() -> torch.dtype:
    """Pick a stable mixed-precision dtype for CUDA tests."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _run_with_amp(fn):
    """Run fn() under CUDA autocast when CUDA is available."""
    if not torch.cuda.is_available():
        return fn()
    amp_dtype = _amp_dtype_for_device()
    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
        return fn()


def test_flash_attn_availability():
    """Test that flash-attn availability flag is set correctly."""
    info = get_flash_attn_info()
    print(f"\nFlash-attn backend: {info}")

    # Minimal sanity checks for backend detection API.
    assert isinstance(info, str) and len(info) > 0
    assert flash_attn_backend in {None, "fa1", "fa2"}

    # Consistency checks between availability flag and backend value.
    if flash_attn_available:
        assert flash_attn_backend in {"fa1", "fa2"}
        assert "flash-attn" in info
    else:
        assert flash_attn_backend is None


def test_load_pretrained_renames_direct_fa_attention_keys():
    """Direct-layout FA checkpoints should load into wrapped flash-attn modules."""
    model = _FakeFlashCheckpointTarget()

    wqkv_weight = torch.full_like(model.self_attn._impl.Wqkv.weight, 1.25)
    wqkv_bias = torch.full_like(model.self_attn._impl.Wqkv.bias, -0.5)
    out_proj_weight = torch.full_like(model.self_attn._impl.out_proj.weight, 0.75)
    out_proj_bias = torch.full_like(model.self_attn._impl.out_proj.bias, 0.125)

    checkpoint = {
        "self_attn.Wqkv.weight": wqkv_weight,
        "self_attn.Wqkv.bias": wqkv_bias,
        "self_attn.out_proj.weight": out_proj_weight,
        "self_attn.out_proj.bias": out_proj_bias,
    }

    load_pretrained(model, checkpoint, verbose=False)

    assert torch.equal(model.self_attn._impl.Wqkv.weight, wqkv_weight)
    assert torch.equal(model.self_attn._impl.Wqkv.bias, wqkv_bias)
    assert torch.equal(model.self_attn._impl.out_proj.weight, out_proj_weight)
    assert torch.equal(model.self_attn._impl.out_proj.bias, out_proj_bias)


def test_load_pretrained_renames_wrapped_flash_keys_for_torch_mha():
    """Wrapped flash-attn checkpoints should load into non-flash PyTorch MHA targets."""
    model = _FakeTorchCheckpointTarget()

    in_proj_weight = torch.full_like(model.self_attn.in_proj_weight, 2.0)
    in_proj_bias = torch.full_like(model.self_attn.in_proj_bias, -1.0)
    out_proj_weight = torch.full_like(model.self_attn.out_proj.weight, 0.5)
    out_proj_bias = torch.full_like(model.self_attn.out_proj.bias, 0.25)

    checkpoint = {
        "self_attn._impl.Wqkv.weight": in_proj_weight,
        "self_attn._impl.Wqkv.bias": in_proj_bias,
        "self_attn._impl.out_proj.weight": out_proj_weight,
        "self_attn._impl.out_proj.bias": out_proj_bias,
    }

    load_pretrained(model, checkpoint, verbose=False)

    assert torch.equal(model.self_attn.in_proj_weight, in_proj_weight)
    assert torch.equal(model.self_attn.in_proj_bias, in_proj_bias)
    assert torch.equal(model.self_attn.out_proj.weight, out_proj_weight)
    assert torch.equal(model.self_attn.out_proj.bias, out_proj_bias)


@pytest.mark.skipif(not flash_attn_available, reason="flash-attn not installed")
def test_flash_mha_instantiation():
    """Test FlashMHA instantiation with standard parameters."""
    embed_dim = 256
    num_heads = 8

    mha = FlashMHA(
        embed_dim=embed_dim,
        num_heads=num_heads,
        bias=True,
        batch_first=True,
        attention_dropout=0.1,
        causal=False,
    )

    assert hasattr(mha, "batch_first")
    assert hasattr(mha, "backend")
    assert hasattr(mha, "embed_dim")
    assert mha.embed_dim == embed_dim
    assert mha.num_heads == num_heads


@pytest.mark.skipif(not flash_attn_available, reason="flash-attn not installed")
def test_flash_mha_dimension_validation():
    """Test that invalid embed_dim/num_heads raises error."""
    with pytest.raises(ValueError, match="must be divisible by"):
        FlashMHA(
            embed_dim=255,  # Not divisible by 8
            num_heads=8,
            batch_first=True,
        )


@pytest.mark.skipif(not flash_attn_available, reason="flash-attn not installed")
def test_flash_mha_batch_first_only():
    """Test that batch_first=False is not supported."""
    with pytest.raises(ValueError, match="batch_first=True"):
        FlashMHA(
            embed_dim=256,
            num_heads=8,
            batch_first=False,
        )


@pytest.mark.skipif(not flash_attn_available, reason="flash-attn not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flash_mha_forward_no_mask():
    """Test forward pass without padding mask on CUDA."""
    embed_dim = 256
    num_heads = 8
    batch_size = 2
    seq_len = 16
    device = "cuda"

    mha = FlashMHA(
        embed_dim=embed_dim,
        num_heads=num_heads,
        bias=True,
        batch_first=True,
        attention_dropout=0.1,
    )
    mha = mha.to(device)

    # Keep source tensor float32 and rely on autocast to match real training/inference usage.
    x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=torch.float32)

    with torch.no_grad():
        out, attn_weights = _run_with_amp(lambda: mha(x))

    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    assert attn_weights is None, "Flash attention should not return weights"


@pytest.mark.skipif(not flash_attn_available, reason="flash-attn not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flash_mha_forward_with_mask():
    """Test forward pass with key padding mask on CUDA."""
    embed_dim = 256
    num_heads = 8
    batch_size = 2
    seq_len = 16
    device = "cuda"

    mha = FlashMHA(
        embed_dim=embed_dim,
        num_heads=num_heads,
        bias=True,
        batch_first=True,
        attention_dropout=0.1,
    )
    mha = mha.to(device)

    x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=torch.float32)
    # scGPT / PyTorch convention: True = padding, False = real token
    key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    key_padding_mask[:, -2:] = True
    # FlashMHA convention: True = keep, False = padding
    valid_key_padding_mask = ~key_padding_mask

    with torch.no_grad():
        out, attn_weights = _run_with_amp(
            lambda: mha(x, key_padding_mask=valid_key_padding_mask)
        )

    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    assert attn_weights is None, "Flash attention should not return weights"


@pytest.mark.skipif(not flash_attn_available, reason="flash-attn not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flash_mha_mask_effect():
    """Test that applied mask actually affects output."""
    embed_dim = 256
    num_heads = 8
    batch_size = 1
    seq_len = 8
    device = "cuda"

    mha = FlashMHA(
        embed_dim=embed_dim,
        num_heads=num_heads,
        bias=True,
        batch_first=True,
        attention_dropout=0.0,  # No dropout for determinism
    )
    mha = mha.to(device)
    mha.eval()

    x = torch.randn(
        batch_size, seq_len, embed_dim, device=device, dtype=torch.float32, requires_grad=False
    )

    # Get output without mask
    with torch.no_grad():
        out_no_mask, _ = _run_with_amp(lambda: mha(x))

    # Get output with mask
    # scGPT / PyTorch convention: True = padding, False = real token
    key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    key_padding_mask[:, -2:] = True
    # FlashMHA convention: True = keep, False = padding
    valid_key_padding_mask = ~key_padding_mask
    with torch.no_grad():
        out_with_mask, _ = _run_with_amp(
            lambda: mha(x, key_padding_mask=valid_key_padding_mask)
        )

    # Outputs should differ (masked attention should ignore padded tokens)
    # Use a generous tolerance since different backends may have slight differences
    assert not torch.allclose(
        out_no_mask.float(), out_with_mask.float(), atol=1e-2
    ), "Mask did not affect output as expected"


@pytest.mark.skipif(not flash_attn_available, reason="flash-attn not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flash_mha_causal():
    """Test causal attention mode."""
    embed_dim = 256
    num_heads = 8
    batch_size = 2
    seq_len = 16
    device = "cuda"

    mha = FlashMHA(
        embed_dim=embed_dim,
        num_heads=num_heads,
        bias=True,
        batch_first=True,
        causal=True,
    )
    mha = mha.to(device)

    x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=torch.float32)

    with torch.no_grad():
        out, attn_weights = _run_with_amp(lambda: mha(x))

    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


@pytest.mark.skipif(not flash_attn_available, reason="flash-attn not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fa2_float32_without_amp_raises():
    """FA2 flash path should fail loudly on float32 without autocast/cast."""
    if flash_attn_backend != "fa2":
        pytest.skip("FA2-specific contract test")

    mha = FlashMHA(
        embed_dim=128,
        num_heads=4,
        batch_first=True,
        attention_dropout=0.0,
    ).to("cuda")
    x = torch.randn(2, 8, 128, device="cuda", dtype=torch.float32)

    with torch.no_grad(), pytest.raises(ValueError, match="fp16/bf16"):
        _ = mha(x)


@pytest.mark.skipif(not flash_attn_available, reason="flash-attn not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fa1_float32_without_amp_behavior():
    """FA1 env contract: raw float32 input should either run or fail with a clear dtype error.

    We don't force one strict behavior because FA1 builds differ across versions,
    but we ensure behavior is explicit and non-silent.
    """
    if flash_attn_backend != "fa1":
        pytest.skip("FA1-specific contract test")

    mha = FlashMHA(
        embed_dim=128,
        num_heads=4,
        batch_first=True,
        attention_dropout=0.0,
    ).to("cuda")
    x = torch.randn(2, 8, 128, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        try:
            out, attn_weights = mha(x)
            assert out.shape == x.shape
            assert attn_weights is None
        except (AssertionError, RuntimeError, ValueError) as e:
            msg = str(e)
            # FA1 may raise a bare AssertionError (no message) or a typed error.
            # A bare assertion is also acceptable explicit failure behavior.
            if msg:
                assert any(k in msg for k in ["float16", "bfloat16", "dtype", "Half", "BFloat"])


@pytest.mark.skipif(not flash_attn_available, reason="flash-attn not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fa1_with_amp_forward_no_mask():
    """FA1 env sanity: AMP path should work and preserve output shape."""
    if flash_attn_backend != "fa1":
        pytest.skip("FA1-specific forward test")

    embed_dim = 128
    num_heads = 4
    batch_size = 2
    seq_len = 12
    device = "cuda"

    mha = FlashMHA(
        embed_dim=embed_dim,
        num_heads=num_heads,
        batch_first=True,
        attention_dropout=0.0,
    ).to(device)

    x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=torch.float32)
    with torch.no_grad():
        out, attn_weights = _run_with_amp(lambda: mha(x))

    assert out.shape == x.shape
    assert attn_weights is None


@pytest.mark.skipif(not flash_attn_available, reason="flash-attn not installed")
def test_backend_detection():
    """Test that correct backend is detected."""
    # Just verify that get_flash_attn_info returns a meaningful string
    info = get_flash_attn_info()
    assert isinstance(info, str)
    assert len(info) > 0
    # Should mention either FA1, FA2, or "not installed"
    assert any(
        x in info for x in ["flash-attn 1", "flash-attn 2", "not installed"]
    ), f"Unexpected backend info: {info}"
