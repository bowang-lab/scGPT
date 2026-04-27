"""
diagnose_flash_attn.py
======================
Minimal synthetic-data diagnostic for the scGPT FlashMHA backend.

Checks:
  1. Which flash-attn backend is active (fa1 / fa2 / none).
  2. Whether FA2's MHA has use_flash_attn=True (the most common reason for
     dense-fallback and memory blowup).
  3. The actual tensor dtype arriving at each FlashMHA forward call.
  4. Whether torch.cuda.amp.autocast is covering those calls.
  5. Peak CUDA memory at batch_size=64 vs 256.

Usage:
    python scripts/diagnose_flash_attn.py                     # uses FA backend installed
    python scripts/diagnose_flash_attn.py --no-flash          # force PyTorch path
    python scripts/diagnose_flash_attn.py --ckpt path/to.pt   # load checkpoint
    python scripts/diagnose_flash_attn.py --amp --dtype bf16  # recommended when no GradScaler

Notes:
    - Works in both FA1 and FA2 environments.
    - Must be run with the interpreter/venv where torch + flash-attn are installed.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# ── allow running from repo root or arbitrary cwd without install ─────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scgpt.model import TransformerModel
from scgpt.model.flash_attn_compat import (
    flash_attn_backend,
    flash_attn_available,
    get_flash_attn_info,
)
from scgpt.tokenizer import GeneVocab

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_vocab(n_genes: int = 1200) -> GeneVocab:
    """Build a tiny GeneVocab from fake gene names."""
    genes = [f"GENE{i}" for i in range(n_genes)]
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    vocab = GeneVocab(genes, specials=special_tokens, default_token="<pad>")
    vocab.set_default_index(vocab["<pad>"])
    return vocab


def make_batch(
    batch_size: int,
    seq_len: int,
    n_bins: int,
    vocab_size: int,
    pad_idx: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
):
    """Returns (gene_ids, values, src_key_padding_mask, batch_labels)."""
    gene_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # sprinkle some padding at the end (last 10 positions)
    gene_ids[:, -10:] = pad_idx
    # values: int for category style, float for continuous
    if dtype == torch.float32 or dtype == torch.float16 or dtype == torch.bfloat16:
        values = torch.rand(batch_size, seq_len, device=device).to(dtype) * n_bins
    else:
        values = torch.randint(0, n_bins, (batch_size, seq_len), device=device).to(dtype)
    src_key_padding_mask = gene_ids.eq(pad_idx)
    batch_labels = torch.randint(0, 2, (batch_size,), device=device)
    return gene_ids, values, src_key_padding_mask, batch_labels


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic hooks
# ─────────────────────────────────────────────────────────────────────────────

_hook_reports: list = []


def _make_mha_hook(layer_idx: int):
    def hook(mod, inp, out):
        x = inp[0]
        impl = getattr(mod, "_impl", None)
        impl_type = type(impl).__name__ if impl is not None else "N/A"

        # FA2 specific: check use_flash_attn flag
        use_flash_attn = getattr(impl, "use_flash_attn", "attr-not-found")
        # Some FA2 builds store it on inner_attn instead
        inner_attn = getattr(impl, "inner_attn", None)
        if use_flash_attn == "attr-not-found" and inner_attn is not None:
            use_flash_attn = getattr(inner_attn, "use_flash_attn", "attr-not-found")

        _hook_reports.append({
            "layer": layer_idx,
            "backend": getattr(mod, "backend", "N/A"),
            "impl_type": impl_type,
            "use_flash_attn": use_flash_attn,
            "input_dtype": str(x.dtype),
            "input_shape": tuple(x.shape),
            "autocast_enabled": torch.is_autocast_enabled(),
            "autocast_dtype": str(
                torch.get_autocast_dtype("cuda")          # torch >= 2.1
                if hasattr(torch, "get_autocast_dtype")
                else torch.get_autocast_gpu_dtype()       # torch 2.0.x
            ) if torch.is_autocast_enabled() else "N/A",
        })
    return hook


def attach_hooks(model: nn.Module) -> list:
    handles = []
    idx = 0
    for m in model.modules():
        # Match only our compat wrapper, not FA1's inner FlashMHA which shares the same
        # class name but lives in flash_attn.flash_attention.
        if (m.__class__.__name__ == "FlashMHA"
                and m.__class__.__module__ == "scgpt.model.flash_attn_compat"):
            handles.append(m.register_forward_hook(_make_mha_hook(idx)))
            idx += 1
    return handles


def remove_hooks(handles: list):
    for h in handles:
        h.remove()


def print_hook_report():
    if not _hook_reports:
        print("  [no FlashMHA hooks fired — model may not use flash path]")
        return
    # Only print first and last layer to keep output short
    samples = [_hook_reports[0]]
    if len(_hook_reports) > 1:
        samples.append(_hook_reports[-1])

    for r in samples:
        print(f"  Layer {r['layer']:>2}  backend={r['backend']}  "
              f"impl={r['impl_type']}  use_flash_attn={r['use_flash_attn']}")
        print(f"           input_dtype={r['input_dtype']}  "
              f"shape={r['input_shape']}")
        print(f"           autocast={r['autocast_enabled']}  "
              f"autocast_dtype={r['autocast_dtype']}")

    if len(_hook_reports) > 2:
        print(f"  ... ({len(_hook_reports)} FlashMHA calls total, "
              f"showing layer 0 and {_hook_reports[-1]['layer']})")


# ─────────────────────────────────────────────────────────────────────────────
# Single forward-pass probe
# ─────────────────────────────────────────────────────────────────────────────

def probe(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    n_bins: int,
    device: torch.device,
    dtype: torch.dtype,
    amp: bool,
    label: str,
):
    """Run one forward pass and report CUDA memory + hook data."""
    if not torch.cuda.is_available():
        raise RuntimeError("probe() requires CUDA; run on a CUDA-enabled machine")

    vocab_size = model.encoder.embedding.num_embeddings
    pad_idx = model.encoder.embedding.padding_idx or 0

    gene_ids, values, src_key_padding_mask, batch_labels = make_batch(
        batch_size, seq_len, n_bins, vocab_size, pad_idx, device, dtype=dtype
    )

    torch.cuda.reset_peak_memory_stats(device)
    _hook_reports.clear()

    # torch.amp.autocast (device_type=) requires torch >= 2.1;
    # fall back to torch.cuda.amp.autocast for torch 2.0.x.
    if hasattr(torch.amp, "autocast"):
        ctx_amp = torch.amp.autocast(device_type="cuda", enabled=amp,
                                     dtype=torch.bfloat16 if amp else torch.float32)
    else:
        ctx_amp = torch.cuda.amp.autocast(enabled=amp,
                                          dtype=torch.bfloat16 if amp else torch.float32)

    use_batch_lbl = model.use_batch_labels
    with torch.no_grad(), ctx_amp:
        _ = model(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=batch_labels if use_batch_lbl else None,
        )

    peak_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2

    print(f"\n{'─'*60}")
    print(f"  {label}  bs={batch_size}  seq_len={seq_len}  "
          f"amp={amp}  input_dtype={dtype}")
    print(f"  Peak CUDA memory: {peak_mb:.1f} MB")
    print_hook_report()

    # Diagnosis
    if _hook_reports:
        ufa = _hook_reports[0].get("use_flash_attn")
        idtype = _hook_reports[0].get("input_dtype")
        impl_type = _hook_reports[0].get("impl_type")
        autocast_enabled = _hook_reports[0].get("autocast_enabled")
        issues = []
        if isinstance(ufa, bool) and not ufa:
            issues.append(
                "use_flash_attn=False on FA2 MHA → dense fallback, "
                "fix: pass use_flash_attn=True when constructing MHA"
            )
        elif (ufa == "attr-not-found"
              and flash_attn_backend == "fa2"
              and impl_type != "_FA2FlashMHA"):
            # FA1 doesn't expose use_flash_attn at all — only warn for FA2 backend.
            issues.append(
                "use_flash_attn attribute not found on FA2 MHA impl — "
                "check FA2 version or module nesting"
            )
        if idtype == "torch.float32" and amp and not autocast_enabled:
            issues.append(
                "input is float32 despite AMP — autocast may not be covering "
                "the FlashMHA call; check that autocast wraps the full model forward"
            )
        if idtype == "torch.float32" and not amp:
            issues.append(
                "input is float32 and AMP is off — FA2 flash kernel "
                "requires fp16 or bf16; enable AMP or cast input explicitly"
            )
        if issues:
            print("  ⚠  ISSUES DETECTED:")
            for issue in issues:
                print(f"     • {issue}")
        else:
            print("  ✓  No obvious issues detected in this pass")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Diagnose scGPT FlashMHA backend")
    parser.add_argument("--no-flash", action="store_true",
                        help="Use use_fast_transformer=False (standard PyTorch path)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a checkpoint .pt to load")
    parser.add_argument("--amp", action="store_true",
                        help="Enable torch.amp.autocast with bfloat16")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32",
                        help="Input tensor dtype (default fp32)")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[64],
                        help="Batch sizes to probe (default: 64)")
    parser.add_argument("--seq-len", type=int, default=1201,
                        help="Sequence length (default: 1201 = 1200 genes + CLS)")
    # Model hyperparams
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--d-hid", type=int, default=512)
    parser.add_argument("--nlayers", type=int, default=12)
    parser.add_argument("--n-bins", type=int, default=51)
    # Model feature flags (match your checkpoint's training config)
    parser.add_argument("--input-emb-style", choices=["continuous", "category", "scaling"],
                        default="category",
                        help="Input embedding style (default: category; scGPT_human uses continuous)")
    parser.add_argument("--no-mvc", action="store_true", help="Disable MVC decoder")
    parser.add_argument("--no-dab", action="store_true", help="Disable DAB adversarial discriminator")
    parser.add_argument("--no-batch-labels", action="store_true", help="Disable batch label encoder")
    parser.add_argument("--no-dsbn", action="store_true", help="Disable domain-specific batchnorm")
    parser.add_argument("--no-explicit-zero-prob", action="store_true",
                        help="Disable explicit zero probability (Bernoulli head)")
    parser.add_argument("--vocab-file", type=str, default=None,
                        help="Path to vocab JSON (default: use synthetic vocab). "
                             "Provide the checkpoint's vocab.json for accurate key matching.")
    args = parser.parse_args()

    # ── dtype ────────────────────────────────────────────────────────────────
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    input_dtype = dtype_map[args.dtype]

    # ── Environment report ───────────────────────────────────────────────────
    print("=" * 60)
    print("ENVIRONMENT")
    print("=" * 60)
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        print(f"  GPU             : {torch.cuda.get_device_name(dev)}")
        print(f"  Total VRAM      : {torch.cuda.get_device_properties(dev).total_memory / 1024**3:.1f} GB")
    print(f"  flash-attn      : {get_flash_attn_info()}")
    print(f"  flash backend   : {flash_attn_backend}")

    # ── Build synthetic model ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = make_synthetic_vocab(n_genes=args.seq_len - 1)
    num_batch_labels = 2  # synthetic: 2 batches

    print("\n" + "=" * 60)
    print("MODEL CONSTRUCTION")
    print("=" * 60)
    use_fast = not args.no_flash
    print(f"  use_fast_transformer = {use_fast}")
    print(f"  fast_transformer_backend = 'flash'")
    do_dab         = not args.no_dab
    use_batch_lbl  = not args.no_batch_labels
    use_dsbn       = not args.no_dsbn
    do_mvc         = not args.no_mvc
    explicit_zero  = not args.no_explicit_zero_prob
    dsbn_val       = "dsbn" if use_dsbn else False

    print(f"  d_model={args.d_model}  nhead={args.nhead}  "
          f"d_hid={args.d_hid}  nlayers={args.nlayers}")
    print(f"  input_emb_style={args.input_emb_style}  do_mvc={do_mvc}  "
          f"do_dab={do_dab}  use_batch_labels={use_batch_lbl}  "
          f"dsbn={dsbn_val}  explicit_zero_prob={explicit_zero}")

    # ── Load real vocab if supplied, else build synthetic ────────────────────
    if args.vocab_file:
        real_vocab = GeneVocab.from_file(args.vocab_file)
        for s in ["<pad>", "<cls>", "<eoc>"]:
            if s not in real_vocab:
                real_vocab.append_token(s)
        vocab = real_vocab
        print(f"  Vocab loaded from {args.vocab_file}: {len(vocab)} tokens")

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        d_hid=args.d_hid,
        nlayers=args.nlayers,
        vocab=vocab,
        dropout=0.0,
        pad_token="<pad>",
        pad_value=-2,
        do_mvc=do_mvc,
        do_dab=do_dab,
        use_batch_labels=use_batch_lbl,
        num_batch_labels=num_batch_labels if use_batch_lbl else None,
        domain_spec_batchnorm=dsbn_val,
        n_input_bins=args.n_bins if args.input_emb_style == "category" else None,
        input_emb_style=args.input_emb_style,
        ecs_threshold=0.8,
        explicit_zero_prob=explicit_zero,
        use_fast_transformer=use_fast,
        fast_transformer_backend="flash",
        pre_norm=False,
    )

    # ── Optional checkpoint load ─────────────────────────────────────────────
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        print(f"\n  Loading checkpoint: {ckpt_path}")
        from scgpt.model.flash_attn_compat import get_flash_attn_parameter_rename_rules
        state = torch.load(ckpt_path, map_location="cpu")
        rules = get_flash_attn_parameter_rename_rules(state)
        if rules:
            import re
            print(f"  Applying {len(rules)} rename rules for FA1→FA2 migration")
            renamed = {}
            for k, v in state.items():
                new_k = k
                for pattern, replacement in rules.items():
                    new_k = re.sub(pattern, replacement, new_k)
                renamed[new_k] = v
            state = renamed
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  Missing keys  : {len(missing)} (first 5: {missing[:5]})")
        if unexpected:
            print(f"  Unexpected keys : {len(unexpected)} (first 5: {unexpected[:5]})")
        print(f"  Checkpoint loaded OK")

    model.eval()
    model.to(device)

    # ── Attach forward hooks ─────────────────────────────────────────────────
    handles = attach_hooks(model)
    num_mha = len(handles)
    print(f"\n  FlashMHA modules found and hooked: {num_mha}")
    if num_mha == 0 and use_fast:
        print("  ⚠  No FlashMHA hooks — fast transformer may have fallen back "
              "to standard PyTorch (check flash-attn install)")

    # ── FA2 internal config report ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FA2 MHA INTERNAL CONFIG (before any forward pass)")
    print("=" * 60)
    found_any = False
    for name, m in model.named_modules():
        if m.__class__.__name__ == "FlashMHA":
            impl = getattr(m, "_impl", None)
            if impl is not None:
                ufa = getattr(impl, "use_flash_attn", "attr-not-found")
                inner = getattr(impl, "inner_attn", None)
                if ufa == "attr-not-found" and inner is not None:
                    ufa = getattr(inner, "use_flash_attn", "attr-not-found")
                print(f"  {name}: impl_type={type(impl).__name__}  "
                      f"use_flash_attn={ufa}")
                found_any = True
            break  # one is enough, they are all the same config
    if not found_any and use_fast:
        print("  No FlashMHA._impl found — wrapper may be using a different attr name")

    # ── Forward pass probes ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FORWARD PASS PROBES")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("\n  CUDA is not available. Skipping forward probes.")
        print("  This diagnostic focuses on CUDA flash-attn behavior.")
        remove_hooks(handles)
        print("\n" + "=" * 60)
        print("SUMMARY & RECOMMENDATIONS")
        print("=" * 60)
        print("  Run this script on a CUDA-enabled machine/environment for full diagnostics.")
        print()
        return

    for bs in args.batch_sizes:
        try:
            probe(
                model=model,
                batch_size=bs,
                seq_len=args.seq_len,
                n_bins=args.n_bins,
                device=device,
                dtype=input_dtype,
                amp=args.amp,
                label=f"[{'flash' if use_fast else 'torch'}]",
            )
        except torch.cuda.OutOfMemoryError:
            print(f"\n  OOM at batch_size={bs} — GPU memory exhausted")
            torch.cuda.empty_cache()

    remove_hooks(handles)

    # ── Summary and recommendations ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 60)

    if not use_fast:
        print("  Running standard PyTorch TransformerEncoderLayer.")
        print("  This uses torch.nn.MultiheadAttention → SDPA backend dispatch.")
        print("  Flash kernel is used automatically if:")
        print("    • input is fp16 or bf16")
        print("    • no additive attention bias (your mask is bool, so OK)")
        print("    • dropout=0 at eval")
        print("  Enable AMP (--amp) to ensure bf16 reaching SDPA.")

    elif flash_attn_backend == "fa2":
        print("  Running FA2 high-level wrapper (FA1-shaped design).")
        print("  Wrapper is expected to report impl_type=_FA2FlashMHA.")
        print("  If input_dtype=float32 was printed above:")
        print("    → FA2 flash kernels require fp16 or bf16.")
        print("    → Enable AMP with --amp flag, or cast inputs before model call.")

    elif flash_attn_backend == "fa1":
        print("  Running FA1 backend — no changes needed for flash path.")

    else:
        print("  flash-attn not available. Model uses standard PyTorch path.")

    print()


if __name__ == "__main__":
    main()
