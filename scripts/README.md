# Diagnostics Scripts

## Flash-Attn Diagnostic

Script: `scripts/diagnose_flash_attn.py`

Purpose:
- Verify whether scGPT is running FA1, FA2, or PyTorch attention path.
- Check if runtime dtype reaching `FlashMHA` is fp16/bf16 (required for FA2 flash kernels).
- Measure peak CUDA memory for selected batch sizes.
- Validate checkpoint loading key mapping across FA1/FA2 naming layouts.

### Recommended Usage

Run with the same Python interpreter/venv used for your training job.

```bash
cd /mnt/data-raid0/github_drop/scGPT
/path/to/python scripts/diagnose_flash_attn.py --help
```

### Common Commands

FA2 environment quick check:

```bash
/path/to/python scripts/diagnose_flash_attn.py --amp --batch-sizes 64 256
```

Checkpoint check (example for scGPT_human, trained with FA1):

```bash
/path/to/python scripts/diagnose_flash_attn.py \
  --ckpt /path/to/scGPT_human/best_model.pt \
  --vocab-file /path/to/scGPT_human/vocab.json \
  --input-emb-style continuous \
  --no-dab --no-batch-labels --no-dsbn --no-explicit-zero-prob \
  --seq-len 1200 \
  --amp --batch-sizes 64 256
```

FA1 environment check:

```bash
/path/to/python scripts/diagnose_flash_attn.py --amp --batch-sizes 64
```

PyTorch-only baseline (disable fast transformer):

```bash
/path/to/python scripts/diagnose_flash_attn.py --no-flash --amp --batch-sizes 64 256
```

### Interpreting Output

- `impl_type=_FA2FlashMHA` in FA2 env:
  Expected. This means the FA1-shaped FA2 wrapper is active.

- `input_dtype=torch.float32` and AMP off:
  FA2 flash kernel is not eligible. Enable `--amp` or use explicit fp16/bf16.

- Very high peak memory in FA2 path:
  Usually indicates non-flash fallback or mismatch in runtime dtype/mask path.
