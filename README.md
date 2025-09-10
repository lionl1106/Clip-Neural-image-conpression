
# clip-feature-codec

End-to-end pipeline for **CLIP feature compression + search + conditional decoding**.

## What’s inside
- `clip_feature_codec/models/` — FiLM/Res/Attention blocks and two decoders:
  - `CLIPCondDecoder` (512→…→image), `FeatureToImageDecoderLite` (memory‑lean).
- `clip_feature_codec/codecs/` — `PerChannelAffineQuantizer` (int8 per‑channel).
- `clip_feature_codec/io/` — simple zstd‑backed bitstream I/O.
- `clip_feature_codec/index/` — FAISS IP index helpers.
- `clip_feature_codec/cli/` — CLI utilities:
  - `encode_images.py` – encode a folder to CLIP, quantize, and write bitstreams.
  - `search_text.py` – build an index from decoded vectors and run text→image search.
- `docs/clip.pdf` — CLIP paper (for quick reference, if provided).
- `examples/optimized_decoder.py` — your original script (if provided).

## Install
```bash
pip install -e .
```

## Quickstart
```bash
# 1) Encode & store
python -m clip_feature_codec.cli.encode_images --img_dir /path/to/images --out_dir store

# 2) (Python) Decode vectors and save decoded.npy for CLI (see examples below)
# 3) Search
python -m clip_feature_codec.cli.search_text --store_dir store --query "a cute golden retriever" --k 5
```

## Environment tips
- Consider `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to mitigate fragmentation.
- Mixed precision (`bfloat16`) and gradient checkpointing reduce peak memory during training.
- Train progressively (e.g., 64→128→256→512px) to keep memory headroom.
