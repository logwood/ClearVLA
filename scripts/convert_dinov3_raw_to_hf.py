#!/usr/bin/env python3
"""Convert an official DINOv3 ViT-B/16 raw checkpoint to local HF safetensors.

The ClearVLA online boundary loads AutoModel from a local directory. This
helper keeps the conversion reproducible and validates the converted model
against Meta's official implementation before writing any output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch


RAW_DEFAULT = "/data/senwang/clearvla/third_party/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
OUTPUT_DEFAULT = "/data/senwang/clearvla/third_party/dinov3/hf-vitb16-lvd1689m"
OFFICIAL_DEFAULT = "/data/senwang/clearvla/third_party/dinov3"
OFFICIAL_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_model(raw: dict[str, torch.Tensor]) -> tuple[torch.nn.Module, list[float], int]:
    from transformers import DINOv3ViTConfig, DINOv3ViTModel

    config = DINOv3ViTConfig(
        patch_size=16,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        hidden_act="gelu",
        attention_dropout=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        rope_theta=100.0,
        image_size=224,
        num_channels=3,
        query_bias=True,
        key_bias=False,
        value_bias=True,
        proj_bias=True,
        mlp_bias=True,
        layerscale_value=1e-5,
        drop_path_rate=0.0,
        use_gated_mlp=False,
        num_register_tokens=4,
        pos_embed_shift=None,
        pos_embed_jitter=None,
        pos_embed_rescale=2.0,
    )
    periods = raw["rope_embed.periods"].detach().float().tolist()
    # HF DINOv3 derives inv_freq from rope_theta. The raw checkpoint stores the
    # official period table in bfloat16, so preserve it in config and let the
    # ClearVLA loader restore it.
    config.dinov3_rope_periods = periods
    model = DINOv3ViTModel(config)
    mapped: dict[str, torch.Tensor] = {
        "embeddings.cls_token": raw["cls_token"],
        "embeddings.register_tokens": raw["storage_tokens"],
        "embeddings.mask_token": raw["mask_token"].unsqueeze(1),
        "embeddings.patch_embeddings.weight": raw["patch_embed.proj.weight"],
        "embeddings.patch_embeddings.bias": raw["patch_embed.proj.bias"],
        "norm.weight": raw["norm.weight"],
        "norm.bias": raw["norm.bias"],
    }
    for key, value in raw.items():
        if not key.startswith("blocks."):
            continue
        parts = key.split(".")
        index = int(parts[1])
        rest = ".".join(parts[2:])
        prefix = "layer.%d." % index
        if rest in ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"):
            mapped[prefix + rest] = value
        elif rest == "attn.qkv.weight":
            query, key_weight, value_weight = value.chunk(3, dim=0)
            mapped[prefix + "attention.q_proj.weight"] = query
            mapped[prefix + "attention.k_proj.weight"] = key_weight
            mapped[prefix + "attention.v_proj.weight"] = value_weight
        elif rest == "attn.qkv.bias":
            query, _, value_bias = value.chunk(3, dim=0)
            mapped[prefix + "attention.q_proj.bias"] = query
            mapped[prefix + "attention.v_proj.bias"] = value_bias
        elif rest in ("attn.proj.weight", "attn.proj.bias"):
            mapped[prefix + "attention.o_proj." + rest.rsplit(".", 1)[-1]] = value
        elif rest == "ls1.gamma":
            mapped[prefix + "layer_scale1.lambda1"] = value
        elif rest == "ls2.gamma":
            mapped[prefix + "layer_scale2.lambda1"] = value
        elif rest in ("mlp.fc1.weight", "mlp.fc1.bias"):
            mapped[prefix + "mlp.up_proj." + rest.rsplit(".", 1)[-1]] = value
        elif rest in ("mlp.fc2.weight", "mlp.fc2.bias"):
            mapped[prefix + "mlp.down_proj." + rest.rsplit(".", 1)[-1]] = value
        elif rest in ("attn.qkv.bias_mask", "rope_embed.periods"):
            continue
        else:
            raise KeyError("unhandled raw DINOv3 key: " + key)
    expected = set(model.state_dict())
    if set(mapped) != expected:
        missing = sorted(expected - set(mapped))
        extra = sorted(set(mapped) - expected)
        raise RuntimeError("mapping mismatch; missing=%s extra=%s" % (missing, extra))
    model.load_state_dict(mapped, strict=True)
    model.rope_embeddings.inv_freq.copy_(torch.tensor(periods).reciprocal())
    model.eval()
    return model, periods, len(mapped)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path(RAW_DEFAULT))
    parser.add_argument("--output", type=Path, default=Path(OUTPUT_DEFAULT))
    parser.add_argument("--official-root", type=Path, default=Path(OFFICIAL_DEFAULT))
    parser.add_argument("--official-commit", default=OFFICIAL_COMMIT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.raw.is_file():
        raise FileNotFoundError(args.raw)
    if args.output.exists() and any(args.output.iterdir()):
        if not args.force:
            raise FileExistsError("refusing non-empty output: %s" % args.output)
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.official_root))
    from dinov3.hub.backbones import dinov3_vitb16

    raw = torch.load(args.raw, map_location="cpu", weights_only=True)
    official = dinov3_vitb16(pretrained=False)
    official.load_state_dict(raw, strict=True)
    official.eval()
    model, periods, mapping_keys = build_model(raw)
    torch.manual_seed(20260929)
    pixels = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        reference = official.get_intermediate_layers(pixels, n=1)[0]
        converted = model(pixel_values=pixels).last_hidden_state[:, 5:]
    max_abs = float((reference - converted).abs().max())
    mean_abs = float((reference - converted).abs().mean())
    if max_abs > 5e-5:
        raise RuntimeError("official/HF parity failed: max=%s mean=%s" % (max_abs, mean_abs))
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="5GB")
    metadata: dict[str, Any] = {
        "schema": "clearvla_dinov3_hf_conversion_v1",
        "source_raw": str(args.raw),
        "source_sha256": sha256_file(args.raw),
        "source_bytes": args.raw.stat().st_size,
        "source_official_repo": str(args.official_root),
        "source_official_commit": args.official_commit,
        "transformers_version": __import__("transformers").__version__,
        "config": model.config.to_dict(),
        "mapping_keys": mapping_keys,
        "parity": {
            "seed": 20260929,
            "input_shape": [1, 3, 256, 256],
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "official_patch_tokens": "get_intermediate_layers n=1 normed, class+4 registers removed",
        },
        "note": "Converted from third-party mirror raw checkpoint; source endpoint provenance is recorded by caller.",
    }
    (args.output / "conversion_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "max_abs": max_abs, "mean_abs": mean_abs}, indent=2))


if __name__ == "__main__":
    main()
