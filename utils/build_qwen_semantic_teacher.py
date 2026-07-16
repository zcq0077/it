"""Build label-free Qwen voyage-semantic embeddings for iTentformer.

The output is aligned with ``text_pool`` in the voyage-context sidecar. It
never reads route labels or future trajectory points, so the same embeddings
can be shared by all cross-validation folds without label leakage.
"""

from argparse import ArgumentParser
from pathlib import Path
import hashlib
import json
import re

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer


SEMANTIC_INSTRUCTION = (
    "Encode this observed AIS voyage context for vessel route-intent prediction. "
    "Focus on the canonical destination, vessel and cargo constraints, draught, "
    "ETA, and navigational status. Do not infer or invent a future route."
)


def text_pool_hash(text_pool):
    digest = hashlib.sha256()
    for text in text_pool:
        digest.update(str(text).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def extract_field(text, start, end):
    match = re.search(re.escape(start) + r"\s+(.*?);\s+" + re.escape(end), text)
    return match.group(1).strip() if match else "unknown"


def canonical_destination(value):
    value = str(value).upper().strip()
    if not value or value == "UNKNOWN":
        return "UNKNOWN"
    compact = re.sub(r"[^A-Z0-9]", "", value)
    return compact or "UNKNOWN"


def semantic_text(context):
    context = str(context)
    if "unavailable" in context.lower():
        return "AIS voyage context unavailable at forecast time."
    destination = extract_field(context, "destination", "ETA")
    return (
        f"{SEMANTIC_INSTRUCTION}\n"
        f"Canonical destination key: {canonical_destination(destination)}.\n"
        f"Observed context: {context}"
    )


def resolve_dtype(name, device):
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    dtype = getattr(torch, name)
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return dtype


def main():
    parser = ArgumentParser(description="Build frozen-Qwen voyage semantic embeddings.")
    parser.add_argument("--context_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument("--max_contexts", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_path)
    if output_path.exists() and not args.force:
        raise FileExistsError(f"{output_path} exists; pass --force to replace it.")
    if args.batch_size < 1 or args.max_length < 8:
        raise ValueError("batch_size must be positive and max_length must be at least 8.")

    payload = pd.read_pickle(args.context_path)
    text_pool = list(payload.get("text_pool", []))
    if not text_pool:
        raise ValueError("Voyage-context sidecar has no text_pool.")
    if args.max_contexts > 0:
        text_pool = text_pool[:args.max_contexts]

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        args.model_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    rows = []
    with torch.inference_mode():
        for start in range(0, len(text_pool), args.batch_size):
            end = min(start + args.batch_size, len(text_pool))
            texts = [semantic_text(item) for item in text_pool[start:end]]
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded, use_cache=False, return_dict=True).last_hidden_state.float()
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = F.normalize(pooled, dim=-1)
            rows.append(pooled.cpu().to(torch.float16).numpy())
            print(f"Encoded {end}/{len(text_pool)} contexts", flush=True)

    embeddings = np.concatenate(rows, axis=0)
    # Context id 0 means unavailable. A zero vector lets the main model detect
    # that no semantic evidence exists instead of learning a fabricated prior.
    embeddings[0] = 0.0
    result = {
        "format_version": 1,
        "context_path": str(args.context_path),
        "model_path": str(args.model_path),
        "text_pool_hash": text_pool_hash(text_pool),
        "text_count": len(text_pool),
        "embedding_dim": int(embeddings.shape[1]),
        "max_length": int(args.max_length),
        "pooling": "masked_mean_l2_normalized",
        "canonical_destination": "uppercase_alnum",
        "label_free": True,
        "embeddings": embeddings,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(result, output_path)
    report_path = output_path.with_suffix(".json")
    report_path.write_text(
        json.dumps({key: value for key, value in result.items() if key != "embeddings"}, indent=2),
        encoding="utf-8",
    )
    print(
        f"Saved {embeddings.shape[0]} x {embeddings.shape[1]} semantic embeddings "
        f"to {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
