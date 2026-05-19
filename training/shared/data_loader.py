"""Load the Text-to-CadQuery dataset.

Source: huggingface.co/ricemonster/NeurIPS11092 — JSONL files
    data_train.jsonl  (90%)
    data_val.jsonl    (5%)
    data_test.jsonl   (5%)
Each row has two fields: `input` (natural-language prompt) and `output` (CadQuery code).

We download the JSONL files, then return a datasets.DatasetDict whose splits
already contain a `messages` column ready for `apply_chat_template`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

DEFAULT_REPO = "ricemonster/NeurIPS11092"

SYSTEM_PROMPT = (
    "You are Genesiss, a CAD assistant. "
    "Given a user description, output a complete CADQuery Python script that builds the part. "
    "Output ONLY python — no markdown fences, no commentary. "
    "End the script with `result = <final_workplane_or_assembly>` so it can be exported."
)


def download_jsonl(repo_id: str = DEFAULT_REPO, cache_dir: Optional[str] = None) -> dict[str, str]:
    """Download the 3 JSONL files via huggingface_hub.hf_hub_download. Returns local paths."""
    from huggingface_hub import hf_hub_download

    out: dict[str, str] = {}
    for split, fname in [
        ("train", "data_train.jsonl"),
        ("val", "data_val.jsonl"),
        ("test", "data_test.jsonl"),
    ]:
        out[split] = hf_hub_download(
            repo_id=repo_id, filename=fname, repo_type="dataset", cache_dir=cache_dir
        )
    return out


def _row_to_messages(row: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["input"]},
            {"role": "assistant", "content": row["output"]},
        ]
    }


def load_dataset_dict(
    repo_id: str = DEFAULT_REPO,
    cache_dir: Optional[str] = None,
    max_train: Optional[int] = None,
    max_eval: Optional[int] = None,
):
    """Return a DatasetDict with `train` + `validation` splits, each with a `messages` column.

    `max_train` / `max_eval` truncate splits for quick sanity runs.
    """
    from datasets import load_dataset

    files = download_jsonl(repo_id=repo_id, cache_dir=cache_dir)
    ds = load_dataset(
        "json",
        data_files={"train": files["train"], "validation": files["val"]},
    )
    if max_train is not None:
        ds["train"] = ds["train"].select(range(min(max_train, len(ds["train"]))))
    if max_eval is not None:
        ds["validation"] = ds["validation"].select(
            range(min(max_eval, len(ds["validation"])))
        )
    ds = ds.map(_row_to_messages, remove_columns=[c for c in ds["train"].column_names])
    return ds


def apply_chat_template(ds, tokenizer):
    """Render the `messages` column to a `text` column using the tokenizer's chat template."""
    def _fmt(batch):
        return {
            "text": [
                tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                for m in batch["messages"]
            ]
        }
    return ds.map(_fmt, batched=True, remove_columns=["messages"])
