"""Load and combine CadQuery training datasets.

We support multiple sources, all normalized to a single `messages` column
(ChatML-style: list of `{role, content}`). They get concatenated + shuffled
before splitting, so the model doesn't see one source then the other.

Currently supported sources:

  ricemonster/NeurIPS11092  (170k, license unspecified — academic use)
      JSONL files with {input, output} pairs. Converted to a 3-turn
      conversation: system (our prompt) + user (input) + assistant (output).

  gudo7208/CAD-Coder  (250k, Apache-2.0)
      Already ChatML. We prepend our system prompt for consistency at
      inference, since this source doesn't ship with one.

Use `load_dataset_dict()` from the notebooks — the default uses both
sources combined.
"""
from __future__ import annotations

from typing import Iterable, Optional

DATASET_RICEMONSTER = "ricemonster/NeurIPS11092"
DATASET_CADCODER = "gudo7208/CAD-Coder"
DEFAULT_DATASETS: tuple[str, ...] = (DATASET_RICEMONSTER, DATASET_CADCODER)

SYSTEM_PROMPT = (
    "You are Genesiss, a CAD assistant. "
    "Given a user description, output a complete CADQuery Python script that builds the part. "
    "Output ONLY python — no markdown fences, no commentary. "
    "End the script with `result = <final_workplane_or_assembly>` so it can be exported."
)


# ---------------------------------------------------------------------------
# ricemonster/NeurIPS11092 — raw JSONL files
# ---------------------------------------------------------------------------
def _download_ricemonster_jsonl(cache_dir: Optional[str] = None) -> dict[str, str]:
    from huggingface_hub import hf_hub_download

    out: dict[str, str] = {}
    for split, fname in [
        ("train", "data_train.jsonl"),
        ("validation", "data_val.jsonl"),
    ]:
        out[split] = hf_hub_download(
            repo_id=DATASET_RICEMONSTER,
            filename=fname,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
    return out


def _ricemonster_row_to_messages(row: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["input"]},
            {"role": "assistant", "content": row["output"]},
        ]
    }


def _load_ricemonster(cache_dir: Optional[str] = None):
    from datasets import load_dataset

    files = _download_ricemonster_jsonl(cache_dir=cache_dir)
    ds = load_dataset(
        "json",
        data_files={"train": files["train"], "validation": files["validation"]},
    )
    train_cols = list(ds["train"].column_names)
    return ds.map(_ricemonster_row_to_messages, remove_columns=train_cols)


# ---------------------------------------------------------------------------
# gudo7208/CAD-Coder — already in ChatML messages format
# ---------------------------------------------------------------------------
def _cadcoder_row_to_messages(row: dict) -> dict:
    msgs = list(row["messages"])
    # Prepend our system prompt so inference behavior matches what the
    # ricemonster-derived rows trained on. Skip if the source already has one.
    if not msgs or msgs[0].get("role") != "system":
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs
    return {"messages": msgs}


def _load_cadcoder(cache_dir: Optional[str] = None):
    from datasets import load_dataset

    ds = load_dataset(DATASET_CADCODER, cache_dir=cache_dir)
    # CAD-Coder ships train/validation/test; we only need train+validation.
    keep = {k: ds[k] for k in ("train", "validation") if k in ds}
    from datasets import DatasetDict

    out = DatasetDict(keep)
    train_cols = list(out["train"].column_names)
    return out.map(_cadcoder_row_to_messages, remove_columns=train_cols)


# ---------------------------------------------------------------------------
# Combined loader
# ---------------------------------------------------------------------------
_LOADERS = {
    DATASET_RICEMONSTER: _load_ricemonster,
    DATASET_CADCODER: _load_cadcoder,
}


def load_dataset_dict(
    datasets: Iterable[str] = DEFAULT_DATASETS,
    cache_dir: Optional[str] = None,
    max_train: Optional[int] = None,
    max_eval: Optional[int] = None,
    seed: int = 3407,
):
    """Return a DatasetDict with `train` + `validation` splits.

    Each split has exactly one column, `messages`, ready for the chat
    template to render.

    `datasets` selects which sources to combine. Default is both
    ricemonster/NeurIPS11092 and gudo7208/CAD-Coder.
    `max_train` / `max_eval` truncate splits for sanity-check runs.
    """
    from datasets import DatasetDict, concatenate_datasets

    datasets = tuple(datasets)
    if not datasets:
        raise ValueError("at least one dataset name is required")

    train_parts, val_parts = [], []
    for name in datasets:
        if name not in _LOADERS:
            raise ValueError(f"unknown dataset {name!r}. Known: {sorted(_LOADERS)}")
        d = _LOADERS[name](cache_dir=cache_dir)
        train_parts.append(d["train"])
        val_parts.append(d["validation"])
        print(
            f"[data] {name}: train={len(d['train']):,}  val={len(d['validation']):,}"
        )

    train = (
        concatenate_datasets(train_parts) if len(train_parts) > 1 else train_parts[0]
    )
    val = concatenate_datasets(val_parts) if len(val_parts) > 1 else val_parts[0]

    # Shuffle the combined train set so the model doesn't see one source then
    # the next. Val is small enough that order doesn't really matter, but
    # shuffle it too for cheap insurance.
    train = train.shuffle(seed=seed)
    val = val.shuffle(seed=seed)

    if max_train is not None:
        train = train.select(range(min(max_train, len(train))))
    if max_eval is not None:
        val = val.select(range(min(max_eval, len(val))))

    print(f"[data] combined: train={len(train):,}  val={len(val):,}")
    return DatasetDict({"train": train, "validation": val})


# Legacy alias kept for any notebook still calling the old name.
def apply_chat_template(ds, tokenizer):
    """Render the `messages` column to a `text` column. Prefer
    `training.shared.format.text_field` in new code."""
    def _fmt(batch):
        return {
            "text": [
                tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                for m in batch["messages"]
            ]
        }

    return ds.map(_fmt, batched=True, remove_columns=["messages"])
