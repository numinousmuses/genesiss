"""Chat-template wiring that's shared between every variant's notebook.

Qwen 3.5 and gpt-oss-20b both ship with their own chat templates in the
tokenizer, so we just use `tokenizer.apply_chat_template`. We expose two helpers:

    text_field(ds, tokenizer)
        adds a `text` column suitable for SFTTrainer with dataset_text_field="text"

    response_only_collator(tokenizer, instruction_part, response_part)
        wraps `train_on_responses_only` so loss is masked on the prompt.
"""
from __future__ import annotations

from typing import Optional


def text_field(ds, tokenizer):
    """Render `messages` → `text` using the tokenizer's chat template."""
    def _fmt(batch):
        return {
            "text": [
                tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                for m in batch["messages"]
            ]
        }
    return ds.map(_fmt, batched=True, remove_columns=["messages"])


def maybe_train_on_responses_only(
    trainer,
    instruction_part: Optional[str] = None,
    response_part: Optional[str] = None,
):
    """Mask loss on the instruction/system tokens, leave it on the assistant response.

    If the unsloth helper isn't available, leave the trainer untouched (loss on
    everything is still fine, just slightly less sample-efficient).
    """
    try:
        from unsloth.chat_templates import train_on_responses_only
    except Exception:  # noqa: BLE001
        return trainer
    if instruction_part is None or response_part is None:
        return trainer
    return train_on_responses_only(
        trainer, instruction_part=instruction_part, response_part=response_part
    )
