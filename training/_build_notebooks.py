"""Generate the 8 training notebooks under training/{colab,kaggle}/.

Run from anywhere (e.g. `python training/_build_notebooks.py`). Idempotent —
overwrites the .ipynb files in place. The notebooks are deliberately verbose
and self-contained so a Colab/Kaggle user can open one and just hit Run All.
"""
from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

HERE = Path(__file__).resolve().parent

Platform = Literal["colab", "kaggle"]


@dataclass
class Variant:
    slug: str               # genesiss-4b
    base: str               # unsloth/Qwen3.5-4B  or  unsloth/gpt-oss-20b
    family: Literal["qwen3.5", "gpt-oss"]
    # Defaults; per-platform overrides applied below.
    max_seq_length: int = 2048
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    lora_r: int = 16
    lora_alpha: int = 16
    learning_rate: float = 2e-4
    save_steps: int = 100


VARIANTS = [
    Variant("genesiss-4b",  "unsloth/Qwen3.5-4B",  "qwen3.5"),
    Variant("genesiss-9b",  "unsloth/Qwen3.5-9B",  "qwen3.5"),
    Variant("genesiss-20b", "unsloth/gpt-oss-20b", "gpt-oss"),
    Variant("genesiss-27b", "unsloth/Qwen3.5-27B", "qwen3.5"),
]


# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------
def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


# ---------------------------------------------------------------------------
# Cell content (platform-aware)
# ---------------------------------------------------------------------------
def cell_install(platform: Platform) -> str:
    if platform == "colab":
        return textwrap.dedent("""\
            # Install Unsloth and friends. Colab — quiet install.
            %%capture
            !pip install --upgrade --quiet pip
            !pip install --upgrade --quiet "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
            !pip install --upgrade --quiet --no-deps "trl<0.10" peft accelerate bitsandbytes
            !pip install --upgrade --quiet "huggingface_hub>=0.25" datasets tomli_w
            """)
    return textwrap.dedent("""\
        # Install Unsloth and friends — Kaggle.
        # Kaggle's CUDA preinstall is fine; we just bring in unsloth + trl + datasets.
        !pip install --quiet --upgrade pip
        !pip install --quiet --upgrade "unsloth @ git+https://github.com/unslothai/unsloth.git"
        !pip install --quiet --upgrade --no-deps "trl<0.10" peft accelerate bitsandbytes
        !pip install --quiet --upgrade "huggingface_hub>=0.25" datasets tomli_w
        """)


def cell_token(platform: Platform) -> str:
    if platform == "colab":
        return textwrap.dedent("""\
            # Pull HF_TOKEN from Colab userdata.  Settings → Secrets → add `HF_TOKEN`.
            import os
            try:
                from google.colab import userdata
                os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
            except Exception:
                # Fallback: paste your token here if you really must.
                os.environ.setdefault("HF_TOKEN", "")
            assert os.environ.get("HF_TOKEN"), "HF_TOKEN not set"

            from huggingface_hub import login
            login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
            """)
    return textwrap.dedent("""\
        # Pull HF_TOKEN from Kaggle secrets.  Add-ons → Secrets → add `HF_TOKEN`.
        import os
        try:
            from kaggle_secrets import UserSecretsClient
            os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            os.environ.setdefault("HF_TOKEN", "")
        assert os.environ.get("HF_TOKEN"), "HF_TOKEN not set"

        from huggingface_hub import login
        login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
        """)


def cell_shared(platform: Platform) -> str:
    """Pull the shared helper modules into the runtime.

    Colab/Kaggle don't have our repo cloned, so we recreate the helpers inline
    in /content (Colab) or /kaggle/working (Kaggle). Sources are vendored from
    training/shared/.
    """
    if platform == "colab":
        work = "/content"
    else:
        work = "/kaggle/working"
    return textwrap.dedent(f"""\
        # Vendor training/shared/* into the runtime so we can `from shared import ...`.
        # If you've cloned the repo into {work}, you can skip this cell and just
        # `import sys; sys.path.insert(0, "{work}/genesiss/training")`.
        import os, urllib.request, sys
        SHARED_BASE = "https://raw.githubusercontent.com/REPLACE_ME/genesiss/main/training/shared"
        TARGET = "{work}/shared"
        os.makedirs(TARGET, exist_ok=True)
        for fname in ["__init__.py", "data_loader.py", "format.py", "checkpoint.py"]:
            url = f"{{SHARED_BASE}}/{{fname}}"
            try:
                urllib.request.urlretrieve(url, f"{{TARGET}}/{{fname}}")
            except Exception as e:
                print(f"[warn] could not fetch {{fname}} from GitHub ({{e}}). "
                      "Paste the file contents manually if running before the repo is public.")
        sys.path.insert(0, "{work}")
        """)


def cell_config(v: Variant, platform: Platform) -> str:
    return textwrap.dedent(f"""\
        # ---- run config ---------------------------------------------------------
        VARIANT       = "{v.slug}"
        BASE_MODEL    = "{v.base}"
        FAMILY        = "{v.family}"                     # qwen3.5 or gpt-oss

        # Where checkpoints live on the Hub.  Each `checkpoint-XXXX` folder is a
        # full Trainer save (model.safetensors, optimizer.pt, scheduler.pt,
        # trainer_state.json, etc.) so we can resume mid-step.
        HUB_CKPT_REPO = f"YOUR_HF_USER/{{VARIANT}}-checkpoints"
        # Final merged model / GGUF goes here.
        HUB_FINAL_REPO = f"YOUR_HF_USER/{{VARIANT}}"

        MAX_SEQ_LENGTH               = {v.max_seq_length}
        PER_DEVICE_TRAIN_BATCH_SIZE  = {v.per_device_train_batch_size}
        GRADIENT_ACCUMULATION_STEPS  = {v.gradient_accumulation_steps}
        LORA_R                       = {v.lora_r}
        LORA_ALPHA                   = {v.lora_alpha}
        LEARNING_RATE                = {v.learning_rate}
        SAVE_STEPS                   = {v.save_steps}
        NUM_EPOCHS                   = 1

        # Local dir Trainer writes checkpoints to (then async-pushed to HUB_CKPT_REPO).
        OUTPUT_DIR = f"./outputs/{{VARIANT}}"
        """)


def cell_resume() -> str:
    return textwrap.dedent("""\
        # Resume from the Hub if a prior session uploaded a checkpoint.
        # `pull_latest_checkpoint` returns the local path, or None.
        from shared.checkpoint import pull_latest_checkpoint
        resumed_path = pull_latest_checkpoint(HUB_CKPT_REPO, OUTPUT_DIR, token=os.environ["HF_TOKEN"])
        print("resumed from:", resumed_path)
        """)


def cell_load_model(v: Variant) -> str:
    if v.family == "qwen3.5":
        # Unsloth says: don't QLoRA Qwen 3.5; use 16-bit LoRA.
        load_kwargs = """\
            max_seq_length = MAX_SEQ_LENGTH,
            load_in_4bit = False,
            load_in_16bit = True,
            full_finetuning = False,"""
    else:
        # gpt-oss-20b: 4-bit fits on a single 16GB card (~14GB used).
        load_kwargs = """\
            max_seq_length = MAX_SEQ_LENGTH,
            load_in_4bit = True,
            full_finetuning = False,"""
    return textwrap.dedent(f"""\
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = BASE_MODEL,
        {load_kwargs}
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r = LORA_R,
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha = LORA_ALPHA,
            lora_dropout = 0,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
            max_seq_length = MAX_SEQ_LENGTH,
        )
        """)


def cell_dataset() -> str:
    return textwrap.dedent("""\
        # Load Text-to-CadQuery (ricemonster/NeurIPS11092) and convert to a `text` column
        # ready for SFTTrainer.  Set max_train=None for full 90% split (~150k rows).
        from shared.data_loader import load_dataset_dict
        from shared.format import text_field

        ds = load_dataset_dict(max_train=None, max_eval=2000)
        ds = ds.map(lambda b: {
            "text": [
                tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                for m in b["messages"]
            ]
        }, batched=True, remove_columns=["messages"])

        print(ds)
        print("--- example ---")
        print(ds["train"][0]["text"][:1200])
        """)


def cell_trainer() -> str:
    return textwrap.dedent("""\
        from trl import SFTTrainer, SFTConfig
        from shared.checkpoint import make_hub_callback

        hub_cb = make_hub_callback(HUB_CKPT_REPO, OUTPUT_DIR, token=os.environ["HF_TOKEN"])

        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = ds["train"],
            eval_dataset = ds["validation"],
            args = SFTConfig(
                output_dir = OUTPUT_DIR,
                max_seq_length = MAX_SEQ_LENGTH,
                dataset_text_field = "text",
                per_device_train_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE,
                gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
                warmup_steps = 20,
                num_train_epochs = NUM_EPOCHS,
                learning_rate = LEARNING_RATE,
                logging_steps = 10,
                optim = "adamw_8bit",
                weight_decay = 0.0,
                lr_scheduler_type = "linear",
                seed = 3407,
                # Save every SAVE_STEPS — the HubCheckpointCallback async-pushes
                # each checkpoint to HUB_CKPT_REPO so we can resume next session.
                save_strategy = "steps",
                save_steps = SAVE_STEPS,
                save_total_limit = 2,
                evaluation_strategy = "steps",
                eval_steps = SAVE_STEPS,
                report_to = "none",
                dataset_num_proc = 2,
            ),
            callbacks = [hub_cb],
        )
        """)


def cell_train() -> str:
    return textwrap.dedent("""\
        # `resume_from_checkpoint=True` makes the Trainer look in OUTPUT_DIR for the
        # latest checkpoint-XXXX (the one we just pulled, if any).
        trainer_stats = trainer.train(resume_from_checkpoint=True if resumed_path else False)
        print(trainer_stats)
        """)


def cell_export(v: Variant) -> str:
    return textwrap.dedent(f"""\
        # ---- export -------------------------------------------------------------
        # 1) Push merged 16-bit weights (for HF inference, vLLM, transformers).
        model.push_to_hub_merged(
            HUB_FINAL_REPO,
            tokenizer,
            save_method = "merged_16bit",
            token = os.environ["HF_TOKEN"],
        )

        # 2) GGUF Q4_K_M for Ollama.  Drop this next to {v.slug}.Modelfile, then:
        #    ollama create {v.slug} -f {v.slug}.Modelfile
        model.save_pretrained_gguf(
            f"./gguf-{{VARIANT}}",
            tokenizer,
            quantization_method = "q4_k_m",
        )

        # 3) Also push the GGUF to the Hub so the CLI's `genesiss models pull` can find it.
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ["HF_TOKEN"])
        api.upload_folder(
            folder_path = f"./gguf-{{VARIANT}}",
            repo_id = HUB_FINAL_REPO,
            repo_type = "model",
            path_in_repo = "gguf",
            commit_message = "merged GGUF Q4_K_M",
        )
        """)


def cell_smoketest() -> str:
    return textwrap.dedent("""\
        # Quick smoke test on the trained adapter — does it emit CadQuery?
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
        messages = [
            {"role": "system", "content":
                "You are Genesiss, a CAD assistant. Output ONLY CADQuery python."},
            {"role": "user", "content": "a 30mm cube with a 10mm hole through the centre"},
        ]
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        out = model.generate(ids, max_new_tokens=400, do_sample=False)
        print(tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True))
        """)


# ---------------------------------------------------------------------------
# Notebook assembly
# ---------------------------------------------------------------------------
def build_notebook(v: Variant, platform: Platform) -> dict:
    title = f"{v.slug} — finetune on Text-to-CadQuery ({platform.capitalize()})"

    family_note = (
        "**Qwen 3.5 family** — Unsloth recommends **bf16 / 16-bit LoRA** (not 4-bit QLoRA) "
        "for Qwen 3.5. The model loads with `load_in_16bit=True, load_in_4bit=False`."
        if v.family == "qwen3.5"
        else "**gpt-oss-20b** is loaded in 4-bit (≈14 GB VRAM) — unsloth handles the MXFP4 MoE quantization."
    )

    platform_note = (
        "Runs on Colab. Recommended accelerator: "
        + ({"genesiss-4b": "T4 / L4", "genesiss-9b": "L4 / A100",
            "genesiss-20b": "A100 / H100", "genesiss-27b": "H100"}[v.slug])
        + "."
        if platform == "colab"
        else "Runs on Kaggle. Recommended accelerator: "
        + ({"genesiss-4b": "T4 ×2 or P100", "genesiss-9b": "T4 ×2",
            "genesiss-20b": "T4 ×2 (QLoRA)",
            "genesiss-27b": "**not feasible on Kaggle** — open the Colab H100 notebook instead"}[v.slug])
        + "."
    )

    cells = [
        md(f"# {title}\n\n"
           f"{family_note}\n\n{platform_note}\n\n"
           "Pattern\n"
           "1. Install deps + log in to HF.\n"
           "2. Pull the latest checkpoint from `HUB_CKPT_REPO` if one exists.\n"
           "3. Train with SFTTrainer — every save fires a callback that async-uploads the new checkpoint.\n"
           "4. On finish, push merged 16-bit + GGUF Q4_K_M to `HUB_FINAL_REPO`.\n"),
        md("## 1 · Install"),
        code(cell_install(platform)),
        md("## 2 · Authenticate with Hugging Face"),
        code(cell_token(platform)),
        md("## 3 · Vendor the shared helpers"),
        code(cell_shared(platform)),
        md("## 4 · Run config"),
        code(cell_config(v, platform)),
        md("## 5 · Resume from Hub (if any)"),
        code(cell_resume()),
        md("## 6 · Load base model + attach LoRA"),
        code(cell_load_model(v)),
        md("## 7 · Load dataset + apply chat template"),
        code(cell_dataset()),
        md("## 8 · Build trainer (with async Hub-upload callback)"),
        code(cell_trainer()),
        md("## 9 · Train"),
        code(cell_train()),
        md("## 10 · Smoke test"),
        code(cell_smoketest()),
        md("## 11 · Export (merged 16-bit + GGUF for Ollama)"),
        code(cell_export(v)),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    for platform in ("colab", "kaggle"):
        out_dir = HERE / platform
        out_dir.mkdir(parents=True, exist_ok=True)
        for v in VARIANTS:
            # Kaggle 27b is not feasible — emit a stub notebook that tells the user.
            if platform == "kaggle" and v.slug == "genesiss-27b":
                nb = {
                    "cells": [md(
                        "# genesiss-27b — Kaggle\n\n"
                        "Kaggle's largest free GPU (P100, T4 ×2) cannot fit a Qwen 3.5 27B LoRA "
                        "even in 16-bit. Use the Colab H100 notebook at "
                        "`training/colab/genesiss-27b.ipynb`.\n"
                    )],
                    "metadata": {
                        "kernelspec": {"name": "python3", "display_name": "Python 3"},
                        "language_info": {"name": "python"},
                    },
                    "nbformat": 4,
                    "nbformat_minor": 5,
                }
            else:
                nb = build_notebook(v, platform)  # type: ignore[arg-type]
            path = out_dir / f"{v.slug}.ipynb"
            path.write_text(json.dumps(nb, indent=1) + "\n")
            print(f"wrote {path.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
