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
    # Per-variant defaults — picked to hit Unsloth's LoRA-hyperparameter guide
    # recommendation of effective batch size ≈ 16 (batch_size × grad_accum) on
    # the smallest realistic GPU for each variant.
    #
    # max_seq_length=4096 based on a 5000-row diagnostic over the combined
    # ricemonster + gudo7208/CAD-Coder train set:
    #   median 656 tokens, p99 = 2636, max = 5234, % over 4096 = 0.04%.
    # 8192 was ~2× wasted activation memory; 2048 would clip the long tail.
    max_seq_length: int = 4096
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8           # eff. batch = 16
    lora_r: int = 16                               # Unsloth canonical for Qwen 3.5
    lora_alpha: int = 16
    learning_rate: float = 2e-4
    save_steps: int = 100
    # Gradient checkpointing mode:
    #   True       → HuggingFace standard: recompute activations on backward,
    #                no CPU offload. Fastest path that still saves memory.
    #   "unsloth"  → Unsloth's offload-heavy mode: 30% less GPU memory than
    #                True but moves gradients to CPU between steps — adds
    #                2-3× per-step overhead. Only worth it when memory is
    #                actually tight.
    #   False      → No checkpointing: fastest possible step, ~2× the memory
    #                of True.
    # Default True. Override to "unsloth" for the 27B variant where 16-bit
    # LoRA at H100 budget is genuinely memory-constrained.
    gradient_checkpointing: object = True


# Per-variant tweaks. gpt-oss canonical recipe uses r=8 / α=16; the bigger
# Qwen variants need smaller per-device batches to fit.
VARIANTS = [
    Variant("genesiss-4b",  "unsloth/Qwen3.5-4B",  "qwen3.5"),
    Variant("genesiss-9b",  "unsloth/Qwen3.5-9B",  "qwen3.5",
            per_device_train_batch_size=1, gradient_accumulation_steps=16),
    Variant("genesiss-20b", "unsloth/gpt-oss-20b", "gpt-oss",
            per_device_train_batch_size=1, gradient_accumulation_steps=16,
            lora_r=8, lora_alpha=16),
    # 27B 16-bit LoRA is memory-bound — the 4096 default fits comfortably on
    # a single H100; per-device batch stays at 1 because the weights alone
    # already eat most of the 80GB budget. Use the offload-heavy checkpointing
    # mode here since memory genuinely is tight.
    Variant("genesiss-27b", "unsloth/Qwen3.5-27B", "qwen3.5",
            per_device_train_batch_size=1, gradient_accumulation_steps=16,
            gradient_checkpointing="unsloth"),
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
    # `%%capture` is a CELL magic — it must be the very first line of the
    # cell, before any comment or Python statement. We tuck the UTF-8 locale
    # workaround (per Unsloth's troubleshooting doc) inside the captured
    # block so it still runs before any later unsloth import.
    #
    # Install pattern follows Unsloth's canonical Colab recipe:
    #   1) `pip install unsloth` from PyPI to get a coherent dep tree
    #      (compatible trl, peft, accelerate, datasets, etc.).
    #   2) Force-reinstall unsloth + unsloth_zoo from git, --no-deps, to pick
    #      up the latest fixes without disturbing step-1 deps.
    #   3) Add anything outside unsloth's transitive deps that we need.
    # DO NOT pin trl/datasets/peft/accelerate yourself — unsloth_zoo has
    # narrow version ranges (e.g. trl ∈ [0.18.2, 0.24.0] != 0.19.0,
    # datasets ∈ [3.4.1, 4.4.0)) and overriding them is what produces the
    # "incompatible" pip resolver warnings.
    if platform == "colab":
        return textwrap.dedent("""\
            %%capture
            import locale
            locale.getpreferredencoding = lambda: "UTF-8"
            # Step 1: PyPI unsloth pulls a coherent dep tree.
            !pip install --upgrade --quiet pip
            !pip install --upgrade --quiet unsloth
            # Step 2: bleeding-edge unsloth + unsloth_zoo from git, deps untouched.
            !pip install --upgrade --quiet --no-deps --force-reinstall \\
                "unsloth @ git+https://github.com/unslothai/unsloth.git" \\
                "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git"
            # Step 3: extras we use directly (not in unsloth's deps).
            # - `kernels`: lets gpt-oss use FA3 (kernels-community/vllm-flash-attn3)
            #   on Hopper. No-op for Qwen 3.5 or non-Hopper but harmless to install.
            #
            # We do NOT install `flash-attn` — Unsloth's official notebooks don't
            # either. xformers is their supported attention path on Ampere/Hopper,
            # and the flash-attn build from source (no pre-built wheel for current
            # Colab torch+CUDA combo) routinely takes 25+ minutes. The big speed
            # lever for our workload is GRADIENT_CHECKPOINTING (see run-config),
            # not FA2 vs xformers.
            !pip install --upgrade --quiet "huggingface_hub>=0.25" tomli_w kernels
            """)
    # Kaggle: same recipe, no %%capture (Kaggle convention is to show output).
    return textwrap.dedent("""\
        # Install Unsloth — Kaggle.
        # Note: Kaggle T4×2 sessions expose 2 GPUs, but a vanilla notebook only
        # uses one. To actually use both, save this code as train.py and run
        # `!torchrun --nproc_per_node=2 train.py` — Unsloth auto-enables DDP
        # when launched under torchrun. The notebook-as-is uses GPU 0 only.
        import locale
        locale.getpreferredencoding = lambda: "UTF-8"
        !pip install --quiet --upgrade pip
        !pip install --quiet --upgrade unsloth
        !pip install --quiet --upgrade --no-deps --force-reinstall \\
            "unsloth @ git+https://github.com/unslothai/unsloth.git" \\
            "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git"
        # See Colab install cell for the rationale. Notably: NO flash-attn.
        !pip install --quiet --upgrade "huggingface_hub>=0.25" tomli_w kernels
        """)


def cell_token(platform: Platform) -> str:
    # Try multiple sources in order so the same cell works whether you're
    # running in the Colab browser UI, in VS Code/JupyterLab against a Colab
    # kernel, or on Kaggle. Static type checkers (Pylance/Pyright) flag
    # `google.colab` / `kaggle_secrets` as unresolved when running locally —
    # the `# type: ignore` comments silence that without affecting runtime.
    return textwrap.dedent("""\
        import os

        def _load_hf_token() -> str | None:
            # 1) Already in the environment (VS Code `.env`, shell export,
            #    runtime env var passed in by an orchestrator, etc).
            tok = os.environ.get("HF_TOKEN")
            if tok:
                return tok
            # 2) Colab secret (browser UI: 🔑 sidebar → Add secret).
            try:
                from google.colab import userdata  # type: ignore[import-not-found]
                tok = userdata.get("HF_TOKEN")
                if tok:
                    return tok
            except Exception:
                pass
            # 3) Kaggle secret (Add-ons → Secrets).
            try:
                from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]
                return UserSecretsClient().get_secret("HF_TOKEN")
            except Exception:
                pass
            return None

        token = _load_hf_token()
        if not token:
            raise RuntimeError(
                "HF_TOKEN not found. Set it via ONE of:\\n"
                "  - Colab browser: 🔑 sidebar → Add secret `HF_TOKEN` (works for VS Code/Jupyter too once set)\\n"
                "  - Kaggle: Add-ons → Secrets → add `HF_TOKEN`\\n"
                "  - Shell:  export HF_TOKEN=hf_xxx  (then restart the kernel)"
            )
        os.environ["HF_TOKEN"] = token

        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        print("[hf] logged in")
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
        SHARED_BASE = "https://raw.githubusercontent.com/numinousmuses/genesiss/main/training/shared"
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
    # Platform-specific hyperparameter overrides. The Variant defaults are
    # tuned for the smallest realistic GPU; if Colab is the target and the
    # variant is one we recommend running on H100, we bump per-device batch
    # size (keeping effective batch ≈ 16) for a ~4× wall-clock win.
    bs = v.per_device_train_batch_size
    ga = v.gradient_accumulation_steps
    note = ""
    if platform == "colab" and v.slug in ("genesiss-9b", "genesiss-20b"):
        # H100 has 80GB — plenty of room to keep more samples in-device per
        # step rather than gradient-accumulating across many micro-batches.
        bs, ga = 4, 4   # effective batch = 16, same as before
        note = "  # H100-tuned: same effective batch as Kaggle, 4× faster wall clock"

    # Default attention implementation per variant.
    # - Qwen 3.5: None lets transformers pick (FA2 on H100, xformers on
    #   Blackwell — both fine).
    # - gpt-oss-20b: GptOssForCausalLM has no SDPA path in current transformers,
    #   so on hardware without working FA2 we MUST force eager. On H100 FA2
    #   works and is much faster — flip ATTN_IMPLEMENTATION to None there.
    attn_default = "eager" if v.family == "gpt-oss" else None
    return textwrap.dedent(f"""\
        # ---- run config ---------------------------------------------------------
        VARIANT       = "{v.slug}"
        BASE_MODEL    = "{v.base}"
        FAMILY        = "{v.family}"                     # qwen3.5 or gpt-oss

        # HF user (override if forking).
        HF_USER       = "numinousmuses"

        # Where checkpoints live on the Hub.  Each `checkpoint-XXXX` folder is a
        # full Trainer save (model.safetensors, optimizer.pt, scheduler.pt,
        # trainer_state.json, etc.) so we can resume mid-step.
        HUB_CKPT_REPO  = f"{{HF_USER}}/{{VARIANT}}-checkpoints"
        # Final merged model / GGUF goes here.
        HUB_FINAL_REPO = f"{{HF_USER}}/{{VARIANT}}"

        MAX_SEQ_LENGTH               = {v.max_seq_length}
        PER_DEVICE_TRAIN_BATCH_SIZE  = {bs}{note}
        GRADIENT_ACCUMULATION_STEPS  = {ga}
        LORA_R                       = {v.lora_r}
        LORA_ALPHA                   = {v.lora_alpha}
        LEARNING_RATE                = {v.learning_rate}
        SAVE_STEPS                   = {v.save_steps}
        # See Variant dataclass for the meaning of values here.
        # True = HF standard (recommended for 4B/9B/20B on 80GB).
        # "unsloth" = aggressive offload mode (for 27B-class memory pressure).
        # False = no checkpointing (fastest, ~2× memory of True).
        GRADIENT_CHECKPOINTING       = {v.gradient_checkpointing!r}
        # Unsloth LoRA guide says 1-3 epochs with diminishing returns past 3.
        # One epoch of the combined ~332k-row dataset is already a LOT of
        # supervision — typical SFT runs use far less. Bump to 2 only if you
        # have headroom in your compute-units budget after a 1-epoch baseline.
        NUM_EPOCHS                   = 1

        # Sanity-check knob. Set to e.g. 5000 for a quick 10-minute run to
        # verify the pipeline before spending compute units on the full
        # ~380k-row combined train split. Set back to None for production.
        MAX_TRAIN                    = None
        MAX_EVAL                     = 2000

        # Hard wall-clock budget per session. Colab's idle kick is ~24h on
        # Pro+; 23h leaves headroom for the final save + HF upload. When the
        # budget expires the trainer saves cleanly and stops — re-running the
        # notebook resumes from the last checkpoint with a fresh budget.
        MAX_TRAIN_SECONDS            = 23 * 3600

        # Attention implementation — auto-detected by what's actually
        # importable AND what the current GPU supports. Two things that
        # tripped us up on Colab:
        #   (a) flash-attn isn't pre-installed even on FA2-capable GPUs
        #   (b) gpt-oss only accepts FA3/FA4, and the FA3 kernel
        #       (kernels-community/vllm-flash-attn3) requires Hopper —
        #       fails at runtime on Ampere with "S aux is currently only
        #       supported for Hopper GPUs".
        # So we probe both: package availability AND compute capability.
        def _auto_attn_for_family(family: str) -> str | None:
            try:
                import torch
                if torch.cuda.is_available():
                    major = torch.cuda.get_device_capability(0)[0]
                else:
                    major = 0
            except Exception:
                major = 0
            is_hopper_or_newer = major >= 9  # H100=9.0, B100/B200=10.0

            try:
                import flash_attn  # noqa: F401
                fa2_ok = True
            except Exception:
                fa2_ok = False
            try:
                import kernels  # noqa: F401
                kernels_ok = True
            except Exception:
                kernels_ok = False

            if family == "gpt-oss":
                # FA3 needs both the kernel package AND a Hopper-class GPU.
                # Anything else → eager (slow but always loads).
                if kernels_ok and is_hopper_or_newer:
                    return "kernels-community/vllm-flash-attn3"
                return "eager"
            # Qwen 3.5: FA2 if the package is importable, else xformers (None).
            return "flash_attention_2" if fa2_ok else None

        ATTN_IMPLEMENTATION          = _auto_attn_for_family(FAMILY)
        print(f"[attn] picked {{ATTN_IMPLEMENTATION!r}} for {{FAMILY}}")

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
        preamble = ""
        load_kwargs = """\
            max_seq_length = MAX_SEQ_LENGTH,
            load_in_4bit = False,
            load_in_16bit = True,
            full_finetuning = False,"""
    else:
        # gpt-oss-20b: 4-bit fits on a single 16GB card (~14GB used).
        # The MoE backend env var enables Unsloth's split-LoRA optimization;
        # docs claim ~35% less VRAM and up to 12x faster MoE training.
        preamble = textwrap.dedent("""\
            import os
            os.environ.setdefault("UNSLOTH_MOE_BACKEND", "grouped_mm")

            """)
        load_kwargs = """\
            max_seq_length = MAX_SEQ_LENGTH,
            load_in_4bit = True,
            full_finetuning = False,"""
    return preamble + textwrap.dedent(f"""\
        from unsloth import FastLanguageModel

        # Pass attn_implementation only when ATTN_IMPLEMENTATION is set (None
        # means: let transformers pick the best available — FA2 → SDPA → eager).
        _load_kwargs = dict(
            model_name = BASE_MODEL,
        {load_kwargs}
        )
        if ATTN_IMPLEMENTATION is not None:
            _load_kwargs["attn_implementation"] = ATTN_IMPLEMENTATION

        model, tokenizer = FastLanguageModel.from_pretrained(**_load_kwargs)

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
            use_gradient_checkpointing = GRADIENT_CHECKPOINTING,
            random_state = 3407,
            max_seq_length = MAX_SEQ_LENGTH,
        )
        """)


def cell_dataset() -> str:
    return textwrap.dedent("""\
        # Load both training sources by default:
        #   - ricemonster/NeurIPS11092 (~170k, Text2CAD-derived, license unspecified)
        #   - gudo7208/CAD-Coder       (~250k, Apache-2.0, ChatML messages format)
        # Both are normalized to the same `messages` column, concatenated, and
        # shuffled with a fixed seed (see training/shared/data_loader.py).
        # MAX_TRAIN/MAX_EVAL come from the run-config cell — MAX_TRAIN=None means
        # use the full combined train set (~380k rows after concatenation).
        from shared.data_loader import load_dataset_dict
        from shared.format import text_field

        ds = load_dataset_dict(max_train=MAX_TRAIN, max_eval=MAX_EVAL)
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
        from shared.checkpoint import make_hub_callback, make_time_budget_callback

        hub_cb     = make_hub_callback(HUB_CKPT_REPO, OUTPUT_DIR, token=os.environ["HF_TOKEN"])
        budget_cb  = make_time_budget_callback(max_seconds=MAX_TRAIN_SECONDS)

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
                # Unsloth LoRA hyperparam guide: warmup 5-10% of total steps,
                # weight_decay 0.01, linear scheduler, adamw_8bit.
                warmup_ratio = 0.05,
                num_train_epochs = NUM_EPOCHS,
                learning_rate = LEARNING_RATE,
                logging_steps = 10,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                # Save every SAVE_STEPS — the HubCheckpointCallback async-pushes
                # each checkpoint to HUB_CKPT_REPO so we can resume next session.
                save_strategy = "steps",
                save_steps = SAVE_STEPS,
                save_total_limit = 2,
                # Newer transformers renamed evaluation_strategy → eval_strategy.
                eval_strategy = "steps",
                eval_steps = SAVE_STEPS,
                report_to = "none",
                dataset_num_proc = 2,
                # Length-grouped batching: orders examples so each batch holds
                # similar-length sequences, collapsing padding waste (our median
                # is 656 tokens but max_seq_length is 4096). Unlike `packing`,
                # every NL→code example stays a complete, intact unit — no
                # splitting CADQuery scripts across batch boundaries, no
                # concatenating unrelated CAD tasks. ~2.5× less wasted compute.
                group_by_length = True,
            ),
            callbacks = [hub_cb, budget_cb],
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
    if v.family == "qwen3.5":
        # Qwen 3.5: save_pretrained_gguf is the canonical Unsloth path. It
        # writes the GGUF *and* an auto-generated Modelfile (with the exact
        # chat template the model was trained on) into the same folder.
        body = textwrap.dedent("""\
            # 1) Push merged 16-bit weights for HF inference / vLLM / transformers.
            model.push_to_hub_merged(
                HUB_FINAL_REPO, tokenizer,
                save_method = "merged_16bit",
                token = os.environ["HF_TOKEN"],
            )

            # 2) GGUF + auto-generated Modelfile (Unsloth writes both into the
            #    same directory; the Modelfile bakes in the trained chat template).
            # If save_pretrained_gguf crashes mid-export, drop
            # maximum_memory_usage to 0.5 per Unsloth's troubleshooting guide:
            #     model.save_pretrained_gguf(GGUF_DIR, tokenizer,
            #         quantization_method="q4_k_m", maximum_memory_usage=0.5)
            GGUF_DIR = f"./gguf-{VARIANT}"
            model.save_pretrained_gguf(
                GGUF_DIR, tokenizer,
                quantization_method = "q4_k_m",
            )
            # Optional: peek at the auto-generated Modelfile.
            print(open(f"{GGUF_DIR}/Modelfile").read())
            """)
    else:
        # gpt-oss-20b: per Unsloth's gpt-oss fine-tune tutorial, GGUF is built
        # by cloning llama.cpp and running convert_hf_to_gguf.py on the merged
        # weights. save_pretrained_gguf doesn't support gpt-oss's MoE layout.
        body = textwrap.dedent("""\
            # 1) Save + push merged 16-bit weights.
            MERGED_DIR = f"./merged-{VARIANT}"
            model.save_pretrained_merged(MERGED_DIR, tokenizer)
            model.push_to_hub_merged(
                HUB_FINAL_REPO, tokenizer,
                token = os.environ["HF_TOKEN"],
            )

            # 2) gpt-oss MoE requires the llama.cpp convert path (Unsloth's
            #    save_pretrained_gguf doesn't support it). Build llama.cpp,
            #    convert, then quantize to Q4_K_M.
            !git clone --depth 1 https://github.com/ggml-org/llama.cpp
            !cd llama.cpp && cmake -B build && cmake --build build --config Release -j
            GGUF_DIR = f"./gguf-{VARIANT}"
            import os; os.makedirs(GGUF_DIR, exist_ok=True)
            !python3 llama.cpp/convert_hf_to_gguf.py {MERGED_DIR} --outfile {GGUF_DIR}/model-f16.gguf --outtype f16
            !./llama.cpp/build/bin/llama-quantize {GGUF_DIR}/model-f16.gguf {GGUF_DIR}/model-q4_k_m.gguf Q4_K_M

            # 3) Write the Modelfile next to the GGUF using Unsloth's auto-generated
            #    template (matches what we trained on).
            mf = tokenizer._ollama_modelfile.replace(
                "{__FILE_LOCATION__}", f"./model-q4_k_m.gguf"
            )
            with open(f"{GGUF_DIR}/Modelfile", "w") as f:
                f.write(mf)
            print(mf)
            """)

    return body + textwrap.dedent("""\

        # 3) Push GGUF + Modelfile to the Hub so `genesiss models pull` can
        #    download them locally and `ollama create` from there.
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ["HF_TOKEN"])
        api.upload_folder(
            folder_path = GGUF_DIR,
            repo_id = HUB_FINAL_REPO,
            repo_type = "model",
            path_in_repo = "gguf",
            commit_message = "GGUF Q4_K_M + Modelfile",
        )

        print(f"\\nDone. To use locally:")
        print(f"  genesiss models pull {VARIANT}")
        print(f"  ollama run {VARIANT}")
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
        + ({"genesiss-4b":  "T4 or L4",
            "genesiss-9b":  "**H100** (recommended — ~2× faster than A100, similar units/epoch) or A100",
            "genesiss-20b": "**H100** (recommended — H100-tuned batch size; A100 also works)",
            "genesiss-27b": "**H100** (the only option that fits 27B at 16-bit LoRA)"}[v.slug])
        + "."
        if platform == "colab"
        else "Runs on Kaggle. Recommended accelerator: "
        + ({"genesiss-4b":  "T4 ×2 or P100",
            "genesiss-9b":  "T4 ×2 (tight — Colab A100/H100 is faster)",
            "genesiss-20b": "T4 ×2 (QLoRA fits on one T4, free — but H100 is faster)",
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
