# Training

Notebooks for finetuning each genesiss variant with Unsloth. Two parallel sets:

```
training/
├── shared/                         # helpers used by all notebooks
│   ├── data_loader.py              # downloads ricemonster/NeurIPS11092 + builds messages
│   ├── format.py                   # apply_chat_template + train_on_responses_only wiring
│   └── checkpoint.py               # resume-from-Hub + async background uploads
├── colab/                          # for Google Colab
├── kaggle/                         # for Kaggle
└── _build_notebooks.py             # regenerates the .ipynb files from the templates
```

The Modelfile used by Ollama is **auto-generated** by Unsloth during export (it
ships inside the same folder as the GGUF, with the exact chat template the
model was trained on baked in) — we don't keep a hand-written one in the repo,
because drift between training template and Modelfile template is the #1 cause
of garbage post-finetune output.

## Workflow

1. Pick a variant + platform. Open the matching notebook.
2. Run cell 1 — installs unsloth + deps, prompts for your HF token.
3. The notebook calls `pull_latest_checkpoint(repo_id, output_dir)` — if a prior
   run uploaded `checkpoint-1234/`, it gets restored here so SFTTrainer's
   `resume_from_checkpoint=True` resumes seamlessly.
4. Training runs. On every `save_steps`, a `HubCheckpointCallback` queues the
   new checkpoint folder to the Hub in a background thread (HfApi
   `run_as_future=True`) so the GPU stays busy.
5. At the end of training, push the merged 16-bit model + a GGUF Q4_K_M +
   the auto-generated Modelfile to `HUB_FINAL_REPO/gguf/`. The local CLI's
   `genesiss models pull <variant>` downloads from there and runs
   `ollama create` on the user's machine.

## Hyperparameters

Defaults follow [Unsloth's LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide.md):

| Setting            | Value (default)    | Why                                                  |
| ------------------ | ------------------ | ---------------------------------------------------- |
| LoRA r             | 16 (Qwen) / 8 (gpt-oss) | Unsloth canonical; gpt-oss recipe uses r=8.    |
| LoRA α             | 16                 | α = r or 2·r per Unsloth guide.                      |
| Target modules     | q/k/v/o/gate/up/down | Standard 7 — all linear layers.                    |
| Effective batch    | 16                 | bs × grad_accum, per Unsloth guide.                  |
| Learning rate      | 2e-4               | Standard LoRA LR.                                    |
| Warmup ratio       | 0.05               | 5% of total steps.                                   |
| Weight decay       | 0.01               | Per Unsloth guide.                                   |
| Scheduler          | linear             | Per Unsloth guide.                                   |
| Optimizer          | adamw_8bit         | Memory-efficient.                                    |

## Context length

| Variant         | `max_seq_length` | Notes                                                    |
| --------------- | ---------------- | -------------------------------------------------------- |
| `genesiss-4b`   | 8192             | Fits T4/L4 comfortably with bs=2.                        |
| `genesiss-9b`   | 8192             | Fine on L4/A100; tight on T4×2.                          |
| `genesiss-20b`  | 8192             | QLoRA + flash attn keeps activation memory in budget.    |
| `genesiss-27b`  | 4096             | 16-bit LoRA at 27B is memory-bound on H100.              |

Base-model native context limits are far larger (Qwen 3.5 → 256K, gpt-oss-20b → 128K), so this is purely a training-side budget choice — at inference we can extrapolate moderately past the trained length.

## Hardware mapping

| Variant         | Colab recommended         | Kaggle recommended |
| --------------- | ------------------------- | ------------------ |
| `genesiss-4b`   | T4 / L4                   | T4 ×2 or P100      |
| `genesiss-9b`   | L4 / A100                 | T4 ×2              |
| `genesiss-20b`  | A100 / H100               | T4 ×2 (QLoRA)      |
| `genesiss-27b`  | H100 (only)               | not feasible — use Colab H100 |

Family-specific notes
- **Qwen 3.5 (4b/9b/27b)** runs in 16-bit LoRA — Unsloth explicitly notes that QLoRA (4-bit) is not recommended for Qwen 3.5. We pass `load_in_16bit=True, load_in_4bit=False`.
- **gpt-oss-20b** runs in 4-bit (`load_in_4bit=True`, ~14 GB VRAM). GGUF export uses the llama.cpp `convert_hf_to_gguf.py` path because `save_pretrained_gguf` doesn't support gpt-oss's MoE layout yet.

## Why checkpoint-to-Hub instead of Drive or Wandb?

Unsloth's `finetuning-from-last-checkpoint` doc demonstrates the pattern with Wandb artifacts. We use HF Hub instead because:
- Same recovery path on Colab and Kaggle (no Drive vs Kaggle Datasets divergence).
- One fewer service to authenticate against — we already need HF for the final upload.
- Survives notebook timeouts on both platforms.
- `HfApi.upload_folder(run_as_future=True)` runs uploads on a background thread, so saves don't stall training.

## Set your HF token

The notebooks read `HF_TOKEN` either from a notebook secret (Colab `userdata`, Kaggle `kaggle_secrets`) or — last resort — a literal you paste into cell 2.
