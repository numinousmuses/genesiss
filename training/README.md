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
├── modelfiles/                     # Ollama Modelfile per variant (used after training)
└── _build_notebooks.py             # regenerates the .ipynb files from the templates
```

## Workflow

1. Pick a variant + platform. Open the matching notebook.
2. Run cell 1 — installs unsloth + deps, prompts for your HF token.
3. The notebook calls `pull_latest_checkpoint(repo_id, output_dir)` — if a prior
   run uploaded `checkpoint-1234/`, it gets restored here so SFTTrainer's
   `resume_from_checkpoint=True` resumes seamlessly.
4. Training runs. On every `save_steps`, a `HubCheckpointCallback` queues the
   new checkpoint folder to the Hub in a background thread (HfApi
   `run_as_future=True`) so the GPU stays busy.
5. At the end of training, push the merged 16-bit model + a GGUF Q4_K_M for
   Ollama.

## Hardware mapping

| Variant         | Colab recommended         | Kaggle recommended |
| --------------- | ------------------------- | ------------------ |
| `genesiss-4b`   | T4 / L4                   | T4 ×2 or P100      |
| `genesiss-9b`   | L4 / A100                 | T4 ×2              |
| `genesiss-20b`  | A100 / H100               | T4 ×2 (QLoRA)      |
| `genesiss-27b`  | H100 (only)               | not feasible — use Colab H100 |

Notes
- **Qwen 3.5 (4b/9b/27b)** runs in 16-bit LoRA — per Unsloth's guidance, QLoRA
  is not recommended for Qwen 3.5. Set `load_in_16bit=True, load_in_4bit=False`.
- **gpt-oss-20b** does run in 4-bit (~14GB VRAM) — Unsloth's MXFP4 handling
  preserves the MoE quantization. Set `load_in_4bit=True`.

## Why checkpoint-to-Hub instead of Drive?

- Survives notebook timeouts on both Colab and Kaggle.
- Same recovery path for both platforms — no separate Drive vs Kaggle Datasets logic.
- Uploads run in a background thread so they don't stall a step.

## Set your HF token

The notebooks read `HF_TOKEN` either from a notebook secret (Colab
`userdata`, Kaggle `kaggle_secrets`) or as a literal you paste into cell 1.
