# Training notes

## Dataset

`ricemonster/NeurIPS11092` on the Hub. JSONL files:

| file               | role        | rows (~) |
| ------------------ | ----------- | -------- |
| `data_train.jsonl` | train (90%) | ~150k    |
| `data_val.jsonl`   | val   (5%)  | ~8.5k    |
| `data_test.jsonl`  | test  (5%)  | ~8.5k    |

Each row has two fields:
```json
{"input": "<natural language description>", "output": "<CadQuery python>"}
```

We rewrap rows as a chat conversation (system + user + assistant) and render through the tokenizer's chat template — see `training/shared/data_loader.py` and `training/shared/format.py`.

## Recipe (Qwen 3.5 variants — 4B / 9B / 27B)

Per Unsloth's Qwen 3.5 page, **don't QLoRA Qwen 3.5** — use 16-bit LoRA.

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3.5-{4B,9B,27B}",
    max_seq_length = 2048,
    load_in_4bit = False,
    load_in_16bit = True,
    full_finetuning = False,
)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, lora_alpha = 16, lora_dropout = 0,
    target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing = "unsloth",
)
```

## Recipe (gpt-oss-20b)

Loads in 4-bit. Unsloth preserves the native MXFP4 quantization on the MoE layer; we LoRA the dense attention/MLP projections same as the Qwen variants.

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gpt-oss-20b",
    max_seq_length = 2048,
    load_in_4bit = True,
    full_finetuning = False,
)
```

VRAM: ~14 GB for QLoRA per Unsloth's docs — so a single T4 ×2 or one A100 works.

## Checkpointing

We don't use Google Drive or Kaggle Datasets — checkpoints live on the **HF Hub**, in a per-variant repo `YOUR_HF_USER/genesiss-XX-checkpoints`. Two pieces:

**1. `pull_latest_checkpoint(repo_id, output_dir, token)`**
Looks at the Hub repo's file list, finds the highest `checkpoint-NNNN/` and downloads it locally before training. Combined with `trainer.train(resume_from_checkpoint=True)`, this means a notebook can die at any point and the next session picks up at the last save.

**2. `make_hub_callback(repo_id, output_dir, token)`**
A `TrainerCallback` whose `on_save` calls `HfApi.upload_folder(..., run_as_future=True)`. `run_as_future` submits the upload to a `ThreadPoolExecutor`, so training continues immediately — uploads happen in parallel with the next training steps. `on_train_end` drains pending uploads before the session ends.

This is why you'll see two repos per variant:
- `genesiss/genesiss-4b-checkpoints` — rolling Trainer checkpoints (overwritten as training advances).
- `genesiss/genesiss-4b` — final merged 16-bit + GGUF Q4_K_M.

## Export → Ollama

After training, the export cell calls:
```python
model.push_to_hub_merged(repo, tokenizer, save_method="merged_16bit", token=...)
model.save_pretrained_gguf(f"./gguf-{variant}", tokenizer, quantization_method="q4_k_m")
```

Drop the resulting `*.Q4_K_M.gguf` next to `training/modelfiles/<variant>.Modelfile`, then:
```bash
ollama create genesiss-4b -f training/modelfiles/genesiss-4b.Modelfile
```

That tag (`genesiss-4b:latest`) is what `genesiss.llm.models.REGISTRY` expects.

## Hardware matrix

| Variant         | Colab                    | Kaggle                            | Notes                                      |
| --------------- | ------------------------ | --------------------------------- | ------------------------------------------ |
| `genesiss-4b`   | T4 / L4                  | T4 ×2 or P100                     | LoRA fits comfortably.                     |
| `genesiss-9b`   | L4 / A100                | T4 ×2                             | Tight on P100.                             |
| `genesiss-20b`  | A100 / H100              | T4 ×2 (QLoRA)                     | MXFP4 MoE handled by unsloth.              |
| `genesiss-27b`  | H100                     | **not feasible — use Colab H100** | 16-bit LoRA at 27B needs an 80GB card.     |
