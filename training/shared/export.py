"""Turn any training checkpoint into a deployable model.

Every `checkpoint-NNNN/` saved during training is a LoRA adapter snapshot
(adapter_model.safetensors + adapter_config.json + optimizer/scheduler state).
This module loads the base model, applies the adapter, merges them, and
produces the deployable artifacts — merged 16-bit weights, a GGUF, and an
Ollama Modelfile — for ANY checkpoint, not just the final one.

Use it to:
  - deploy a mid-training checkpoint that already passes your quality bar
  - pick the best checkpoint by eval loss instead of blindly taking the last
  - recover a deployable model after a session died, from the Hub checkpoint

For Qwen 3.5 the GGUF is produced by Unsloth's save_pretrained_gguf (which
also writes the Modelfile). For gpt-oss the MoE layout needs the llama.cpp
convert path — this module merges + pushes 16-bit and prints the llama.cpp
commands to run rather than shelling out to a multi-minute build itself.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

Family = Literal["qwen3.5", "gpt-oss"]


def resolve_checkpoint(checkpoint: str, token: Optional[str] = None) -> str:
    """Return a local directory path for `checkpoint`.

    Accepts:
      - a local directory path  → returned unchanged
      - "repo_id:checkpoint-NNNN" → that subfolder is downloaded from the Hub
      - "repo_id" → the whole repo is downloaded (use when it IS the adapter)
    """
    if os.path.isdir(checkpoint):
        return checkpoint

    from huggingface_hub import snapshot_download

    if ":" in checkpoint:
        repo_id, ckpt = checkpoint.split(":", 1)
        local = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=[f"{ckpt}/*"],
            token=token,
        )
        return str(Path(local) / ckpt)

    return snapshot_download(repo_id=checkpoint, repo_type="model", token=token)


def export_checkpoint(
    checkpoint: str,
    family: Family,
    final_repo: str,
    token: str,
    max_seq_length: int = 4096,
    quantization: str = "q4_k_m",
    push: bool = True,
) -> str:
    """Merge a checkpoint's adapter into its base model and export for Ollama.

    `checkpoint` — local path, "repo:checkpoint-NNNN", or a plain adapter repo.
    `family`     — "qwen3.5" or "gpt-oss" (decides the GGUF path).
    `final_repo` — HF repo to push the merged model + gguf/ folder to.
    Returns the local GGUF directory path.
    """
    from unsloth import FastLanguageModel

    ckpt_dir = resolve_checkpoint(checkpoint, token=token)
    print(f"[export] loading checkpoint from {ckpt_dir}")

    # Unsloth reads the base model id from the adapter_config.json inside the
    # checkpoint, loads the base, and applies the adapter — all from this call.
    load_kwargs: dict[str, object] = {
        "model_name": ckpt_dir,
        "max_seq_length": max_seq_length,
    }
    if family == "qwen3.5":
        load_kwargs["load_in_4bit"] = False
        load_kwargs["load_in_16bit"] = True
    else:  # gpt-oss
        load_kwargs["load_in_4bit"] = True

    model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)

    gguf_dir = f"./gguf-export-{Path(ckpt_dir).name}"
    os.makedirs(gguf_dir, exist_ok=True)

    if family == "qwen3.5":
        # Push merged 16-bit + write GGUF and Modelfile in one shot.
        if push:
            model.push_to_hub_merged(
                final_repo, tokenizer, save_method="merged_16bit", token=token
            )
        model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method=quantization)
        print(f"[export] GGUF + Modelfile written to {gguf_dir}")
    else:
        # gpt-oss: merge to 16-bit, then llama.cpp convert (not done here).
        merged = f"./merged-{Path(ckpt_dir).name}"
        model.save_pretrained_merged(merged, tokenizer)
        if push:
            model.push_to_hub_merged(final_repo, tokenizer, token=token)
        print(
            f"[export] merged 16-bit written to {merged}\n"
            "[export] gpt-oss GGUF needs the llama.cpp path — run:\n"
            "  git clone --depth 1 https://github.com/ggml-org/llama.cpp\n"
            "  cd llama.cpp && cmake -B build && cmake --build build --config Release -j\n"
            f"  python3 llama.cpp/convert_hf_to_gguf.py {merged} "
            f"--outfile {gguf_dir}/model-f16.gguf --outtype f16\n"
            f"  ./llama.cpp/build/bin/llama-quantize {gguf_dir}/model-f16.gguf "
            f"{gguf_dir}/model-{quantization}.gguf {quantization.upper()}"
        )

    if push:
        from huggingface_hub import HfApi

        HfApi(token=token).upload_folder(
            folder_path=gguf_dir,
            repo_id=final_repo,
            repo_type="model",
            path_in_repo="gguf",
            commit_message=f"GGUF {quantization} from {Path(ckpt_dir).name}",
        )
        print(f"[export] uploaded {gguf_dir} → {final_repo}/gguf/")

    return gguf_dir
