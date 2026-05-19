"""Checkpoint persistence helpers — resume on launch, async-push during training.

Resume on launch:
    pull_latest_checkpoint(repo_id, output_dir)
        — looks for the highest checkpoint-XXXX subfolder on the Hub and
          downloads it into output_dir/checkpoint-XXXX so SFTTrainer's
          resume_from_checkpoint=True picks it up automatically.

Async push during training:
    HubCheckpointCallback(repo_id, output_dir)
        — TrainerCallback fired by HuggingFace Trainer on every save. It
          submits the freshly-written checkpoint folder to HfApi using
          run_as_future=True so the upload runs in a background thread and
          training is NOT blocked.

These two pieces together let a Colab/Kaggle session die at any point and
resume on the next session without losing more than the last save_steps.
"""
from __future__ import annotations

import logging
import os
import re
from concurrent.futures import Future
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resume on launch
# ---------------------------------------------------------------------------
_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def _latest_remote_checkpoint(repo_id: str, token: Optional[str]) -> Optional[str]:
    """Inspect repo file list, return the name of the latest checkpoint-XXXX dir, or None."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    except Exception as e:
        log.warning("could not list %s: %s (assuming no prior checkpoint)", repo_id, e)
        return None

    steps: list[int] = []
    for f in files:
        top = f.split("/", 1)[0]
        m = _CKPT_RE.match(top)
        if m:
            steps.append(int(m.group(1)))
    if not steps:
        return None
    return f"checkpoint-{max(steps)}"


def pull_latest_checkpoint(
    repo_id: str,
    output_dir: str | os.PathLike,
    token: Optional[str] = None,
) -> Optional[Path]:
    """If the Hub repo has a checkpoint-XXXX folder, download it into output_dir.

    Returns the local checkpoint path on success, None when nothing was pulled.
    SFTTrainer with resume_from_checkpoint=True will pick it up automatically.
    """
    from huggingface_hub import snapshot_download

    name = _latest_remote_checkpoint(repo_id, token)
    if name is None:
        log.info("no remote checkpoint at %s", repo_id)
        return None

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log.info("pulling %s/%s → %s", repo_id, name, out)
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{name}/*"],
        local_dir=str(out),
        token=token,
    )
    return out / name


# ---------------------------------------------------------------------------
# Async push during training
# ---------------------------------------------------------------------------
def make_hub_callback(repo_id: str, output_dir: str | os.PathLike, token: Optional[str] = None):
    """Return a TrainerCallback that uploads each new checkpoint in the background.

    HfApi.upload_folder(..., run_as_future=True) submits to a ThreadPoolExecutor,
    so trainer.train() continues immediately while upload happens in parallel.
    """
    from huggingface_hub import HfApi
    from transformers import TrainerCallback

    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    except Exception as e:
        log.warning("create_repo failed (continuing): %s", e)

    inflight: list[Future] = []

    class HubCheckpointCallback(TrainerCallback):
        def on_save(self, args, state, control, **_):
            step = state.global_step
            ckpt_dir = Path(output_dir) / f"checkpoint-{step}"
            if not ckpt_dir.exists():
                log.warning("on_save fired but %s missing", ckpt_dir)
                return
            log.info("queuing upload of %s → %s", ckpt_dir, repo_id)
            fut = api.upload_folder(
                folder_path=str(ckpt_dir),
                repo_id=repo_id,
                repo_type="model",
                path_in_repo=f"checkpoint-{step}",
                commit_message=f"checkpoint @ step {step}",
                run_as_future=True,
            )
            inflight.append(fut)
            # opportunistic drain: log completed uploads, but never block training
            for done in [f for f in inflight if f.done()]:
                try:
                    done.result()
                    log.info("upload ok")
                except Exception as e:  # noqa: BLE001
                    log.warning("upload failed (will retry next save): %s", e)
                inflight.remove(done)

        def on_train_end(self, args, state, control, **_):
            # Final drain — blocking — so the session doesn't exit with pending uploads.
            log.info("draining %d in-flight uploads…", len(inflight))
            for f in inflight:
                try:
                    f.result()
                except Exception as e:  # noqa: BLE001
                    log.warning("final upload failed: %s", e)

    return HubCheckpointCallback()
