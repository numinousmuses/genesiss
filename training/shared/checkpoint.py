"""Session-survival helpers for long training runs.

Three pieces, all surviving Colab/Kaggle session timeouts:

  pull_latest_checkpoint(repo_id, output_dir)
      Looks for the highest checkpoint-XXXX subfolder on the Hub and
      downloads it into output_dir/checkpoint-XXXX so SFTTrainer's
      resume_from_checkpoint=True picks it up automatically.

  make_hub_callback(repo_id, output_dir)
      TrainerCallback fired on every save. Submits the freshly-written
      checkpoint folder to HfApi using run_as_future=True so the upload
      runs in a background thread and training is NOT blocked.

  make_time_budget_callback(max_seconds)
      TrainerCallback that stops training cleanly when a wall-clock budget
      is exhausted (default 23h — leaves headroom under Colab's 24h idle
      kick). Forces a final save before signalling stop, so the last
      checkpoint reaches HF Hub. The budget resets on every train() call,
      so each resumed session gets a fresh window.

Together: a notebook that dies (idle kick, OOM, network blip) loses at
most `save_steps` of progress, and a run too big for one session
automatically self-terminates with a clean save before getting killed.
"""
from __future__ import annotations

import logging
import os
import re
import time
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
    from huggingface_hub.errors import RepositoryNotFoundError

    api = HfApi(token=token)
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    except RepositoryNotFoundError:
        # Expected on the very first run — the repo gets auto-created by the
        # HubCheckpointCallback on the first save. Don't alarm the user.
        log.info("%s doesn't exist yet — starting fresh", repo_id)
        return None
    except Exception as e:
        # Other failures (auth, network) are real and worth flagging loudly.
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


# ---------------------------------------------------------------------------
# Wall-clock time budget — stop cleanly before Colab idle-kicks the runtime
# ---------------------------------------------------------------------------
def make_time_budget_callback(max_seconds: float = 23 * 3600):
    """Return a TrainerCallback that signals a graceful stop after `max_seconds`.

    The budget resets on every `trainer.train()` call — meaning each resumed
    session starts with a fresh window. Combined with `resume_from_checkpoint`,
    a multi-session run looks like:
        session 1 (23h) → save → terminate → upload finishes
        session 2 (23h) → resume from last checkpoint → save → ...
    until num_train_epochs is reached or you stop kicking off sessions.

    When the budget expires, the callback sets `control.should_save = True`
    and `control.should_training_stop = True`, so Trainer does one final
    save and the HubCheckpointCallback uploads it before train() returns.
    """
    from transformers import TrainerCallback

    class TimeBudgetCallback(TrainerCallback):
        def __init__(self):
            self._start: Optional[float] = None
            self._fired = False

        def on_train_begin(self, args, state, control, **_):
            self._start = time.monotonic()
            self._fired = False
            log.info("time budget: %.0fs (%.1fh)", max_seconds, max_seconds / 3600)

        def on_step_end(self, args, state, control, **_):
            if self._start is None or self._fired:
                return
            elapsed = time.monotonic() - self._start
            if elapsed >= max_seconds:
                self._fired = True
                log.warning(
                    "time budget exhausted at step %d (%.1fh elapsed) — saving + stopping",
                    state.global_step, elapsed / 3600,
                )
                control.should_save = True
                control.should_training_stop = True

    return TimeBudgetCallback()
