"""Wall-clock time of each stage of a prediction.

Every prediction gets a ``timing_<id>.json`` next to its confidence files: its
preprocessing (the MSA server and parsing), the run it belonged to (downloads,
loading the model), and each stage of the model itself. On a GPU the timers
synchronize the device at every stage boundary; kernels run asynchronously, so
an unsynchronized timer would measure how long the launches took, not the work.
"""

import json
import time
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Callable, Optional

import torch
from torch import nn

# Model submodules to time, by attribute, and the stage each is reported as.
# The template, MSA and pairformer modules run once per recycling step.
MODEL_STAGES = {
    "input_embedder": "input_embedding",
    "rel_pos": "input_embedding",
    "token_bonds": "input_embedding",
    "token_bonds_type": "input_embedding",
    "contact_conditioning": "input_embedding",
    "template_module": "template_module",
    "msa_module": "msa_module",
    "pairformer_module": "pairformer",
    "distogram_module": "distogram",
    "diffusion_conditioning": "diffusion_conditioning",
    "confidence_module": "confidence",
    "affinity_module": "affinity",
    "affinity_module1": "affinity",
    "affinity_module2": "affinity",
}


def synchronize(device: torch.device) -> None:
    """Wait for the device to finish the work queued on it."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class PredictionTimer:
    """Times the stages of each prediction a model makes.

    ``attach`` wraps the model's stages once. ``start_prediction`` clears the
    stage times before each prediction and ``report`` reads them after it;
    ``run`` holds what happens once per run, filled in by the caller.
    """

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.seconds: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self.run: dict[str, float] = {}
        self.last_end: Optional[float] = None
        self.active: set[str] = set()

    def add(self, stage: str, seconds: float) -> None:
        """Add one call's time to a stage."""
        self.seconds[stage] += seconds
        self.calls[stage] += 1

    def wrap(self, stage: str, fn: Callable) -> Callable:
        """Time every call to ``fn`` as ``stage``."""

        @wraps(fn)
        def timed(*args: object, **kwargs: object) -> object:
            # A stage called from inside itself, like the confidence module
            # running its samples one at a time, is timed once, from outside.
            if stage in self.active:
                return fn(*args, **kwargs)
            self.active.add(stage)
            synchronize(self.device)
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                synchronize(self.device)
                self.add(stage, time.perf_counter() - start)
                self.active.discard(stage)

        return timed

    def attach(self, model: nn.Module) -> None:
        """Wrap each stage of a Boltz model, once."""
        if getattr(model, "_prediction_timer", None) is self:
            return
        model._prediction_timer = self  # noqa: SLF001

        for attr, stage in MODEL_STAGES.items():
            module = getattr(model, attr, None)
            # At inference a compiled module is bypassed for the original.
            module = getattr(module, "_orig_mod", module)
            if isinstance(module, nn.Module):
                module.forward = self.wrap(stage, module.forward)
        model.structure_module.sample = self.wrap(
            "diffusion_sampling", model.structure_module.sample
        )
        model.forward = self.wrap("forward_total", model.forward)
        model.predict_step = self.wrap("predict_step_total", model.predict_step)

    def start_run(self) -> None:
        """Mark the start of a run, where loading the first batch begins."""
        self.last_end = time.perf_counter()

    def start_prediction(self, device: torch.device) -> None:
        """Clear the stage times, counting the wait since the last prediction.

        That wait is the data loading: reading the processed input and building
        its features, and for the first prediction starting the dataloader.
        """
        now = time.perf_counter()
        self.seconds.clear()
        self.calls.clear()
        self.device = device
        if self.last_end is not None:
            self.add("data_loading", now - self.last_end)

    def end_prediction(self) -> None:
        """Mark the end of a prediction, where loading the next one begins."""
        self.last_end = time.perf_counter()

    def report(self, record_id: str, preprocessing: Optional[dict]) -> dict:
        """Return the times of the prediction just made, for its timing file."""
        return {
            "id": record_id,
            "device": str(self.device),
            "preprocessing": preprocessing,
            "run": {name: round(seconds, 4) for name, seconds in self.run.items()},
            "prediction": {
                stage: {"seconds": round(seconds, 4), "calls": self.calls[stage]}
                for stage, seconds in self.seconds.items()
            },
        }


def write_timing(path: Path, timing: dict) -> None:
    """Write a timing file."""
    with path.open("w") as f:
        json.dump(timing, f, indent=4)


def write_preprocessing_timing(
    timing_dir: Path, record_id: str, seconds: dict[str, float]
) -> None:
    """Save a record's preprocessing times, for the writer to report.

    Preprocessing may run in worker processes, so the times travel on disk.
    """
    timing_dir.mkdir(parents=True, exist_ok=True)
    rounded = {name: round(value, 4) for name, value in seconds.items()}
    write_timing(timing_dir / f"{record_id}.json", rounded)


def read_preprocessing_timing(
    timing_dir: Optional[Path], record_id: str
) -> Optional[dict]:
    """Read a record's preprocessing times, or None if none were saved."""
    if timing_dir is None:
        return None
    path = timing_dir / f"{record_id}.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)
