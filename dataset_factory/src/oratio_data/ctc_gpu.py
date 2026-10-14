"""wav2vec2-base-960h CTC for re-alignment checks: GPU when available, else ONNX on CPU."""

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

MODEL_ID = "facebook/wav2vec2-base-960h"


@lru_cache(maxsize=1)
def _model() -> Any:
    import torch
    from transformers import Wav2Vec2ForCTC

    return Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to("cuda").eval().half(), torch


def runner(window: np.ndarray) -> np.ndarray:
    model, torch = _model()
    with torch.inference_mode():
        batch = torch.from_numpy(window[np.newaxis, :]).to("cuda").half()
        return np.asarray(model(batch).logits[0].float().cpu().numpy(), dtype=np.float32)


ONNX_REPO = "Xenova/wav2vec2-base-960h"
ONNX_REVISION = "a19f851b3d42865797e410752b4c570c871e4825"
ONNX_FILE = "onnx/model_quantized.onnx"


def has_gpu() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def best_runner() -> Callable[[np.ndarray], np.ndarray]:
    """The GPU runner when CUDA is present, otherwise the pinned int8 ONNX model on CPU."""
    if has_gpu():
        return runner
    from huggingface_hub import hf_hub_download

    from speechgrade.encoder import CtcEncoder

    path = hf_hub_download(ONNX_REPO, ONNX_FILE, revision=ONNX_REVISION)
    return CtcEncoder.from_onnx(Path(path))._run
