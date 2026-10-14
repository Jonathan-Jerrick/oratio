"""The same wav2vec2-base-960h CTC model on a Colab GPU, for fast re-alignment checks."""

from functools import lru_cache
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
