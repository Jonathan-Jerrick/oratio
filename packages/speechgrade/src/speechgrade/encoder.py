"""CTC acoustic model (wav2vec2-base-960h, ONNX) and transcript alignment on top of it."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt

from speechgrade.align import WordSpan, align_words
from speechgrade.audio import SAMPLE_RATE

# Character vocabulary of facebook/wav2vec2-base-960h (Xenova ONNX export, revision a19f851).
VOCAB = {
    "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, "E": 5, "T": 6, "A": 7, "O": 8, "N": 9,
    "I": 10, "H": 11, "S": 12, "R": 13, "D": 14, "L": 15, "U": 16, "M": 17, "W": 18, "C": 19,
    "F": 20, "G": 21, "Y": 22, "P": 23, "B": 24, "V": 25, "K": 26, "'": 27, "X": 28, "J": 29,
    "Q": 30, "Z": 31,
}  # fmt: skip
BLANK = VOCAB["<pad>"]
FRAME_SECONDS = 0.02
CHUNK_SECONDS = 20.0
_EPS = 1e-7

Runner = Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]


def _log_softmax(logits: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return np.asarray(shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True)), np.float32)


class CtcEncoder:
    """Frame log-probabilities over VOCAB at 20 ms, computed in 20 s windows."""

    def __init__(self, runner: Runner) -> None:
        self._run = runner

    @classmethod
    def from_onnx(cls, path: Path, threads: int | None = None) -> "CtcEncoder":
        import onnxruntime as ort

        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        name = session.get_inputs()[0].name

        def run(window: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
            return np.asarray(session.run(None, {name: window[np.newaxis, :]})[0][0], np.float32)

        return cls(run)

    def log_probs(self, samples: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        step = round(CHUNK_SECONDS * SAMPLE_RATE)
        pieces = []
        for start in range(0, len(samples), step):
            window = samples[start : start + step].astype(np.float32)
            if len(window) < SAMPLE_RATE // 10:
                break
            normalised = (window - window.mean()) / np.sqrt(window.var() + _EPS)
            pieces.append(_log_softmax(self._run(normalised.astype(np.float32))))
        return np.concatenate(pieces) if pieces else np.zeros((0, len(VOCAB)), np.float32)

    def align(self, samples: npt.NDArray[np.float32], transcript: str) -> list[WordSpan]:
        return align_words(self.log_probs(samples), transcript, VOCAB, BLANK, FRAME_SECONDS)
