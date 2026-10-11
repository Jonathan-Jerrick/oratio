import io

import numpy as np
import pytest
import soundfile as sf

from speechgrade.audio import MAX_BYTES, SAMPLE_RATE, AudioError, decode_wav


def _wav(seconds: float, sample_rate: int = SAMPLE_RATE, channels: int = 1) -> bytes:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    tone = 0.25 * np.sin(2 * np.pi * 220.0 * t)
    buffer = io.BytesIO()
    sf.write(
        buffer, np.stack([tone] * channels, axis=1), sample_rate, format="WAV", subtype="PCM_16"
    )
    return buffer.getvalue()


def test_decodes_mono_float32_at_16k() -> None:
    audio = decode_wav(_wav(2.0))

    assert audio.samples.dtype == np.float32
    assert audio.samples.shape == (2 * SAMPLE_RATE,)
    assert audio.seconds == pytest.approx(2.0)
    assert float(np.max(np.abs(audio.samples))) == pytest.approx(0.25, abs=1e-3)


def test_stereo_is_mixed_to_mono() -> None:
    assert decode_wav(_wav(2.0, channels=2)).samples.ndim == 1


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"ID3\x03" + b"\x00" * 64, "not a WAV"),
        (b"RIFF", "not a WAV"),
        (b"RIFF\x00\x00\x00\x00WAVE" + b"junk" * 8, "could not be decoded"),
        (_wav(2.0, sample_rate=44_100), "sample rate"),
        (_wav(0.2), "allowed"),
        (b"RIFF" + b"\x00" * MAX_BYTES, "limit"),
    ],
)
def test_rejects_unacceptable_uploads(data: bytes, message: str) -> None:
    with pytest.raises(AudioError, match=message):
        decode_wav(data)


def test_rejects_audio_longer_than_the_limit() -> None:
    with pytest.raises(AudioError, match="allowed"):
        decode_wav(_wav(171.0))
