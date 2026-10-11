import hashlib
import io
import json
import urllib.request
from dataclasses import asdict
from pathlib import Path

import pytest

from speechgrade.artifacts import Artifact, ArtifactError, fetch, load_lock, main, verify

REPO_ROOT = Path(__file__).resolve().parents[3]


def _artifact(payload: bytes) -> Artifact:
    return Artifact(
        name="encoder",
        repo_id="org/repo",
        revision="abc123",
        filename="onnx/model.onnx",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )


def test_verify_accepts_matching_file(tmp_path: Path) -> None:
    artifact = _artifact(b"weights")
    (tmp_path / "encoder.onnx").write_bytes(b"weights")

    assert verify(artifact, tmp_path) == tmp_path / "encoder.onnx"


def test_verify_rejects_tampered_file(tmp_path: Path) -> None:
    artifact = _artifact(b"weights")
    (tmp_path / "encoder.onnx").write_bytes(b"tampered")

    with pytest.raises(ArtifactError, match="sha256"):
        verify(artifact, tmp_path)


def test_verify_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="missing"):
        verify(_artifact(b"weights"), tmp_path)


def test_url_pins_revision() -> None:
    assert _artifact(b"x").url == "https://huggingface.co/org/repo/resolve/abc123/onnx/model.onnx"


def test_load_lock_round_trips(tmp_path: Path) -> None:
    artifact = _artifact(b"weights")
    lock = tmp_path / "artifacts.lock.json"
    lock.write_text(json.dumps({"artifacts": [asdict(artifact)]}))

    assert load_lock(lock) == [artifact]


def test_committed_lock_is_fully_pinned() -> None:
    artifacts = load_lock(REPO_ROOT / "artifacts.lock.json")

    assert artifacts
    assert len({a.name for a in artifacts}) == len(artifacts)
    for artifact in artifacts:
        assert len(artifact.revision) == 40
        assert len(artifact.sha256) == 64
        assert artifact.size > 0


def _serve(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> list[str]:
    requested: list[str] = []

    def fake_urlopen(url: str, timeout: float) -> io.BytesIO:
        requested.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return requested


def test_fetch_downloads_then_reuses_verified_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _artifact(b"weights")
    requested = _serve(monkeypatch, b"weights")

    first = fetch(artifact, tmp_path / "models")
    second = fetch(artifact, tmp_path / "models")

    assert first == second
    assert first.read_bytes() == b"weights"
    assert requested == [artifact.url]


def test_fetch_rejects_download_with_wrong_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve(monkeypatch, b"not the weights")

    with pytest.raises(ArtifactError, match="sha256"):
        fetch(_artifact(b"weights"), tmp_path)


def test_cli_fetches_every_locked_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact = _artifact(b"weights")
    _serve(monkeypatch, b"weights")
    lock = tmp_path / "artifacts.lock.json"
    lock.write_text(json.dumps({"artifacts": [asdict(artifact)]}))

    assert main(["--lock", str(lock), "--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "encoder.onnx").is_file()
    assert artifact.sha256 in capsys.readouterr().out
