"""Read and verify the pinned model artifacts listed in artifacts.lock.json."""

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_CHUNK_BYTES = 1 << 20
_TIMEOUT_S = 60


class ArtifactError(Exception):
    """A locked artifact is missing or does not match its recorded hash."""


@dataclass(frozen=True, slots=True)
class Artifact:
    name: str
    repo_id: str
    revision: str
    filename: str
    sha256: str
    size: int

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo_id}/resolve/{self.revision}/{self.filename}"

    @property
    def local_name(self) -> str:
        return f"{self.name}{Path(self.filename).suffix}"


def load_lock(path: Path) -> list[Artifact]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Artifact(**entry) for entry in raw["artifacts"]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def verify(artifact: Artifact, directory: Path) -> Path:
    path = directory / artifact.local_name
    if not path.is_file():
        raise ArtifactError(f"{artifact.name}: missing at {path}")
    actual = sha256_file(path)
    if actual != artifact.sha256:
        raise ArtifactError(f"{artifact.name}: sha256 {actual} != locked {artifact.sha256}")
    return path


def fetch(artifact: Artifact, directory: Path) -> Path:
    """Download one artifact unless a verified copy is already present."""
    target = directory / artifact.local_name
    if target.is_file() and sha256_file(target) == artifact.sha256:
        return target
    directory.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with (
        urllib.request.urlopen(artifact.url, timeout=_TIMEOUT_S) as response,  # noqa: S310
        partial.open("wb") as handle,
    ):
        shutil.copyfileobj(response, handle, _CHUNK_BYTES)
    partial.replace(target)
    return verify(artifact, directory)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and verify locked model artifacts.")
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    for artifact in load_lock(args.lock):
        path = fetch(artifact, args.out)
        sys.stdout.write(f"verified {path.name} {artifact.sha256}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
