"""Acquisition steps that need Colab-only tooling: the Hugging Face Hub, git and datasets."""

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from oratio_data.licenses import LicenseClass
from oratio_data.sources import RightsRegister

REV_REPO = "https://github.com/revdotcom/speech-datasets.git"
LFS_POINTER = b"version https://git-lfs"
AUDIO_SUFFIXES = {".mp3", ".wav", ".flac"}
VOXPOPULI_UTTERANCES = 1500
VOXPOPULI_PER_SPEAKER = 10


@dataclass(frozen=True, slots=True)
class HubDataset:
    source_id: str
    repo: str
    license_class: LicenseClass
    folder: str
    patterns: tuple[str, ...] | None = None


HUB_DATASETS = (
    HubDataset("B-SO762", "mispeech/speechocean762", LicenseClass.APACHE, "speechocean762"),
    HubDataset("B-DISFL", "amaai-lab/DisfluencySpeech", LicenseClass.APACHE, "disfluencyspeech"),
    HubDataset(
        "B-DISFL-NYRA",
        "nyralabs/disfluency_speech_english",
        LicenseClass.APACHE,
        "disfluency_speech_english",
    ),
    HubDataset(
        "S-EARN22",
        "distil-whisper/earnings22",
        LicenseClass.CC_BY_SA,
        "earnings22_full",
        ("full/*", "README.md"),
    ),
    HubDataset(
        "S-EARN21",
        "distil-whisper/earnings21",
        LicenseClass.CC_BY_SA,
        "earnings21_full",
        ("full/*", "README.md"),
    ),
    HubDataset(
        "B-SEP28K",
        "saeedzou/sep28k-fluencybank-stutter-dataset",
        LicenseClass.NC_RESEARCH,
        "sep28k_fluencybank",
    ),
    HubDataset(
        "B-PFILL",
        "ylacombe/podcast_fillers_by_license",
        LicenseClass.MIXED_CC,
        "podcast_fillers_by_license",
    ),
)


def _tree_bytes(folder: Path) -> int:
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())


def acquire_hub(root: Path, scratch: Path, register: RightsRegister) -> None:
    """Snapshot each Hub dataset at its current revision and record that revision."""
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    for dataset in HUB_DATASETS:
        target = root / "raw" / "hf" / dataset.folder
        marker = target / ".complete.json"
        if marker.exists():
            continue
        info = api.dataset_info(dataset.repo)
        local = scratch / "hf" / dataset.folder
        snapshot_download(
            repo_id=dataset.repo,
            repo_type="dataset",
            revision=info.sha,
            local_dir=str(local),
            allow_patterns=list(dataset.patterns) if dataset.patterns else None,
        )
        shutil.copytree(local, target, dirs_exist_ok=True)
        shutil.rmtree(local, ignore_errors=True)
        size = _tree_bytes(target)
        license_short = str((info.card_data or {}).get("license") or "")
        marker.write_text(
            json.dumps({"repo": dataset.repo, "revision": info.sha, "bytes": size}, indent=1)
        )
        register.add(
            target,
            dataset.license_class,
            source_id=dataset.source_id,
            item_id=f"{dataset.repo}@{info.sha}",
            url=f"https://huggingface.co/datasets/{dataset.repo}",
            license_short=license_short,
            evidence_url=f"https://huggingface.co/datasets/{dataset.repo}",
            bytes=str(size),
            sha256="per-file in the Hub revision",
        )


def acquire_rev_transcripts(root: Path, scratch: Path, register: RightsRegister) -> None:
    """Copy Rev's speaker-labelled Earnings transcripts, skipping LFS audio pointers."""
    target = root / "raw" / "earnings" / "rev_speech_datasets"
    marker = target / ".complete.json"
    if marker.exists():
        return
    work = scratch / "rev"
    shutil.rmtree(work, ignore_errors=True)
    subprocess.run(  # noqa: S603
        ["git", "clone", "--depth", "1", REV_REPO, str(work)],  # noqa: S607
        check=True,
        env={"GIT_LFS_SKIP_SMUDGE": "1", "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    commit = subprocess.run(  # noqa: S603
        ["git", "-C", str(work), "rev-parse", "HEAD"],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    kept = 0
    for path in work.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.suffix.lower() in AUDIO_SUFFIXES:
            continue
        if path.read_bytes()[: len(LFS_POINTER)] == LFS_POINTER:
            continue
        destination = target / path.relative_to(work)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        kept += 1
    marker.write_text(json.dumps({"repo": REV_REPO, "commit": commit, "files": kept}, indent=1))
    register.add(
        target,
        LicenseClass.CC_BY_SA,
        source_id="S-EARN-REV",
        item_id=f"revdotcom/speech-datasets@{commit}",
        url=REV_REPO,
        license_short="CC BY-SA 4.0",
        evidence_url="https://github.com/revdotcom/speech-datasets/tree/main/earnings22",
        bytes=str(_tree_bytes(target)),
        sha256=f"git commit {commit}",
    )


def acquire_voxpopuli(root: Path, register: RightsRegister) -> None:
    """Stream VoxPopuli English and keep a speaker-balanced sample for norms."""
    import soundfile as sf
    from datasets import load_dataset

    target = root / "raw" / "voxpopuli_en"
    marker = target / ".complete.json"
    if marker.exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | None]] = []
    per_speaker: dict[str, int] = {}
    for example in load_dataset("facebook/voxpopuli", "en", split="train", streaming=True):
        speaker = example.get("speaker_id") or "unknown"
        if per_speaker.get(speaker, 0) >= VOXPOPULI_PER_SPEAKER:
            continue
        name = f"{example['audio_id']}.flac"
        sf.write(target / name, example["audio"]["array"], example["audio"]["sampling_rate"])
        rows.append(
            {
                "file": name,
                "speaker_id": speaker,
                "gender": example.get("gender"),
                "text": example.get("normalized_text") or example.get("raw_text"),
            }
        )
        per_speaker[speaker] = per_speaker.get(speaker, 0) + 1
        if len(rows) >= VOXPOPULI_UTTERANCES:
            break
    (target / "metadata.json").write_text(json.dumps(rows, indent=1))
    marker.write_text(json.dumps({"n": len(rows), "speakers": len(per_speaker)}, indent=1))
    register.add(
        target,
        LicenseClass.CC0,
        source_id="S-VOXPOP",
        item_id="facebook/voxpopuli:en",
        url="https://huggingface.co/datasets/facebook/voxpopuli",
        license_short="CC0 1.0",
        evidence_url="https://huggingface.co/datasets/facebook/voxpopuli",
        bytes=str(_tree_bytes(target)),
        sha256="per-file",
    )
