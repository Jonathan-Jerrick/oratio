"""Raw-source acquisition with a rights register. Resumable; safe to re-run."""

import csv
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit

import httpx

from oratio_data.baselines import bitrate_rank, recording_key
from oratio_data.licenses import LicenseClass

USER_AGENT = "OratioDatasetBuilder/0.2 (research; https://github.com/Jonathan-Jerrick/oratio)"
REGISTER_FIELDS = (
    "source_id",
    "item_id",
    "file",
    "url",
    "license_short",
    "license_url",
    "license_class",
    "evidence_url",
    "bytes",
    "sha256",
    "retrieved_at",
    "notes",
)
MIN_INTERVAL_S = 1.0
MAX_ATTEMPTS = 5
MAX_BACKOFF_S = 120.0
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_CHUNK_BYTES = 1 << 20

ARCHIVE = "https://archive.org"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
COMMONS_ROOTS = ("Category:Audio files of speeches", "Category:Audio files of speeches in English")
COMMONS_MAX_DEPTH = 3
COMMONS_MAX_BYTES = 250_000_000
COMMONS_BATCH = 40
AUDIO_SUFFIXES = (".ogg", ".oga", ".opus", ".flac", ".wav", ".mp3")
PUBLIC_DOMAIN_MARKERS = ("public domain", "pd-", "pd ", "cc0", "pdm")
LIBRIVOX_ITEMS = (
    "gettysburgaddress_150anniv_1311_librivox",
    "gettysburg_address_librivox",
    "gettysburg_johng_librivox",
    "gettysburg_shurtagal_librivox",
    "gettysburg_address4_librivox",
    "lincolnsinauguraladdresses_1501_librivox",
)
LIBRIVOX_SEARCH = (
    "collection:librivoxaudio AND (title:(speech OR speeches OR oration OR orations OR address) "
    "OR subject:(speeches))"
)
ORATION_KEYWORDS = ("oration", "speech", "address", "stateoftheunion", "sotu", "inaugural")
PODCASTFILLERS_CSV = "https://zenodo.org/records/7121457/files/PodcastFillers.csv?download=1"


class AcquisitionError(RuntimeError):
    """A source could not be fetched after all retries."""


def clean_filename(url: str) -> str:
    """The file name a URL points at, without query string or percent-encoding."""
    name = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
    if not name or "/" in name or name in {".", ".."}:
        raise AcquisitionError(f"no usable file name in {url}")
    return name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class RightsRegister:
    """Append-only CSV of every acquired file with its license evidence."""

    root: Path
    path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.path = self.root / "manifests" / "rights_register.csv"

    def files(self) -> set[str]:
        if not self.path.exists():
            return set()
        with self.path.open(newline="", encoding="utf-8") as handle:
            return {row["file"] for row in csv.DictReader(handle)}

    def has(self, target: Path) -> bool:
        return target.exists() and str(target.relative_to(self.root)) in self.files()

    def add(self, target: Path, license_class: LicenseClass, **row: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        record = {
            **dict.fromkeys(REGISTER_FIELDS, ""),
            **row,
            "file": str(target.relative_to(self.root)),
            "license_class": license_class.value,
            "bytes": str(target.stat().st_size) if target.is_file() else row.get("bytes", ""),
            "sha256": sha256_file(target) if target.is_file() else row.get("sha256", ""),
            "retrieved_at": _now(),
        }
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=REGISTER_FIELDS)
            if new:
                writer.writeheader()
            writer.writerow(record)


@dataclass
class Fetcher:
    """HTTP client that spaces requests per host and backs off when told to."""

    client: httpx.Client
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    min_interval_s: float = MIN_INTERVAL_S
    _last: dict[str, float] = field(default_factory=dict)

    def _wait_turn(self, host: str) -> None:
        elapsed = self.clock() - self._last.get(host, float("-inf"))
        if elapsed < self.min_interval_s:
            self.sleep(self.min_interval_s - elapsed)
        self._last[host] = self.clock()

    @staticmethod
    def _backoff(response: httpx.Response | None, attempt: int) -> float:
        header = response.headers.get("retry-after", "") if response is not None else ""
        wait = float(header) if header.isdigit() else float(2**attempt)
        return min(wait, MAX_BACKOFF_S)

    def _request(self, url: str, params: dict[str, Any] | None, sink: Path | None) -> Any:
        host = urlsplit(url).netloc
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._wait_turn(host)
            response: httpx.Response | None = None
            try:
                with self.client.stream("GET", url, params=params) as response:
                    if response.status_code in RETRY_STATUSES:
                        raise httpx.HTTPStatusError(
                            "retry", request=response.request, response=response
                        )
                    response.raise_for_status()
                    if sink is None:
                        response.read()
                        return response.json()
                    with sink.open("wb") as handle:
                        for chunk in response.iter_bytes(_CHUNK_BYTES):
                            handle.write(chunk)
                    return None
            except httpx.HTTPStatusError as error:
                if error.response.status_code not in RETRY_STATUSES:
                    raise AcquisitionError(f"{url}: HTTP {error.response.status_code}") from error
            except httpx.TransportError:
                pass
            if attempt < MAX_ATTEMPTS:
                self.sleep(self._backoff(response, attempt))
        raise AcquisitionError(f"{url}: gave up after {MAX_ATTEMPTS} attempts")

    def json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        return self._request(url, params, None)

    def download(self, url: str, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        self._request(url, None, partial)
        partial.replace(target)
        return target


def _librivox_files(metadata: dict[str, Any], prefer_64kb: bool) -> list[str]:
    """One file per recording: the preferred bitrate where several are published."""
    names = [f["name"] for f in metadata.get("files", []) if f["name"].endswith(".mp3")]
    order = sorted(names, key=lambda n: (bitrate_rank(n) != 2) if prefer_64kb else bitrate_rank(n))
    best: dict[str, str] = {}
    for name in order:
        best.setdefault(recording_key(name), name)
    return sorted(best.values())


def _acquire_librivox_item(
    item: str, folder: str, source_id: str, prefer_64kb: bool, root: Path,
    fetcher: Fetcher, register: RightsRegister, budget: list[float],
) -> None:  # fmt: skip
    metadata = fetcher.json(f"{ARCHIVE}/metadata/{item}")
    license_url = metadata.get("metadata", {}).get("licenseurl", "")
    for name in _librivox_files(metadata, prefer_64kb):
        target = root / "raw" / folder / item / name
        if register.has(target):
            continue
        if budget[0] <= 0:
            return
        url = f"{ARCHIVE}/download/{item}/{quote(name)}"
        fetcher.download(url, target)
        budget[0] -= target.stat().st_size
        register.add(
            target,
            LicenseClass.PUBLIC_DOMAIN,
            source_id=source_id,
            item_id=item,
            url=url,
            license_short="Public Domain (LibriVox)",
            license_url=license_url,
            evidence_url=f"{ARCHIVE}/details/{item}",
        )


def acquire_librivox(root: Path, fetcher: Fetcher, register: RightsRegister) -> list[str]:
    """Download the core LibriVox items and catalogue every speech-like LibriVox item."""
    search = fetcher.json(
        f"{ARCHIVE}/advancedsearch.php",
        {"q": LIBRIVOX_SEARCH, "fl[]": "identifier", "rows": 300, "output": "json"},
    )
    catalog = sorted(doc["identifier"] for doc in search["response"]["docs"])
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "librivox_speech_catalog.json").write_text(json.dumps(catalog, indent=1))
    for item in LIBRIVOX_ITEMS:
        _acquire_librivox_item(
            item, "librivox", "S-LV", False, root, fetcher, register, [float("inf")]
        )
    return catalog


def acquire_librivox_orations(
    root: Path,
    fetcher: Fetcher,
    register: RightsRegister,
    max_items: int = 40,
    max_bytes: float = 3e9,
) -> list[str]:
    catalog = json.loads((root / "manifests" / "librivox_speech_catalog.json").read_text())
    picks = sorted(
        item
        for item in catalog
        if any(key in item.lower() for key in ORATION_KEYWORDS) and item not in LIBRIVOX_ITEMS
    )[:max_items]
    budget = [max_bytes]
    for item in picks:
        _acquire_librivox_item(
            item, "librivox_orations", "S-LV-ORATIONS", True, root, fetcher, register, budget
        )
    return picks


def _commons_files(fetcher: Fetcher, root_category: str) -> set[str]:
    seen: set[str] = set()
    files: set[str] = set()
    frontier = [(root_category, 0)]
    while frontier:
        category, depth = frontier.pop()
        if category in seen:
            continue
        seen.add(category)
        extra: dict[str, Any] = {}
        while True:
            page = fetcher.json(
                COMMONS_API,
                {"action": "query", "list": "categorymembers", "cmtitle": category,
                 "cmlimit": 500, "format": "json", **extra},
            )  # fmt: skip
            for member in page.get("query", {}).get("categorymembers", []):
                title = member["title"]
                if title.startswith("Category:") and depth < COMMONS_MAX_DEPTH:
                    frontier.append((title, depth + 1))
                elif title.startswith("File:") and title.lower().endswith(AUDIO_SUFFIXES):
                    files.add(title)
            if "continue" not in page:
                break
            extra = page["continue"]
    return files


def commons_catalog(fetcher: Fetcher) -> list[dict[str, Any]]:
    """Every audio file under the Commons speech categories, with its license metadata."""
    titles = sorted(set().union(*(_commons_files(fetcher, root) for root in COMMONS_ROOTS)))
    rows: list[dict[str, Any]] = []
    for start in range(0, len(titles), COMMONS_BATCH):
        page = fetcher.json(
            COMMONS_API,
            {"action": "query", "titles": "|".join(titles[start : start + COMMONS_BATCH]),
             "prop": "imageinfo", "iiprop": "url|size|extmetadata", "format": "json"},
        )  # fmt: skip
        for entry in page.get("query", {}).get("pages", {}).values():
            info = (entry.get("imageinfo") or [{}])[0]
            meta = info.get("extmetadata", {})
            short = meta.get("LicenseShortName", {}).get("value", "")
            rows.append(
                {
                    "title": entry.get("title", ""),
                    "url": info.get("url", ""),
                    "bytes": info.get("size") or 0,
                    "license_short": short,
                    "license_url": meta.get("LicenseUrl", {}).get("value", ""),
                    "artist": meta.get("Artist", {}).get("value", "")[:200],
                    "evidence_url": info.get("descriptionurl", ""),
                    "is_pd": any(marker in short.lower() for marker in PUBLIC_DOMAIN_MARKERS),
                }
            )
    return sorted(rows, key=lambda row: row["title"])


def acquire_commons(root: Path, fetcher: Fetcher, register: RightsRegister) -> list[str]:
    """Download public-domain Commons speech audio; returns the titles that failed."""
    catalog = commons_catalog(fetcher)
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "commons_speech_catalog.json").write_text(json.dumps(catalog, indent=1))
    failed: list[str] = []
    for row in catalog:
        if not (row["is_pd"] and row["url"] and row["bytes"] <= COMMONS_MAX_BYTES):
            continue
        target = root / "raw" / "commons" / clean_filename(row["url"])
        if register.has(target):
            continue
        try:
            fetcher.download(row["url"], target)
        except AcquisitionError:
            failed.append(row["title"])
            continue
        is_cc0 = "cc0" in row["license_short"].lower()
        register.add(
            target,
            LicenseClass.CC0 if is_cc0 else LicenseClass.PUBLIC_DOMAIN,
            source_id="S-PD-COMMONS",
            item_id=row["title"],
            url=row["url"],
            license_short=row["license_short"],
            license_url=row["license_url"],
            evidence_url=row["evidence_url"],
            notes=row["artist"][:120],
        )
    return failed


def acquire_podcastfillers_labels(root: Path, fetcher: Fetcher, register: RightsRegister) -> None:
    target = root / "raw" / "podcastfillers" / "PodcastFillers.csv"
    if register.has(target):
        return
    fetcher.download(PODCASTFILLERS_CSV, target)
    register.add(
        target,
        LicenseClass.NC_RESEARCH,
        source_id="B-PFILL-LABELS",
        item_id="zenodo:7121457",
        url=PODCASTFILLERS_CSV,
        license_short="Adobe research license (non-commercial)",
        license_url="https://zenodo.org/records/7121457",
        evidence_url="https://zenodo.org/records/7121457",
    )
