import csv
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from oratio_data.licenses import LicenseClass
from oratio_data.sources import (
    AcquisitionError,
    Fetcher,
    RightsRegister,
    acquire_commons,
    acquire_librivox,
    acquire_librivox_orations,
    acquire_podcastfillers_labels,
    clean_filename,
)

Handler = Callable[[httpx.Request], httpx.Response]


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def __call__(self) -> float:
        return self.now


def _fetcher(handler: Handler, clock: Clock | None = None) -> Fetcher:
    clock = clock or Clock()
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return Fetcher(client=client, sleep=clock.sleep, clock=clock)


def _rows(root: Path) -> list[dict[str, str]]:
    with (root / "manifests" / "rights_register.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize(
    ("url", "name"),
    [
        ("https://w.example/a/Speech.ogg?utm_source=commons&utm_content=original", "Speech.ogg"),
        ("https://w.example/a/Gandhi%27s_Statement%2C_1931.ogg", "Gandhi's_Statement,_1931.ogg"),
        ("https://archive.org/download/item/file_64kb.mp3", "file_64kb.mp3"),
    ],
)  # fmt: skip
def test_clean_filename_drops_query_and_decodes(url: str, name: str) -> None:
    assert clean_filename(url) == name


@pytest.mark.parametrize(
    "url", ["https://example.org/", "https://example.org/a/%2E%2E", "https://e.org/x%2Fy"]
)
def test_clean_filename_refuses_unsafe_names(url: str) -> None:
    with pytest.raises(AcquisitionError, match="no usable file name"):
        clean_filename(url)


def test_requests_to_one_host_are_spaced_out(tmp_path: Path) -> None:
    clock = Clock()
    fetcher = _fetcher(lambda request: httpx.Response(200, content=b"x"), clock)

    for index in range(3):
        fetcher.download(f"https://host.example/{index}.ogg", tmp_path / f"{index}.ogg")

    assert clock.sleeps == [1.0, 1.0]


def test_rate_limited_download_waits_for_retry_after_then_succeeds(tmp_path: Path) -> None:
    clock = Clock()
    replies = iter(
        [httpx.Response(429, headers={"retry-after": "7"}), httpx.Response(200, content=b"audio")]
    )
    fetcher = _fetcher(lambda request: next(replies), clock)

    target = fetcher.download("https://host.example/a.ogg", tmp_path / "a.ogg")

    assert target.read_bytes() == b"audio"
    assert 7.0 in clock.sleeps
    assert not (tmp_path / "a.ogg.part").exists()


def test_download_gives_up_after_repeated_failures_and_leaves_no_file(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(503)

    with pytest.raises(AcquisitionError, match="gave up after 5 attempts"):
        _fetcher(handler).download("https://host.example/a.ogg", tmp_path / "a.ogg")

    assert len(calls) == 5
    assert not (tmp_path / "a.ogg").exists()


def test_client_errors_are_not_retried(tmp_path: Path) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(404)

    with pytest.raises(AcquisitionError, match="HTTP 404"):
        _fetcher(handler).json("https://host.example/missing")

    assert len(calls) == 1


def test_network_errors_are_retried(tmp_path: Path) -> None:
    attempts = iter([True, False])

    def handler(request: httpx.Request) -> httpx.Response:
        if next(attempts):
            raise httpx.ConnectError("reset")
        return httpx.Response(200, json={"ok": True})

    assert _fetcher(handler).json("https://host.example/x") == {"ok": True}


def test_register_records_hash_size_and_license(tmp_path: Path) -> None:
    register = RightsRegister(tmp_path)
    target = tmp_path / "raw" / "commons" / "a.ogg"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"audio")

    assert not register.has(target)
    register.add(target, LicenseClass.PUBLIC_DOMAIN, source_id="S", item_id="File:a.ogg")

    (row,) = _rows(tmp_path)
    assert register.has(target)
    assert row["file"] == "raw/commons/a.ogg"
    assert row["license_class"] == "public_domain"
    assert row["bytes"] == "5"
    assert len(row["sha256"]) == 64
    assert row["retrieved_at"].endswith("Z")


def _commons(request: httpx.Request) -> httpx.Response:
    params = request.url.params
    if request.url.host == "upload.wikimedia.org":
        return httpx.Response(200, content=b"ogg-bytes")
    if params.get("list") == "categorymembers":
        if params["cmtitle"] == "Category:Audio files of speeches":
            members = [
                "File:Free.ogg",
                "File:Owned.ogg",
                "File:Photo.jpg",
                "File:Huge.ogg",
                "Category:Sub",
            ]
        elif params["cmtitle"] == "Category:Sub":
            members = ["File:Zero.ogg"]
        else:
            members = ["File:Free.ogg"]
        return httpx.Response(
            200, json={"query": {"categorymembers": [{"title": t} for t in members]}}
        )
    licenses = {"File:Free.ogg": "Public domain", "File:Owned.ogg": "CC BY-SA 4.0",
                "File:Huge.ogg": "Public domain", "File:Zero.ogg": "CC0"}  # fmt: skip
    pages = {
        str(i): {
            "title": title,
            "imageinfo": [{
                "url": f"https://upload.wikimedia.org/x/{title[5:]}?utm_source=commons",
                "size": 900_000_000 if title == "File:Huge.ogg" else 9,
                "descriptionurl": f"https://commons.wikimedia.org/wiki/{title}",
                "extmetadata": {"LicenseShortName": {"value": licenses[title]}},
            }],
        }
        for i, title in enumerate(params["titles"].split("|"))
    }  # fmt: skip
    return httpx.Response(200, json={"query": {"pages": pages}})


def test_commons_keeps_only_public_domain_audio_under_the_size_cap(tmp_path: Path) -> None:
    register = RightsRegister(tmp_path)

    failed = acquire_commons(tmp_path, _fetcher(_commons), register)

    assert failed == []
    assert sorted(p.name for p in (tmp_path / "raw" / "commons").iterdir()) == [
        "Free.ogg",
        "Zero.ogg",
    ]
    assert {r["item_id"]: r["license_class"] for r in _rows(tmp_path)} == {
        "File:Free.ogg": "public_domain",
        "File:Zero.ogg": "cc0",
    }
    catalog = json.loads((tmp_path / "manifests" / "commons_speech_catalog.json").read_text())
    assert [row["title"] for row in catalog] == [
        "File:Free.ogg",
        "File:Huge.ogg",
        "File:Owned.ogg",
        "File:Zero.ogg",
    ]


def test_commons_is_resumable_and_reports_failed_files(tmp_path: Path) -> None:
    register = RightsRegister(tmp_path)
    downloads: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "upload.wikimedia.org":
            downloads.append(request.url.path)
            if request.url.path.endswith("Zero.ogg"):
                return httpx.Response(404)
        return _commons(request)

    first = acquire_commons(tmp_path, _fetcher(handler), register)
    second = acquire_commons(tmp_path, _fetcher(handler), register)

    assert first == second == ["File:Zero.ogg"]
    assert downloads.count("/x/Free.ogg") == 1


def _archive(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/advancedsearch.php":
        docs = [
            {"identifier": i}
            for i in ("famous_orations_1", "cookbook", "gettysburg_address_librivox")
        ]
        return httpx.Response(200, json={"response": {"docs": docs}})
    if path.startswith("/metadata/"):
        item = path.rsplit("/", 1)[-1]
        files = [{"name": f"{item}.mp3"}, {"name": f"{item}_64kb.mp3"}, {"name": "cover.jpg"}]
        return httpx.Response(
            200, json={"metadata": {"licenseurl": "https://pd.example"}, "files": files}
        )
    return httpx.Response(200, content=b"mp3!")


def test_librivox_downloads_core_items_and_catalogues_the_rest(tmp_path: Path) -> None:
    register = RightsRegister(tmp_path)
    fetcher = _fetcher(_archive)

    catalog = acquire_librivox(tmp_path, fetcher, register)
    picks = acquire_librivox_orations(tmp_path, fetcher, register)

    assert catalog == ["cookbook", "famous_orations_1", "gettysburg_address_librivox"]
    assert picks == ["famous_orations_1"]
    core = tmp_path / "raw" / "librivox" / "gettysburg_address_librivox"
    assert [p.name for p in core.iterdir()] == ["gettysburg_address_librivox.mp3"]
    orations = tmp_path / "raw" / "librivox_orations" / "famous_orations_1"
    assert [p.name for p in orations.iterdir()] == ["famous_orations_1_64kb.mp3"]
    assert {r["license_class"] for r in _rows(tmp_path)} == {"public_domain"}
    assert {r["source_id"] for r in _rows(tmp_path)} == {"S-LV", "S-LV-ORATIONS"}


def test_librivox_orations_stop_at_the_size_budget(tmp_path: Path) -> None:
    register = RightsRegister(tmp_path)
    fetcher = _fetcher(_archive)
    acquire_librivox(tmp_path, fetcher, register)

    acquire_librivox_orations(tmp_path, fetcher, register, max_bytes=0)

    assert not (tmp_path / "raw" / "librivox_orations").exists()


def test_podcastfillers_labels_are_fetched_once_as_non_commercial(tmp_path: Path) -> None:
    register = RightsRegister(tmp_path)
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, content=b"event,start\n")

    for _ in range(2):
        acquire_podcastfillers_labels(tmp_path, _fetcher(handler), register)

    assert len(calls) == 1
    assert _rows(tmp_path)[0]["license_class"] == "nc_research"
