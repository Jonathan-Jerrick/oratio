"""NB03: word-level training table, LightGBM detector, automated validation, samples, publish."""

import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from oratio_data.licenses import require_publishable
from speechgrade.align import WordSpan
from speechgrade.audio import Audio
from speechgrade.compare import deviations, pair_words
from speechgrade.features import extract_tracks
from speechgrade.pipeline import analyze
from speechgrade.rubric import load_rubric
from speechgrade.words import word_features

NONE = "none"
WORD_LEVEL_SKIP = {"FLU-OMIT", "FLU-REP", "FLU-FALSE", "FLU-FILL"}
IOU_MATCH = 0.3
LGBM_ROUNDS = 400
SAMPLE_PICKS = (
    ("S-LV", 3),
    ("S-LV", 5),
    ("S-PD-COMMONS", 2),
    ("S-PD-COMMONS", 4),
    ("S-EARN", 1),
    ("S-EARN", 3),
    ("S-LV-ORATIONS", 4),
    ("S-LV-ORATIONS", 2),
)


def _labels(root: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted((root / "ocdd" / "labels").glob("*.json"))]


def _words(record: dict[str, Any]) -> list[WordSpan]:
    return [WordSpan(w["text"], w["start"], w["end"], 0.0) for w in record["words"]]


def _audio(root: Path, record: dict[str, Any]) -> Audio:
    samples, rate = sf.read(root / "ocdd" / record["audio"], dtype="float32")
    return Audio(samples, rate)


def word_rows(root: Path, record: dict[str, Any], control: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per participant word: z-deviations, raw features, and the injected code (or none)."""
    rubric = load_rubric()
    words, base_words = _words(record), _words(control)
    ours = word_features(words, extract_tracks(_audio(root, record).samples))
    theirs = word_features(base_words, extract_tracks(_audio(root, control).samples))
    pairs = pair_words([w.word for w in words], [w.word for w in base_words])
    z = deviations(ours, theirs, pairs, rubric.sigma_good)
    rows = []
    for index, word in enumerate(words):
        if index not in z:
            continue
        middle = (word.start + word.end) / 2
        code = next(
            (
                r["code"]
                for r in record["regions"]
                if r["code"] not in WORD_LEVEL_SKIP and r["start"] <= middle <= r["end"]
            ),
            NONE,
        )
        rows.append(
            {
                "clip_id": record["clip_id"],
                "passage_id": record["passage_id"],
                "split": record["split"],
                "word": index,
                "label": code,
                **{f"z_{k}": v for k, v in z[index].items()},
                **{f"p_{k}": v for k, v in ours[index].items()},
            }
        )
    return rows


def build_table(root: Path) -> Path:
    """Word-level features for every flawed clip against its control; written once, reused."""
    import pandas as pd

    out = root / "features" / "words.parquet"
    if out.exists():
        return out
    records = _labels(root)
    controls = {r["clip_id"]: r for r in records if r["kind"] == "control"}
    rows: list[dict[str, Any]] = []
    flawed = [r for r in records if r["kind"] != "control" and r["baseline_id"] in controls]
    for number, record in enumerate(flawed, start=1):
        rows.extend(word_rows(root, record, controls[record["baseline_id"]]))
        if number % 50 == 0:
            print(f"{number}/{len(flawed)} clips", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out)
    return out


def train(root: Path) -> dict[str, Any]:
    """LightGBM over word rows; reports macro F1 against the rules on the test split."""
    import lightgbm as lgb
    import pandas as pd
    from sklearn.metrics import f1_score

    table = pd.read_parquet(build_table(root))
    features = [c for c in table.columns if c.startswith(("z_", "p_"))]
    classes = sorted(table["label"].unique())
    index = {c: i for i, c in enumerate(classes)}
    parts = {s: table[table["split"] == s] for s in ("train", "val", "test")}
    counts = parts["train"]["label"].value_counts()
    weight = parts["train"]["label"].map(lambda c: len(parts["train"]) / (len(classes) * counts[c]))
    params = {
        "objective": "multiclass",
        "num_class": len(classes),
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "seed": 0,
        "deterministic": True,
        "num_threads": 4,
        "verbose": -1,
    }
    data = lgb.Dataset(parts["train"][features], parts["train"]["label"].map(index), weight=weight)
    valid = lgb.Dataset(parts["val"][features], parts["val"]["label"].map(index))
    booster = lgb.train(
        params,
        data,
        LGBM_ROUNDS,
        valid_sets=[valid],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    models = root / "models"
    models.mkdir(exist_ok=True)
    booster.save_model(str(models / "lgbm_flaw.txt"))
    (models / "lgbm_classes.json").write_text(
        json.dumps({"classes": classes, "features": features})
    )
    test = parts["test"]
    predicted = [classes[i] for i in np.asarray(booster.predict(test[features])).argmax(axis=1)]
    rules = _rules_word_labels(test)
    labels = [c for c in classes if c != NONE]
    return {
        "lgbm_word_macro_f1": float(
            f1_score(test["label"], predicted, labels=labels, average="macro")
        ),
        "rules_word_macro_f1": float(
            f1_score(test["label"], rules, labels=labels, average="macro")
        ),
        "test_words": len(test),
        "classes": classes,
        "best_iteration": booster.best_iteration,
    }


def _rules_word_labels(table: Any) -> list[str]:
    rubric = load_rubric()
    out = []
    for _, row in table.iterrows():
        hits = [
            (abs(row[f"z_{r.feature}"]), r.code)
            for r in rubric.rules
            if f"z_{r.feature}" in row and r.direction * row[f"z_{r.feature}"] > r.z_threshold
        ]
        out.append(max(hits)[1] if hits else NONE)
    return out


def _iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def evaluate(root: Path) -> dict[str, Any]:
    """Event-level grounding on test clips, score-vs-level, dose-response and invariance."""
    from scipy.stats import spearmanr

    records = _labels(root)
    controls = {r["clip_id"]: r for r in records if r["kind"] == "control"}
    test = [r for r in records if r["split"] == "test" and r["kind"] != "control"]
    hits = misses = false = 0
    onset_errors: list[float] = []
    levels: list[int] = []
    scores: list[float] = []
    dose: dict[str, dict[float, list[bool]]] = defaultdict(lambda: defaultdict(list))
    per_dimension: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for record in test:
        control = controls[record["baseline_id"]]
        report = analyze(
            _audio(root, record), _words(record), _audio(root, control), _words(control)
        )
        found = [(r.dimension, (r.start, r.end)) for r in report.regions]
        truth = [
            (r["dimension"], (r["start"], max(r["end"], r["start"] + 0.05)))
            for r in record["regions"]
        ]
        used: set[int] = set()
        for dimension, span in truth:
            match = next(
                (
                    i
                    for i, (d, s) in enumerate(found)
                    if i not in used and d == dimension and _iou(s, span) >= IOU_MATCH
                ),
                None,
            )
            if match is None:
                misses += 1
                per_dimension[dimension][1] += 1
            else:
                used.add(match)
                hits += 1
                per_dimension[dimension][0] += 1
                onset_errors.append(abs(found[match][1][0] - span[0]) * 1000)
            if record["kind"] == "sweep":
                dose[record["code"]][record["magnitude"]].append(match is not None)
        false += len(found) - len(used)
        for i in set(range(len(found))) - used:
            per_dimension[found[i][0]][2] += 1
        if record["kind"] == "graded":
            levels.append(record["level"])
            scores.append(report.scores.overall)
    for _ in [c for c in controls.values() if c["split"] == "test"]:
        levels.append(0)
        scores.append(100.0)
    precision = hits / max(1, hits + false)
    recall = hits / max(1, hits + misses)
    rho = spearmanr(levels, scores).statistic if len(set(levels)) > 1 else float("nan")
    return {
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": 2 * precision * recall / max(1e-9, precision + recall),
        "median_onset_error_ms": float(np.median(onset_errors)) if onset_errors else None,
        "spearman_score_vs_level": float(rho),
        "per_dimension_f1": {
            d: 2 * h / max(1, 2 * h + m + f) for d, (h, m, f) in per_dimension.items()
        },
        "dose_response": {
            c: {f"{m:.1f}": float(np.mean(v)) for m, v in sorted(by.items())}
            for c, by in dose.items()
        },
        "mean_score_by_level": {
            str(lv): float(np.mean([s for l2, s in zip(levels, scores, strict=True) if l2 == lv]))
            for lv in sorted(set(levels))
        },
        "test_clips": len(test),
    }


def export_samples(root: Path) -> list[dict[str, Any]]:
    """A handful of test clips with reports, audio and truth, in the dashboard's index format."""
    records = _labels(root)
    controls = {r["clip_id"]: r for r in records if r["kind"] == "control"}
    out = root / "ocdd" / "samples"
    out.mkdir(parents=True, exist_ok=True)
    index = []
    used: set[str] = set()
    for source, level in SAMPLE_PICKS:
        pick = next(
            (
                r
                for r in records
                if r["source_id"] == source
                and r["level"] == level
                and r["kind"] == "graded"
                and r["variant"] == 0
                and r["passage_id"] not in used
                and r["split"] == "test"
            ),
            None,
        ) or next(
            (
                r
                for r in records
                if r["source_id"] == source
                and r["level"] == level
                and r["kind"] == "graded"
                and r["passage_id"] not in used
            ),
            None,
        )
        if pick is None:
            continue
        used.add(pick["passage_id"])
        control = controls[pick["baseline_id"]]
        report = analyze(_audio(root, pick), _words(pick), _audio(root, control), _words(control))
        sample_id = pick["clip_id"]
        (out / f"{sample_id}.json").write_text(report.model_dump_json())
        (out / f"{sample_id}.labels.json").write_text(json.dumps(pick["regions"]))
        for record, name in ((pick, sample_id), (control, control["clip_id"])):
            target = out / f"{name}.mp3"
            if not target.exists():
                subprocess.run(  # noqa: S603
                    [  # noqa: S607
                        "ffmpeg",
                        "-nostdin",
                        "-v",
                        "error",
                        "-y",
                        "-i",
                        str(root / "ocdd" / record["audio"]),
                        "-b:a",
                        "64k",
                        str(target),
                    ],
                    check=True,
                )
        index.append(
            {
                "id": sample_id,
                "title": _title(pick),
                "speaker": pick.get("speaker") or pick["source_id"],
                "source": pick["source_id"],
                "license": pick["license_class"],
                "level": pick["level"],
                "baseline_id": control["clip_id"],
                "report": f"{sample_id}.json",
                "audio": f"{sample_id}.mp3",
                "baseline_audio": f"{control['clip_id']}.mp3",
                "overall": report.scores.overall,
                "truth": f"{sample_id}.labels.json",
            }
        )
    (out / "index.json").write_text(json.dumps(index, indent=1))
    return index


def _title(record: dict[str, Any]) -> str:
    name = record["passage_id"].split("__")
    return str(name[-2].replace("_", " ")[:80] if len(name) >= 2 else record["passage_id"])


def publish(root: Path, metrics: dict[str, Any], token: str) -> dict[str, str]:
    """Upload OCDD and the detector to public Hub repos, refusing any non-publishable clip."""
    from huggingface_hub import HfApi

    for record in _labels(root):
        require_publishable(record["license_class"], record["clip_id"])
    api = HfApi(token=token)
    owner = api.whoami()["name"]
    dataset, models = f"{owner}/oratio-ocdd", f"{owner}/oratio-models"
    api.create_repo(dataset, repo_type="dataset", exist_ok=True, private=False)
    api.create_repo(models, repo_type="model", exist_ok=True, private=False)
    (root / "ocdd" / "metrics.json").write_text(json.dumps(metrics, indent=1))
    data_commit = api.upload_folder(
        folder_path=str(root / "ocdd"),
        repo_id=dataset,
        repo_type="dataset",
        commit_message="OCDD v1",
    )
    model_commit = api.upload_folder(
        folder_path=str(root / "models"),
        repo_id=models,
        repo_type="model",
        commit_message="LightGBM flaw detector v1",
    )
    return {
        "dataset": dataset,
        "dataset_commit": data_commit.oid,
        "models": models,
        "models_commit": model_commit.oid,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
