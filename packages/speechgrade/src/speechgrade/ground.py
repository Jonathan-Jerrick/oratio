"""Turn word-level deviations into time-bounded, explained flaw regions."""

from dataclasses import dataclass

from speechgrade.align import WordSpan
from speechgrade.compare import Pair
from speechgrade.rubric import Rubric, Rule
from speechgrade.schema import Measurement, Region
from speechgrade.score import deduction
from speechgrade.words import WordFeatures

FLUENCY = "fluency"
_PAUSE_FEATURE = "pause_after"


@dataclass(frozen=True, slots=True)
class _Finding:
    code: str
    dimension: str
    start: float
    end: float
    words: list[str]
    z: float
    z_threshold: float
    weight: float
    measurement: Measurement | None


def _runs(indices: list[int], max_step: int) -> list[list[int]]:
    runs: list[list[int]] = []
    for index in indices:
        if runs and index - runs[-1][-1] <= max_step:
            runs[-1].append(index)
        else:
            runs.append([index])
    return runs


def _rule_findings(
    rule: Rule,
    words: list[WordSpan],
    features: list[WordFeatures],
    baseline_features: list[WordFeatures],
    baseline_of: dict[int, int],
    z_scores: dict[int, dict[str, float]],
    rubric: Rubric,
) -> list[_Finding]:
    def applies(index: int) -> bool:
        reference = baseline_features[baseline_of[index]][rule.feature]
        above = rule.baseline_min is None or reference >= rule.baseline_min
        below = rule.baseline_max is None or reference < rule.baseline_max
        return above and below and rule.direction * z_scores[index][rule.feature] > rule.z_threshold

    flagged = sorted(index for index in z_scores if applies(index))
    info = rubric.features[rule.feature]
    findings: list[_Finding] = []
    for run in _runs(flagged, rubric.scoring.merge_gap_words + 1):
        peak = max(run, key=lambda index: abs(z_scores[index][rule.feature]))
        ours = features[peak][rule.feature]
        theirs = baseline_features[baseline_of[peak]][rule.feature]
        end = words[run[-1]].end
        if rule.feature == _PAUSE_FEATURE and run[-1] + 1 < len(words):
            end = words[run[-1] + 1].start
        findings.append(
            _Finding(
                code=rule.code,
                dimension=rule.dimension,
                start=words[run[0]].start,
                end=end,
                words=[words[index].word for index in range(run[0], run[-1] + 1)],
                z=z_scores[peak][rule.feature],
                z_threshold=rule.z_threshold,
                weight=rule.weight,
                measurement=Measurement(
                    feature=rule.feature,
                    label=info.label,
                    unit=info.unit,
                    participant=round(ours, 4),
                    baseline=round(theirs, 4),
                    delta=round(ours - theirs, 4),
                    sigma_good=rubric.sigma_good[rule.feature],
                    z=round(z_scores[peak][rule.feature], 4),
                    z_threshold=rule.z_threshold,
                ),
            )
        )
    return findings


def _insertion_code(words: list[WordSpan], index: int, fillers: tuple[str, ...]) -> str:
    word = words[index].word
    if word in fillers:
        return "FLU-FILL"
    neighbours = {words[i].word for i in (index - 1, index + 1) if 0 <= i < len(words)}
    return "FLU-REP" if word in neighbours else "FLU-FALSE"


def _event_findings(
    words: list[WordSpan], baseline_words: list[WordSpan], pairs: list[Pair], rubric: Rubric
) -> list[_Finding]:
    events = rubric.events

    def finding(code: str, start: float, end: float, text: list[str]) -> _Finding:
        return _Finding(
            code, FLUENCY, start, end, text, events.z, events.z_threshold, events.weight, None
        )

    findings: list[_Finding] = []
    inserted = [
        pair.participant for pair in pairs if pair.participant is not None and pair.baseline is None
    ]
    by_code: dict[str, list[int]] = {}
    for index in inserted:
        by_code.setdefault(_insertion_code(words, index, events.fillers), []).append(index)
    for code, indices in by_code.items():
        for run in _runs(indices, 1):
            text = [words[index].word for index in run]
            findings.append(finding(code, words[run[0]].start, words[run[-1]].end, text))

    cursor = 0.0
    omitted: list[str] = []
    for pair in pairs:
        if pair.participant is None and pair.baseline is not None:
            omitted.append(baseline_words[pair.baseline].word)
            continue
        if pair.participant is None:
            continue
        if omitted:
            findings.append(finding("FLU-OMIT", cursor, words[pair.participant].start, omitted))
            omitted = []
        cursor = words[pair.participant].end
    if omitted:
        findings.append(finding("FLU-OMIT", cursor, cursor, omitted))
    return findings


def _suppress(findings: list[_Finding], rubric: Rubric) -> list[_Finding]:
    """Drop a finding that a more specific rule already explains over the same span."""
    weaker = {rule.code: set(rule.suppresses) for rule in rubric.rules if rule.suppresses}
    return [
        item
        for item in findings
        if not any(
            item.code in weaker.get(other.code, ())
            and other.start <= item.start
            and item.end <= other.end
            for other in findings
        )
    ]


def _explain(finding: _Finding, rubric: Rubric) -> tuple[str, str]:
    text = " ".join(finding.words).lower()
    fields = {
        "words": text,
        "first_word": finding.words[0].lower(),
        "last_word": finding.words[-1].lower(),
    }
    fix = rubric.fixes[finding.code].format(**fields)
    if finding.measurement is None:
        return rubric.rationales[finding.code].format(start=finding.start, **fields), fix
    measured = finding.measurement
    rationale = rubric.rationales["measured"].format(
        seconds=finding.end - finding.start,
        label=measured.label,
        unit=measured.unit,
        participant=measured.participant,
        baseline=measured.baseline,
        z=abs(measured.z),
        side="below" if measured.z < 0 else "above",
        **fields,
    )
    return rationale, fix


def find_regions(
    words: list[WordSpan],
    baseline_words: list[WordSpan],
    features: list[WordFeatures],
    baseline_features: list[WordFeatures],
    pairs: list[Pair],
    z_scores: dict[int, dict[str, float]],
    rubric: Rubric,
) -> list[Region]:
    """Flaw regions in time order, each with its measurement, deduction and explanation."""
    baseline_of = {
        pair.participant: pair.baseline
        for pair in pairs
        if pair.participant is not None and pair.baseline is not None
    }
    findings = _event_findings(words, baseline_words, pairs, rubric)
    for rule in rubric.rules:
        findings.extend(
            _rule_findings(rule, words, features, baseline_features, baseline_of, z_scores, rubric)
        )
    findings = _suppress(findings, rubric)
    findings.sort(key=lambda item: (item.start, item.end, item.code))

    regions: list[Region] = []
    for number, item in enumerate(findings, start=1):
        rationale, fix = _explain(item, rubric)
        points = deduction(
            item.z, item.z_threshold, item.weight, item.end - item.start, rubric.scoring
        )
        regions.append(
            Region(
                region_id=f"R{number}",
                code=item.code,
                dimension=item.dimension,
                start=round(item.start, 4),
                end=round(item.end, 4),
                words=" ".join(item.words).lower(),
                measurement=item.measurement,
                z=round(item.z, 4),
                deduction=round(points, 4),
                rationale=rationale,
                fix=fix,
            )
        )
    return regions
