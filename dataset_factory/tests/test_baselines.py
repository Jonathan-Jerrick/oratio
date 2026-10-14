from itertools import pairwise

import numpy as np
import pytest

from oratio_data.baselines import (
    MAX_PASSAGE_S,
    MIN_PASSAGE_S,
    Word,
    assign_speakers,
    bitrate_rank,
    cut_by_speaker,
    cut_passages,
    energy_snr_db,
    prepared_remarks_end,
    read_nlp,
    recording_key,
    screen,
    sentence_ends,
)
from oratio_data.dsp import SAMPLE_RATE


def _speech(
    sentences: int,
    words_per_sentence: int = 12,
    gap: float = 0.5,
    word_s: float = 0.3,
    inner: float = 0.1,
) -> list[Word]:
    """Sentences of 0.3 s words separated by 0.1 s, each ending with a full stop and a pause."""
    words, clock = [], 0.0
    for s in range(sentences):
        for w in range(words_per_sentence):
            last = w == words_per_sentence - 1
            words.append(Word(f"word{s}_{w}{'.' if last else ''}", clock, clock + word_s, 0.95))
            clock += word_s + (gap if last else inner)
    return words


def test_sentence_ends_need_punctuation_and_a_pause_or_a_long_pause() -> None:
    words = [
        Word("We", 0.0, 0.3, 0.9),
        Word("shall.", 0.4, 0.8, 0.9),
        Word("Pay", 1.2, 1.5, 0.9),
        Word("any.", 1.55, 1.8, 0.9),
        Word("price", 1.85, 2.2, 0.9),
        Word("bear", 3.0, 3.3, 0.9),
    ]

    assert sentence_ends(words) == [1, 4, 5]


def test_passages_are_sentence_aligned_and_within_bounds() -> None:
    words = _speech(60)

    passages = cut_passages(words, max_passages=10)

    assert len(passages) >= 3
    for passage in passages:
        assert MIN_PASSAGE_S <= passage.seconds <= MAX_PASSAGE_S + 0.6
        assert words[passage.last].text.endswith(".")
        assert passage.first == 0 or words[passage.first - 1].text.endswith(".")
        assert passage.start <= words[passage.first].start
        assert passage.end >= words[passage.last].end
    assert all(a.last < b.first for a, b in pairwise(passages))


def test_passages_keep_a_little_room_tone_but_never_borrow_neighbouring_words() -> None:
    words = _speech(30)
    first, second = cut_passages(words, max_passages=2)

    assert first.start == 0.0
    assert first.end <= words[first.last + 1].start
    assert second.start >= words[second.first - 1].end
    assert second.start == pytest.approx(words[second.first].start - 0.3)


def test_unbroken_monologue_longer_than_the_maximum_is_skipped() -> None:
    run_on = [Word(f"w{i}", i * 0.4, i * 0.4 + 0.3, 0.9) for i in range(300)]
    closing = Word("end.", 120.0, 120.3, 0.9)
    later = [Word(w.text, w.start + 121.0, w.end + 121.0, 0.9) for w in _speech(20)]

    passages = cut_passages([*run_on, closing, *later], max_passages=5)

    assert passages
    assert all(p.first > 300 for p in passages)


def test_max_passages_is_respected() -> None:
    assert len(cut_passages(_speech(60), max_passages=1)) == 1
    assert cut_passages(_speech(3), max_passages=5) == []


def test_screen_flags_low_confidence_odd_rate_and_mixed_speakers() -> None:
    words = _speech(14)
    (passage,) = cut_passages(words, max_passages=1)
    shaky = [Word(w.text, w.start, w.end, 0.2 if i % 5 == 0 else 0.9) for i, w in enumerate(words)]
    mixed = [Word(w.text, w.start, w.end, 0.9, "A" if i < 20 else "B") for i, w in enumerate(words)]

    assert screen(words, passage) == []
    assert screen(shaky, passage) == ["low_asr_confidence"]
    assert screen(mixed, passage) == ["multiple_speakers"]
    hurried = _speech(30, gap=0.3, word_s=0.15, inner=0.02)
    assert "speech_rate" in screen(hurried, cut_passages(hurried, 1)[0])


def test_energy_snr_separates_clean_from_noisy_audio() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(10 * SAMPLE_RATE) / SAMPLE_RATE
    speech = 0.3 * np.sin(2 * np.pi * 150 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0)
    clean = (speech + 1e-4 * rng.standard_normal(len(t))).astype(np.float32)
    noisy = (speech + 0.05 * rng.standard_normal(len(t))).astype(np.float32)

    assert energy_snr_db(clean) > 40
    assert energy_snr_db(noisy) < 20
    assert energy_snr_db(clean[:100]) == 0.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("thank you we will now begin the question-and-answer session", 7),
        ("our revenue grew. operator, please open the line for questions", 5),
        ("our revenue grew ten percent this quarter", 7),
        ("q&a will follow", 0),
    ],
)
def test_prepared_remarks_end_at_the_q_and_a(text: str, expected: int) -> None:
    words = [Word(token, i, i + 0.5, 0.9) for i, token in enumerate(text.split())]

    assert prepared_remarks_end(words) == expected


def test_speakers_follow_the_reference_transcript_across_asr_errors() -> None:
    reference = read_nlp(
        "token|speaker|ts|endTs|punctuation|prepunctuation|case|tags\n"
        + "\n".join(
            f"{token}|{speaker}|||||LC|[]"
            for token, speaker in [
                ("Good", "0"), ("morning", "0"), ("everyone", "0"),
                ("Thanks", "1"), ("operator", "1"), ("revenue", "1"), ("grew", "1"),
            ]
        )
    )  # fmt: skip
    asr = [
        Word(t, i, i + 0.4, 0.9)
        for i, t in enumerate(
            ["Good", "morning,", "every", "one.", "Thanks,", "operator.", "Revenue", "grew."]
        )
    ]

    labelled = assign_speakers(asr, reference)

    assert [w.speaker for w in labelled] == ["0", "0", "0", "1", "1", "1", "1", "1"]
    assert assign_speakers(asr, [])[0].speaker == ""


def test_passages_never_cross_a_change_of_speaker() -> None:
    first, second = _speech(14), _speech(14)
    offset = first[-1].end + 1.0
    words = [Word(w.text, w.start, w.end, 0.9, "ceo") for w in first] + [
        Word(w.text, w.start + offset, w.end + offset, 0.9, "cfo") for w in second
    ]

    passages = cut_by_speaker(words, max_passages=5)

    assert len(passages) == 2
    assert all(len({w.speaker for w in words[p.first : p.last + 1]}) == 1 for p in passages)
    assert passages[1].first >= len(first)
    assert all(screen(words, p) == [] for p in passages)
    assert len(cut_by_speaker(words, max_passages=1)) == 1
    assert len(cut_passages(words, max_passages=5)) >= 1


@pytest.mark.parametrize(
    ("path", "key", "rank"),
    [
        ("raw/lv/item/lincoln_amb.mp3", "raw/lv/item/lincoln_amb", 0),
        ("raw/lv/item/lincoln_amb_128kb.mp3", "raw/lv/item/lincoln_amb", 1),
        ("raw/lv/item/lincoln_amb_64kb.mp3", "raw/lv/item/lincoln_amb", 2),
        ("raw/commons/Speech_1964.ogg", "raw/commons/Speech_1964", 0),
    ],
)
def test_bitrate_variants_share_one_recording_key(path: str, key: str, rank: int) -> None:
    assert recording_key(path) == key
    assert bitrate_rank(path) == rank
