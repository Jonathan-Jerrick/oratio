# Methodology

Oratio scores delivery against a reference reading of the same words. It does not ask whether a speech is good in the abstract. It asks where this recording departs from a strong reading of the same text, by how much, and whether that departure is larger than the variation good speakers show among themselves.

Every number in a report traces back to a measurement. Models find the candidate regions, and the measurement engine (`packages/speechgrade`) explains them.

## 1. The contrastive setup

A report compares two recordings:

- **Participant.** The recording being scored.
- **Baseline.** A strong reading of the same text. In the dataset this is the clean source passage. In the app it is a reference chosen from the published baselines.

Both recordings are force-aligned to their words with a CTC acoustic model (wav2vec2-base-960h, run as ONNX). The two word sequences are then paired by text. Every comparison that follows is made between a participant word and its paired baseline word, so the system never has to guess what "normal" sounds like for a particular sentence.

A piecewise-linear time map is built from the matched words. It lets the dashboard play the baseline at the moment that corresponds to a flagged participant region, and draw the baseline's pitch and loudness under the participant's.

## 2. Measurements

The engine extracts frame tracks from each recording:

- pitch in semitones (`f0_st`)
- loudness in dB (`energy_db`)
- spectral brightness, as the high-frequency energy ratio (`hf_ratio`)
- voice clarity, as cepstral peak prominence (`cpp_db`)

From those tracks it computes one feature vector per word:

| Feature | What it captures |
|---|---|
| `log_duration`, `relative_length` | How long the word takes, alone and relative to its neighbours |
| `pause_after` | Silence after the word |
| `rate_variability` | How unevenly the local speaking rate moves |
| `f0_range_st`, `f0_slope_st`, `f0_mean_st`, `f0_peak_st` | Pitch movement and contour |
| `fry_fraction` | Share of creaky, very low-pitched frames |
| `energy_mean_db`, `energy_range_db`, `energy_trend_db` | Loudness level, spread and decay |
| `hf_balance_db`, `cpp_db` | Brightness and voice clarity |
| `nuclei_per_syllable` | Syllable nuclei heard against syllables expected, a swallowed-word cue |
| `prominence` | How much the word stands out from its neighbours |

## 3. From differences to deviations

For each matched word pair, the engine takes the difference between participant and baseline on every feature. It divides that difference by `sigma_good`, the spread of the same feature across good readings, which gives a z-score.

A rule fires when a z-score passes its threshold in the rule's direction. Most rules use |z| > 2. Prolongation uses |z| > 3, and when it fires it suppresses the pacing rules for the same word, so one long vowel is not counted three times.

The rubric (`speechgrade/data/rubric.json`, version 0.2.0) holds every threshold, direction and weight. Its SHA-256 is written into each report's provenance.

## 4. Flaw codes

The 23 codes fall under seven dimensions:

| Dimension | Codes |
|---|---|
| Pacing | PAC-RUSH, PAC-DRAG, PAC-ERR |
| Pausing | PAU-DROP, PAU-LONG, PAU-MID |
| Intonation | INT-MONO, INT-EXAG, INT-UP |
| Vocal energy | ENE-LOW, ENE-SPIKE, ENE-TRAIL, ENE-FLAT |
| Emphasis | EMP-DROP, EMP-MISP |
| Fluency | FLU-REP, FLU-FALSE, FLU-PROL, FLU-OMIT, FLU-FILL |
| Clarity | CLA-MUFF, CLA-SWAL, VQ-FRY |

Fillers (um, uh, er and similar) and other fluency events are found from the aligned word sequence, not from per-word z-scores.

## 5. Temporal grounding

Adjacent words that fire the same code are merged into one region, bridging gaps of up to one word. A region carries:

- the start and end time in seconds
- the words it spans
- the measurement that triggered it: feature, participant value, baseline value, difference, `sigma_good`, z and the threshold
- a plain-language rationale and a fix

The dashboard draws regions as coloured chips over the pitch, loudness, brightness and clarity lanes. Clicking a chip plays that stretch, then the matching stretch of the baseline.

## 6. Scoring

Each region costs points:

```
excess          = min(max(|z| - z_threshold, 0), 4)
duration_weight = max(min(1, sqrt(seconds / 2.0)), 0.3)
deduction       = weight * excess * duration_weight
```

Each dimension starts at 10 and loses the deductions of its regions, with a floor of 0. The overall score is the mean of the seven dimension scores times 10, so it runs from 0 to 100.

Every lost point belongs to a region you can click. Nothing is deducted for a reason the report cannot show.

## 7. Learned detector

NB03 trains a LightGBM multiclass model on one row per participant word. Each row holds the word's z-deviations and raw features, and its label is the injected code covering the word's midpoint (or `none`).

- **Training.** Class weights balance the rare codes. Early stopping runs on the validation split.
- **Evaluation.** NB03 reports word-level macro F1 for the model and for the fixed rules on the same test words, so the gain over the rules is measured, not assumed.
- **Speed.** LightGBM gives exact per-feature attributions (TreeSHAP, via `pred_contrib`) in milliseconds on CPU. The deployment spike times this call on the Lambda CPU alongside the encoder.

## 8. Evaluation

Everything is computed automatically on the held-out test split; no step needs a human listener.

| Check | How it is measured |
|---|---|
| Event grounding | A detected region matches an injected one when the dimension agrees and IoU ≥ 0.3. NB03 reports precision, recall, F1 and median onset error in milliseconds. |
| Per-dimension F1 | The same matching, broken down by dimension |
| Severity ordering | Spearman correlation between injected level (L0–L5) and overall score, plus the mean score at each level |
| Dose response | For single-flaw sweeps at magnitudes 0.2 to 1.0, the detection rate at each magnitude. It should rise with magnitude. |
| Label integrity | Generated clips are re-aligned (every control, and a fixed sample of flawed clips when no GPU is free) and rejected if the median word drift from their labels exceeds 30 ms |
| Signal gates | Clipping, digital silence and duration checks on every clip |

The live numbers are on the Benchmarks page of the dashboard and in `metrics.json` on the dataset repository.

## 9. Serving

- **API.** `services/api` runs FastAPI in one container image on AWS Lambda (ap-south-1) behind a Function URL. It streams progress as server-sent events: align, features, detect, score.
- **Pinned models.** Every model file is pinned in `artifacts.lock.json` by Hugging Face revision and SHA-256. The image build fails on any mismatch, and nothing is downloaded at start-up.
- **Feedback.** The judge panel writes per-dimension feedback with an LLM (Groq). A fact-checker compares every number in the text against the report and rejects text that cites a value the report does not contain. When no key is set, a template writes the feedback from the report alone.

## 10. Limits

- The baseline is one reading. A different but equally good choice of emphasis can be flagged. `sigma_good` softens this but does not remove it.
- Synthetic flaws are cleaner than natural ones. Scores on real recordings are expected to be noisier than the test-split numbers.
- The dataset is English only.
