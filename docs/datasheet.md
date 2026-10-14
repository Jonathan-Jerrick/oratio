# Datasheet: Oratio Contrastive Delivery Dataset (OCDD) v1

This datasheet follows the questions in Gebru et al., *Datasheets for Datasets*. Generation code lives in `dataset_factory/` and runs in notebooks NB01 to NB03.

## Motivation

**Why was it built?**
Track C of the Multimodal AI Hackathon 2026 asks for models that compare a speech with a reference, find the exact seconds where delivery falls short, and explain why. No public dataset pairs a good reading with degraded readings of the same words and labels each flaw with its code, severity and timing. OCDD fills that gap.

**Who built it?**
Jonathan Jerrick, for the hackathon entry Oratio.

## Composition

**What does an instance look like?**
An instance is one 16 kHz mono FLAC clip (16-bit PCM) with a JSON label record. Each record holds:

- `clip_id`, `baseline_id`, `passage_id`, `source_id`, `split`
- `kind`: `control`, `graded` or `sweep`
- `level`, `variant`, `code`, `magnitude`
- the word list with start and end times
- the injected regions, each with code, dimension, magnitude, start and end in clip time, and start and end in source time
- licence class, speaker and an evidence URL for the licence

**Which kinds of clip are there?**

| Kind | What it is |
|---|---|
| Control (`__L0p`) | The source passage passed through the same processing chain with no flaw applied. It is the baseline for every flawed clip made from that passage. |
| Graded (`__L1` to `__L5`, two variants each) | Several flaws at once, sized by level |
| Sweep | One flaw (PAC-RUSH, PAU-LONG, INT-MONO, ENE-LOW, CLA-MUFF or FLU-PROL) at magnitudes 0.2, 0.4, 0.6, 0.8 and 1.0, made from every fourth passage, for dose-response checks |

**How are severity levels defined?**

| Level | Regions | Magnitude | Max share of the clip covered | Whole-clip flaw allowed |
|---|---|---|---|---|
| L1 | 1 | 0.15–0.30 | 3 % | no |
| L2 | 2 | 0.25–0.45 | 8 % | no |
| L3 | 3–4 | 0.40–0.65 | 15 % | no |
| L4 | 5–7 | 0.60–0.85 | 30 % | no |
| L5 | 8–10 | 0.80–1.00 | 50 % | yes (INT-MONO, ENE-FLAT or CLA-MUFF) |

**Which flaws are injected?**
The 23 codes listed in [methodology.md](methodology.md), across pacing, pausing, intonation, vocal energy, emphasis, fluency and clarity.

**Where do the source recordings come from?**

| Source ID | Material | Licence class |
|---|---|---|
| S-LV | LibriVox readings, including several renditions of the Gettysburg Address | Public domain |
| S-LV-ORATIONS | LibriVox recordings of historical speeches | Public domain |
| S-PD-COMMONS | Speech recordings from Wikimedia Commons | Public domain or CC0 |
| S-EARN | Prepared remarks from Earnings-21 and Earnings-22 | CC BY-SA 4.0 |

Baseline passages per source in v1: S-LV 15, S-EARN 12, S-LV-ORATIONS 10, S-PD-COMMONS 10. Passages are chosen by signal-to-noise ratio, at most one per recording.

**Is anything left out on purpose?**
TED talks and any other non-redistributable material. `require_publishable` refuses any clip whose licence class is not public domain, CC0, CC BY, CC BY-SA or the author's own, and the publish step re-checks every record before upload. Non-commercial research sets can serve as external checks but are never copied into OCDD.

**Is there a recommended split?**
Yes: train 70 %, validation 15 %, test 15 %. The split comes from a SHA-256 hash of the source recording, so every passage, level and sweep made from one recording falls in the same split, and no speaker reading leaks from train into test.

**Does it contain personal or sensitive data?**
The voices are real people reading public material, released by them or their estates under the licences above. There are no names beyond the speaker credit in the source and no private speech.

## Collection and generation

**How was the audio acquired?**
NB01 downloads from the sources above. It records each file in an append-only register with its licence and evidence URL, transcribes with faster-whisper large-v3-turbo, and cuts passages capped by length per source.

**How were the flaws made?**
NB02 plans flaws with a fixed seed per passage and level, then applies them with signal-processing injectors: time-stretch, pause insertion and removal, pitch-contour flattening and exaggeration, gain shaping, spectral tilt, prolongation, repetition and filler insertion. Each injector also returns a warp map from source time to clip time, so the labels move with the audio when an edit changes its length.

**How were the labels checked?**
Automatically, with no listening step:

- **Re-alignment drift.** The flawed clip is force-aligned again. For the words no injector touched, the new word times are compared with the times the warp map predicts. A clip is rejected if the median drift exceeds 30 ms. Every control is audited, and so is a fixed sample of flawed clips when the run has no GPU.
- **Signal gates.** A clip is rejected if more than 0.1 % of its samples clip, if it holds more than 20 ms of digital silence that the source did not have, if its length is off by more than 10 ms from what the warp map predicts, or if any label falls outside the clip.
- **Plan checks.** A plan is refused if it breaks its level's limits on region count, magnitude or coverage.

A plan that cannot be placed is retried up to three times with a new seed. A clip that fails any gate is dropped and logged, never patched.

## Uses

**What is it for?**
Training and evaluating detectors that find delivery flaws in time against a reference, and checking that a score falls as injected severity rises.

**What should it not be used for?**
- Judging real people's speaking ability or making decisions about them. The flaws are synthetic and the baselines are one reading each.
- Accent or dialect scoring. Nothing in OCDD labels accent, and accent is not a flaw.

## Distribution

The dataset is published on the Hugging Face Hub as `oratio-ocdd`, and the detector as `oratio-models`, under the account that ran NB03. Clips keep their source licence. CC BY-SA material keeps its share-alike terms and attribution through `evidence_url`. The generation code is MIT-licensed.

## Maintenance

Issues go to the GitHub repository. A new version gets a new `version` in `dataset.json` and a new Hub commit. Old commits stay available, so results can cite an exact revision.
