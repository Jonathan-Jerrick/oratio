# Oratio

Oratio scores how a speech is delivered. It compares a recording against a strong reading of the same text, marks the exact seconds where pacing, pauses, pitch, energy, emphasis, fluency or clarity fall short, and explains each mark with the measurement behind it.

Built for Track C of the Multimodal AI Hackathon 2026: Contrastive Speech Analytics and Temporal Flaw Grounding.

**Live demo:** [oratio-steel.vercel.app](https://oratio-steel.vercel.app)

- **Explore.** Open an analyzed Gettysburg Address reading and see each injected flaw beside what Oratio detected.
- **Analyze.** Upload or record your own reading.
- **Method and data.** [How it works](docs/methodology.md) and the [dataset datasheet](docs/datasheet.md).

## Layout

| Path | Contents |
|---|---|
| `apps/web` | Dashboard: Vite, React 19, TypeScript, Tailwind. Deployed on Vercel. |
| `packages/speechgrade` | Core library: alignment, features, grounding, scoring. No PyTorch. |
| `services/api` | FastAPI service, shipped as one container image on AWS Lambda. |
| `dataset_factory` | Builds OCDD: source acquisition, flaw planning and injection, gates, training and evaluation. |
| `notebooks` | NB01 to NB03, run on Colab: baselines, dataset generation, detector training and publishing. |
| `infra` | AWS SAM templates and the cost budget. |
| `docs` | Methodology and dataset datasheet. |
| `artifacts.lock.json` | Every served model file, pinned by Hugging Face revision and SHA-256. |

## How it works

1. Both recordings are aligned to their words, and the words are paired by text.
2. Pitch, loudness, brightness, clarity, timing and pauses are measured for every word.
3. Each participant word is compared with its baseline word, in units of how much good readers vary on that feature.
4. Words that pass a rubric threshold are merged into time regions. Each region gets a code, the measurement behind it and a fix.
5. Every point lost traces back to one of those regions.

The full method is in [docs/methodology.md](docs/methodology.md).

## Run it locally

You need Python 3.12, [uv](https://docs.astral.sh/uv/) and Docker.

```bash
uv sync --frozen
uv run make verify     # lint, type check, tests
uv run make dev        # API on http://localhost:8000/api/health
```

Dashboard:

```bash
cd apps/web
npm ci
VITE_API_URL=http://localhost:8000 npm run dev
```

Build and run the same image that Lambda runs:

```bash
make image             # downloads the pinned models and checks their hashes
make image-run
```

## Reproducibility

The image build downloads only the files named in `artifacts.lock.json` and fails if any SHA-256 differs. Base images are pinned by digest and Python dependencies by `uv.lock`. Nothing is downloaded when the service starts.

## License

MIT. See [LICENSE](LICENSE).
