# Oratio

Oratio scores how a speech is delivered. It compares a recording against a strong reading of the same text, marks the exact seconds where pacing, pauses, pitch, energy, emphasis, fluency or clarity fall short, and explains each mark with the measurement behind it.

Built for Track C of the Multimodal AI Hackathon 2026: Contrastive Speech Analytics and Temporal Flaw Grounding.

**Status:** under construction. The repository currently holds the project skeleton, the pinned-model build, and the deployment spike. Links to the live demo, the dataset and the models will appear here when they exist.

## Layout

| Path | Contents |
|---|---|
| `packages/speechgrade` | Core library: alignment, features, grounding, scoring. No PyTorch. |
| `services/api` | FastAPI service, shipped as one container image on AWS Lambda. |
| `infra` | AWS SAM templates and the cost budget. |
| `artifacts.lock.json` | Every served model file, pinned by Hugging Face revision and SHA-256. |

## Run it locally

You need Python 3.12, [uv](https://docs.astral.sh/uv/) and Docker.

```bash
uv sync --frozen
uv run make verify     # lint, type check, tests
uv run make dev        # API on http://localhost:8000/api/health
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
