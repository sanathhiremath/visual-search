# Visual Search

Version 0.1. A working local image/text retrieval prototype using the actual pretrained OpenAI CLIP ViT-B/32 encoder. A small attributed Oxford-IIIT Pet photo catalog supports independent image queries and a fixed set of text queries, with explicit baselines and reproducible evidence.

## Run on Windows or Ubuntu

Python 3.11+ is required. Create an environment with `python -m venv .venv`.

Windows installation: `.venv\Scripts\python -m pip install -e ".[dev]"`.

Ubuntu installation: `.venv/bin/python -m pip install -e '.[dev]'`.

Activate the environment or use its Python for these commands:

```text
python run.py prepare
python run.py demo
python run.py query --text "a photo of a fluffy white dog"
python -m pytest -q
```

`prepare` explicitly downloads 40 attributed photographs and the 354 MB original checkpoint. It verifies the checkpoint SHA256 before use. Photos and their per-image attribution are retained; model weights and the local numeric index stay in `.cache/` and are ignored by Git.

`demo` is offline after preparation. It encodes a 32-image catalog, builds a normalized exact-cosine index, evaluates 16 queries, exercises an index insertion/upsert/removal cycle, and writes `artifacts/demo.json`. The arbitrary-text `query` command performs genuine model inference against the saved index; it does not look up a precomputed query by string.

The portfolio uses recorded query results and attributed image copies. It does not download or execute CLIP in the browser.

## Data and evaluation

Eight breeds are fixed before inference. Four images per breed come from the dataset's train split; one separate image per breed comes from the test split. Query identities and file hashes are checked against the catalog before evaluation. The original encoder's pretraining overlap with Oxford Pets is unknown, so this is not a claim of model-training independence.

The 16 queries comprise eight breed-naming text prompts and eight held-out image queries. Relevance means shared dataset breed label. That is a narrow proxy, not independently collected human semantic-search judgments.

Text baseline: normalized token overlap with catalog breed metadata. This privileged-label baseline is expected to be strong for prompts naming breeds. Image baseline: cosine similarity between 48-dimensional RGB histograms computed from actual pixels.

## Recorded results

| Query type | CLIP precision at 5 | Baseline precision at 5 | CLIP mean encoding/search time |
| --- | ---: | ---: | ---: |
| Text, 8 queries | 0.775 | 0.800, metadata | 59.24 ms |
| Image, 8 queries | 0.650 | 0.150, color histogram | 91.22 ms |
| Combined, 16 queries | 0.7125 | 0.4750, mixed baselines | 75.23 ms |

There are four relevant catalog photos per query, so the maximum precision at 5 is 0.8. CLIP improves the image baseline here, while straightforward metadata matching is better for these exact breed-naming text queries. Do not interpret the mixed-baseline aggregate as universal semantic-model superiority.

All reported times are local, warmed, single-query CPU measurements with two PyTorch threads. They include query encoding and exact index search but exclude initialization, network, downloads, HTTP serving, and concurrent load. Actual run values are in the JSON artifact. Index size and version/content hash are measured from the saved index.

## Architecture and update semantics

`prepare.py` downloads a frozen, attributed photo manifest and verifies model weights. `encoder.py` owns genuine CLIP preprocessing/encoding and simple baselines. `index.py` validates normalized finite embeddings, provides deterministic cosine rankings, rejects duplicate identities, and persists without Python pickle. `metrics.py` separates retrieval scoring and split checks from model execution. `demo.py` runs the experiment and records provenance and limitations.

The update demonstration runs only after evaluation: insert a held-out query image, observe its new top match, repeat the same upsert without growing the index, and remove it. Counts are 32 → 33 → 33 → 32, with the original content hash restored. The inserted image does not contribute to retrieval metrics. This is local exact-search index behavior, not distributed ingestion or a consistency guarantee across services.

## Scope and next steps

This is a fixed-taxonomy research demonstration, not a production search service. Cosine scores are not confidence probabilities. Out-of-domain text still returns the nearest pet photographs because no calibrated abstention boundary is implemented. Breed judgments do not cover attributes, broad semantics, or open-domain catalog relevance.

Next: independently judged query sets, richer catalog updates and deletions, approximate-index comparison at larger scale, reranking, duplicate/near-duplicate analysis, calibrated out-of-domain handling, and measured HTTP/concurrent serving. No reranker or human relevance study is claimed in this version.

See [ATTRIBUTION.md](ATTRIBUTION.md) for dataset/photo and model rights. The adapted photo copies remain under CC BY-SA 4.0 with the original owners' rights preserved; that does not relicense the whole portfolio codebase.

## Verification

Twelve tests cover zero/nonfinite embedding rejection, ranking ties, dimension checks, idempotent updates, hash restoration, serialization stability, relevance metrics, split leakage, real pixel histogram computation, lexical matching, and the actual generated artifact contract. Model-download and full-inference work are explicit CLI steps rather than hidden test network dependencies.

## Repository

Source: https://github.com/sanathhiremath/visual-search


This repository contains a local working prototype. Reproduction commands, measured scope, and limitations are documented above.
