# Cached evaluation runner

Validation and planning work in the main environment, without torch or Transformers:

```sh
.venv/bin/python experiments/evaluate_models.py --dataset /Users/dev/Datasets/eden-vision-eval --validate-only
.venv/bin/python experiments/evaluate_models.py --dataset /Users/dev/Datasets/eden-vision-eval --model all --taxonomy all --dry-run
```

After a person has completed and validated canonical Ground Truth, run:

```sh
/Users/dev/.venvs/eden-vision-ai-model-eval/bin/python experiments/evaluate_models.py --dataset /Users/dev/Datasets/eden-vision-eval --model all --taxonomy all
```

Individual selections are `--model clip|siglip2` and `--taxonomy eden-v1|eden-v2`.
The default taxonomy remains eden-v1. An empty dataset returns without AI imports,
model loading, or report writes, including in all/all mode.

## Ground Truth and candidates

`dataset.json` declares the **annotation taxonomy**. The unchanged manifest validator
checks all Ground Truth against that version. Selecting a different candidate taxonomy
does not relabel, discard, or promote annotations. Both models see identical images,
prompts and candidate order. Candidates follow the taxonomy resource's list order;
prompts use `a photo of a {display_name}` with lowercased display names.

For a matrix involving eden-v2, hierarchy metrics use the shared eden-v2 relationship
taxonomy. The runner verifies that parent and broad-category relationships from every
participating taxonomy agree with this superset. For v1-only data and evaluation, the
reference is eden-v1. Reports name all three roles explicitly. A v2 Ground Truth concept
absent from v1 candidates remains in the evaluation and is listed in
`truth_concepts_outside_candidates`; it cannot receive exact credit from v1.

Existing metric definitions are unchanged: single-label exact/top-k and hierarchy
metrics, multi-label precision/recall, broad-category hits, and diagnostic coherence
retain their existing denominators. Undefined ratios remain null. UNKNOWN is annotation
status only; no UNKNOWN detector or calibrated threshold is evaluated. A v1 UNKNOWN
annotation is not automatically relabeled for v2; human review is needed if its meaning
changes with taxonomy expansion.

## Runtime and reproducibility

The evaluation-only runtimes require native Apple Silicon MPS, pinned local model
snapshots in the existing Hugging Face cache, and the model-eval environment. They do
not download missing checkpoints or enable CPU fallback. CLIP is closed and its MPS
cache released before SigLIP2 loads. Within each model, taxonomies run sequentially.

One normalized text embedding cache is held in process memory. Its identity includes
model ID/revision, taxonomy version, fixed prompt template, ordered concept IDs and
actual prompt strings. A changed identity replaces the cache; a repeated identity
reuses it. Requests run only the image encoder and model-specific matching formula:
CLIP uses normalized projected features and exp(logit_scale); SigLIP2 also adds logit_bias.
There is no persistent embedding cache, production adapter, or confidence policy.

Each image runs once. Latency includes file read/hash/decode, image processing, device
transfers, image encoding, matching, and ranking; load/text preparation are separate.
The first request after preparation is flagged, and component timings are preserved.
This is not the earlier warm synthetic benchmark. MPS timing is synchronized. Memory
is observational RSS/MPS allocated/driver memory, not a peak measurement.

## Reports

The default output is the dataset's `reports/` directory:

- `clip-eden-v1-report.json`
- `clip-eden-v2-report.json`
- `siglip2-eden-v1-report.json`
- `siglip2-eden-v2-report.json`
- `model-taxonomy-comparison.json`

Each report records revisions, environment, cache identity, manifest/metadata/image
SHA-256 provenance, metrics, timing/memory and per-image raw rankings. Images are
processed in stable ID order and their hashes are checked again before inference.
The comparison includes metrics, latency and memory per condition, without a winner
or cross-model raw-score magnitude comparison. Raw logits are not confidence.

Existing reports cause failure before model loading. Use a new explicit `--output-dir`
or opt into replacement with `--overwrite`. If a later model fails, earlier completed
condition reports remain; no incomplete comparison is produced. Baseline experiment
scripts and baseline JSON files are not changed by this runner.
