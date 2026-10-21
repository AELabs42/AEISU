# Research records

quality/ holds the frozen 600-pair QRISP selection, thirteen methods, five conditions, per-image metrics,
input/target/output hashes, original protocols and statistical results. JSONL score files are gzip-compressed.
Nearest-neighbor identities in the original selection record are historical preparation metadata;
nearest-neighbor degradation is not one of the five evaluated conditions.

    python research/reproduce_tables.py --output outputs/recomputed_tables

This CPU-only command recomputes every condition/stratum/method aggregate and checks it against summary.json.
It does not rerun inference or recompute LPIPS from images. Metrics use the original float outputs, before PNG rounding.
The five environments have equal weight. Similarity strata describe the frozen source-pair sample, not model-based selection.

    python research/reproduce_statistics.py --output outputs/recomputed_statistics --draws 50000

This runs the archived paired scene/sequence bootstrap and multiplicity analysis in a new copy of the score archive.
Only compressed-file I/O is adapted; estimators and random seeds are retained. Packaged record hashes differ from original
record hashes because paths were redacted and files compressed; both identities are recorded in the source manifest.

studies/ retains core and output-head controls, training-degradation results, isolated-edge profiles, saved-output blur controls,
and conventional-CUDA precision evaluation. The old output-head protocol is separate from the expanded QRISP evaluation.
The core, clamp, and game-capture Bicubic controls now use the matched fixed-stride training campaign.
div2k_classic/ records the 1,000-epoch MSE control; div2k_l1_3000/ records the 3,000-epoch L1 model;
div2k_l1_3000_mixed/ records its separate calibrated FP8/FP16 artifact and quality evaluation.
runtime/ records the matched 6 October timing campaign, trial distributions, implementation boundaries and instruction audit.
Small release smoke-test timings are not substituted for those measurements.

archival_sources/ contains the experiment and figure-generation sources. These are provenance records, not portable entry points:
private path prefixes are redacted and some original scripts require separately obtained comparator models or archived image outputs.
The portable model metrics are implemented in aeisu/metrics.py. The release does not bundle comparator weights or QRISP images.
Input and output hashes allow matching independently obtained files without redistributing the image corpus.

release/source_manifest.json distinguishes exact copies from redacted/compressed files and records both SHA-256 digests.
Numerical score values are not changed by path redaction. Local source paths are symbolic, not claims that the files are bundled.
The original historical scene-level acquisition record remains incomplete.
