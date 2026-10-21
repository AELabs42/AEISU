# Contributing

AEISU is maintained by Charles Dale B. Gleason. Small, focused pull requests and reproducibility reports are welcome.

Before submitting a change, run `python -m unittest discover -s tests -v`. Keep portable PyTorch code independent
of the custom Windows/CUDA runtimes. Do not add automatic dataset or model downloads without an explicit opt-in.

For bugs, include the commit, operating system, Python/PyTorch versions, checkpoint, exact command, and a minimal
reproducer. For GPU problems, include the GPU and driver/toolkit versions. Remove personal paths and credentials
from logs. Do not upload restricted benchmark frames or private game captures; use synthetic or redistributable images.

For performance changes, report resolution, precision, implementation path, warmup, synchronization, timing
interval, and output-parity checks. Separate resident-kernel latency from capture, transfers, and encoding.
For quality changes, preserve input/target pairing and metric definitions and show unfavorable outcomes too.

Please open an issue before a large refactor or new dependency. Author-owned contributions are accepted under
the repository's MIT license; third-party materials need their original license and attribution.
