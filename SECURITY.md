# Reporting security issues

Use GitHub's private vulnerability reporting for this repository when available, rather than posting exploit details
or credentials in a public issue. For ordinary correctness bugs, use the issue tracker.

Load checkpoints only from trusted sources. The supplied model loader uses PyTorch's restricted `weights_only`
loading path; that is not a guarantee that arbitrary third-party files are safe. Do not upload private images,
credentials, or unredacted local logs with reports.

The native desktop runtime captures windows and uses GPU/graphics APIs. Use it only with applications you are
authorized to capture and do not bypass anti-cheat or application protections. This research release is not a
safety-critical image-processing system.
