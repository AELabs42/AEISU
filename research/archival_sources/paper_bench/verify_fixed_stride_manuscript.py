"""Check the fixed-stride manuscript tables against saved results, without Torch."""

import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "output/training_campaigns/fixed_stride_2026-10-16"
REVIEW = ROOT / "output/editorial_review/fixed_stride_results_2026-10-16"
CONDITIONS = ("ssaa_area", "bicubic_aa", "fxaa", "smaa", "native")
METRICS = ("y_psnr_shave2_db", "y_ssim_shave2", "lpips_alex_v01")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(source, label):
    for block in re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", source, re.S):
        if "\\label{" + label + "}" in block:
            return block
    raise ValueError(label)


def row(source, label, name):
    block = table(source, label)
    line = next(line for line in block.splitlines() if line.startswith(name + " &"))
    return [float(value) for value in re.findall(r"[+-]?(?:\d*\.\d+|\d+)", line.split("&", 1)[1])]


def assert_cells(actual, expected, places):
    assert len(actual) == len(expected) == len(places)
    for left, right, digits in zip(actual, expected, places):
        assert abs(left - right) <= 0.500001 * 10 ** -digits, (left, right, digits)
    return len(actual)


def main():
    core = load(CAMPAIGN / "evaluations/core/summary.json")
    clamp = load(CAMPAIGN / "evaluations/clamp/summary.json")
    protocol = load(CAMPAIGN / "evaluations/core/protocol.json")
    main_source = (ROOT / "docs/manuscript/inverse_supersampling.tex").read_text(encoding="utf-8")
    supplement = (ROOT / "docs/manuscript/inverse_supersampling_supplement.tex").read_text(encoding="utf-8")
    count = 0
    groups = ((core, "tab:core-quality", ("reference", "fixed_inputs", "fixed_base", "dense_core"),
               ("Reference", "Fixed inputs", "Fixed bilinear base", r"Dense $64\times2$ core")),
              (clamp, "tab:clamp-quality", ("clamped", "unclamped"), ("Soft clamp", "Identity")))
    for summary, label, models, names in groups:
        for model, name in zip(models, names):
            expected = [summary["means"]["fp16"][condition][model]["all"][metric]
                        for condition in CONDITIONS for metric in METRICS]
            count += assert_cells(row(supplement, label, name), expected, [3, 5, 5] * 5)
            mean = summary["means"]["fp8_stress"]["ssaa_area"][model]["all"]
            baseline = summary["means"]["fp16"]["ssaa_area"][model]["all"]
            expected = [mean[METRICS[0]]] + [mean[metric] - baseline[metric] for metric in METRICS]
            stress_label = "tab:core-quantization" if summary is core else "tab:clamp-quantization"
            count += assert_cells(row(supplement, stress_label, name), expected, [3, 5, 6, 6])
            if summary is core:
                expected = [protocol["models"][model]["training_mse"] * 1e4] + [baseline[metric] for metric in METRICS]
                count += assert_cells(row(main_source, "tab:core-controls", name), expected, [4, 3, 5, 5])
    assert all(model["gradient_phase"] == "fixed" for model in protocol["models"].values())
    for study in ("core", "clamp"):
        assert load(CAMPAIGN / f"evaluations/{study}/verification.json")["passed"]
    comparisons = {}
    for study, summary, models in (("core", core, ("fixed_inputs", "fixed_base", "dense_core")),
                                   ("clamp", clamp, ("unclamped",))):
        old = load(REVIEW / f"old_{study}_summary.json")
        comparisons[study] = {}
        for model in models:
            comparisons[study][model] = {
                condition: {metric: {"old": old["means"]["fp16"][condition][model]["all"][metric],
                                     "new": summary["means"]["fp16"][condition][model]["all"][metric],
                                     "new_minus_old": summary["means"]["fp16"][condition][model]["all"][metric]
                                     - old["means"]["fp16"][condition][model]["all"][metric]}
                            for metric in METRICS}
                for condition in CONDITIONS
            }
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for study in ("core", "clamp")
              for filename in ("summary.json", "protocol.json", "verification.json")
              for path in [CAMPAIGN / "evaluations" / study / filename]}
    result = {"passed": True, "numeric_table_cells_checked": count, "source_hashes": hashes,
              "old_new_fp16_comparison": comparisons, "gpu_work": False,
              "scope": "Rounded table cells and saved evaluation verification; not a new inference run"}
    (REVIEW / "numerical_checks.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {count} numeric cells in five updated tables; both output verifications passed.")


if __name__ == "__main__":
    main()
