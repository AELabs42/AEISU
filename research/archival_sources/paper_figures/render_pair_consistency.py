"""Render source-pair diagnostics and a transparent stratified benchmark report."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_bench"))
from analyze_rendered_pairs import OUTPUT, arrays, frozen_metrics, image_id, source_pairs
from pair_consistency import area_float
from luma_metrics import y_channel


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def render(ranking: dict, analysis: dict) -> None:
    colors = {"most_similar": "#2776ad", "middle": "#949494", "least_similar": "#ca642d"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, dataset in zip(axes, ("qrisp", "gameir")):
        scores = frozen_metrics(dataset)["native"]
        for group, color in colors.items():
            rows = [row for row in ranking["rankings"][dataset]["primary"] if row["group"] == group]
            x = [row["pair_y_psnr_db"] for row in rows]
            y = [scores[row["image"]]["aif_mixed"]["y_psnr_shave2_db"]
                 - scores[row["image"]]["fsr1_easu_fp32"]["y_psnr_shave2_db"] for row in rows]
            ax.scatter(x, y, color=color, label=group.replace("_", " "), s=22, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(dataset.upper())
        ax.set_xlabel("LR vs area-reduced HR agreement (Y PSNR, dB)\nHigher = more consistent source pair")
        ax.set_ylabel("Model minus EASU target Y PSNR (dB)")
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8)
    fig.savefig(OUTPUT / "pair_consistency_vs_model_gap.png", dpi=180)
    fig.savefig(OUTPUT / "pair_consistency_vs_model_gap.svg")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
    for index, dataset in enumerate(("qrisp", "gameir")):
        for column, method in enumerate(("aif_mixed", "fsr1_easu_fp32")):
            ax = axes[index, column]
            for group in ("most_similar", "middle", "least_similar"):
                points = analysis["datasets"][dataset]["blur"]["primary"][group][method]
                x = [float(key) for key in points]
                y = [point["psnr_gain_db"] for point in points.values()]
                ax.plot(x, y, marker="o", color=colors[group], label=group.replace("_", " "))
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(f"{dataset.upper()} / {'Model' if method == 'aif_mixed' else 'EASU'}")
            ax.set_xlabel("Gaussian sigma (HR pixels)")
            ax.set_ylabel("Change in target Y PSNR (dB)")
            ax.grid(alpha=0.15)
            ax.legend(fontsize=8)
    fig.savefig(OUTPUT / "blur_sensitivity.png", dpi=180)
    fig.savefig(OUTPUT / "blur_sensitivity.svg")
    plt.close(fig)

    pairs = source_pairs()
    for dataset, selected in pairs.items():
        lookup = {image_id(pair, dataset): pair for pair in selected}
        ordered = ranking["rankings"][dataset]["primary"]
        examples = ordered[:2] + ordered[-2:]
        fig, axes = plt.subplots(4, 3, figsize=(12, 10), constrained_layout=True)
        for index, row in enumerate(examples):
            low, high = arrays(dataset, lookup[row["image"]])
            reduced = area_float(high)
            error = np.abs(y_channel(low) - y_channel(reduced))
            axes[index, 0].imshow(low)
            axes[index, 1].imshow(reduced)
            heatmap = axes[index, 2].imshow(error, cmap="magma", vmin=0, vmax=0.05)
            axes[index, 0].set_title(f"{row['group'].replace('_', ' ')}: {row['image']}\nRendered LR")
            axes[index, 1].set_title(f"Area-reduced HR / pair PSNR {row['pair_y_psnr_db']:.2f} dB")
            axes[index, 2].set_title("Absolute Y difference (fixed 0–0.05 scale)")
            for ax in axes[index]:
                ax.axis("off")
        fig.colorbar(heatmap, ax=axes[:, 2], shrink=0.65)
        fig.savefig(OUTPUT / f"{dataset}_pair_examples.png", dpi=160)
        plt.close(fig)


def report(ranking: dict, analysis: dict) -> None:
    lines = ["# Source-pair consistency and blur diagnostic", "",
             "Exploratory analysis of the existing 60 QRISP and 120 GameIR benchmark frames. "
             "No retraining, inference changes, manuscript edits, or replacement of the full benchmark.", "",
             "## Selection protocol", "",
             "Rank independently within each dataset by encoded-Y MSE between supplied rendered LR and "
             "float64 aligned 2x2 area-reduced HR, with a four-LR-pixel border crop. The lowest-error quarter "
             "is most similar, the middle half is retained, and the highest-error quarter is least similar. "
             "Ranking uses only input/target images and was saved before reading method scores. "
             "A second ranking smooths both LR-grid images with sigma1 before scoring.", "",
             "This measures sampling consistency, not specifically asymmetric effects. Scene complexity, "
             "aliasing, alignment, and brightness can also affect the ranking. Subset PSNR differences "
             "are not controlled causal estimates. These subsets were not a predeclared primary benchmark.", "",
             "## Native/supplied-render benchmark", "",
             "| Dataset | Group | Frames | Model Y PSNR | EASU Y PSNR | Model minus EASU | Model Y MSE x1000 | EASU Y MSE x1000 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for dataset, data in analysis["datasets"].items():
        for group in ("all", "most_similar", "middle", "least_similar"):
            values = data["benchmarks"]["primary"][group]["native"]
            ours, easu = values["aif_mixed"], values["fsr1_easu_fp32"]
            a, b = ours["y_psnr_shave2_db"], easu["y_psnr_shave2_db"]
            lines.append(f"| {dataset.upper()} | {group} | {ours['images']} | {a:.4f} | {b:.4f} | {a-b:+.4f} | "
                         f"{ours['y_mse_shave2']*1000:.5f} | {easu['y_mse_shave2']*1000:.5f} |")
    lines += ["", "All existing methods and SSAA/SMAA/FXAA/native conditions are reaggregated in the "
              "dataset subset-metrics JSON files. QRISP SSAA uses Enhanced targets, unlike Native targets "
              "in the other conditions; their absolute scores are not an isolated AA-strength comparison.", "",
              "## Controlled smoothing diagnostic", "",
              "Fixed Gaussian sigmas 0, 0.5, 1 and 2 HR pixels are applied to each saved native-input "
              "reconstruction. The target remains unchanged. Gaussian filtering of Y is algebraically "
              "equivalent to filtering RGB then converting to Y; this equivalence was independently checked. "
              "Metrics use the original two-HR-pixel crop and equal per-frame means.", "",
              "| Dataset | Group | Method | Gain sigma0.5 | Gain sigma1 | Gain sigma2 |",
              "|---|---|---|---:|---:|---:|"]
    for dataset, data in analysis["datasets"].items():
        for group in ("all", "most_similar", "least_similar"):
            for method, points in data["blur"]["primary"][group].items():
                gains = " | ".join(f"{points[s]['psnr_gain_db']:+.4f}" for s in ("0.5", "1.0", "2.0"))
                lines.append(f"| {dataset.upper()} | {group} | {method} | {gains} |")
    lines += ["", "## Interpretation", "",
              "QRISP supports the proposed association: the model has a small PSNR lead over EASU on "
              "the most-consistent quarter and a larger deficit on the least-consistent quarter. The "
              "sigma1 smoothing penalty becomes a gain on the least-consistent quarter. GameIR is weaker "
              "evidence: EASU still leads on the most-consistent quarter, and the difference in smoothing "
              "benefit between extreme groups is smaller. Mild smoothing benefits the model even on "
              "consistent pairs. Sigma2 hurts both methods in every extreme group: more blur is not "
              "monotonically rewarded.", "",
              "Therefore, these results support a QRISP-specific relationship between pair inconsistency "
              "and smoothing-sensitive PSNR, but do not prove that asymmetric effects are the only cause "
              "of EASU's advantage or that improved PSNR necessarily means worse perceptual quality. "
              "Visual inspection and scene-complexity controls are needed to distinguish the mechanisms.", "",
              "## Artifacts", "",
              "- ranking.json: both source-only rankings and frozen quartile identities.",
              "- sorted_pairs.md: every pair in primary rank order.",
              "- qrisp_subset_metrics.json and gameir_subset_metrics.json: all method/condition/group scores.",
              "- analysis.json: complete aggregate results; *_blur.json: per-image smoothing scores.",
              "- pair_consistency_vs_model_gap.png and blur_sensitivity.png: editable SVG companions.",
              "- *_pair_examples.png: first two and last two ranked source pairs, not model-output selection."]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lines = ["# Source-only pair ranking", "", "Lower pair MSE means more consistency; models do not determine this ordering."]
    for dataset in ("qrisp", "gameir"):
        lines += ["", f"## {dataset.upper()}", "", "| Rank | Pair | Group | Pair Y PSNR | LR Y variance |",
                  "|---:|---|---|---:|---:|"]
        for row in ranking["rankings"][dataset]["primary"]:
            lines.append(f"| {row['rank']} | {row['image']} | {row['group']} | {row['pair_y_psnr_db']:.4f} | {row['lr_y_variance']:.6f} |")
    (OUTPUT / "sorted_pairs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ranking, analysis = read(OUTPUT / "ranking.json"), read(OUTPUT / "analysis.json")
    render(ranking, analysis)
    report(ranking, analysis)
    print(f"Report and diagnostic figures: {OUTPUT}")


if __name__ == "__main__":
    main()
