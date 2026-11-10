"""Compile report sources and render all pages; never rerun training."""
from pathlib import Path
import json
import sys

from report_runtime import get_typst
from rebuild_data_figures import prepare as prepare_data_figures

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "Scripts"))
from step4_compare_benchmark import generate_assets


def main():
    import pymupdf

    compiler = get_typst()
    prepare_data_figures()
    generate_assets()
    targets = [
        ("figures/fig1_ddpm_chain.typ", "typst/figures/fig1_ddpm_chain.pdf"),
        ("typst/figures/fig2_ldm_pipeline.typ", "typst/figures/fig2_ldm_pipeline.pdf"),
        ("typst/figures/fig_speed_flops_comparison.typ", "typst/figures/fig_speed_flops_comparison.pdf"),
        ("typst/figures/fig_exp_pixel_forward_reverse.typ", "typst/figures/fig_exp_pixel_forward_reverse.pdf"),
        ("typst/figures/fig_exp_ldm_latent_and_noise.typ", "typst/figures/fig_exp_ldm_latent_and_noise.pdf"),
        ("typst/figures/fig_exp_ldm_cross_attn_conditioning.typ", "typst/figures/fig_exp_ldm_cross_attn_conditioning.pdf"),
        ("typst/main.typ", "output/pdf/diffusion_models_report.pdf"),
    ]
    for source, destination in targets:
        result, warnings = compiler.compile_with_warnings(str(ROOT / source), root=str(ROOT))
        if warnings:
            raise RuntimeError(f"{source}: compiler warnings: {warnings}")
        target = ROOT / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(result)
        if destination.startswith("typst/figures/"):
            (ROOT / "figures" / target.name).write_bytes(result)
            document = pymupdf.open(stream=result, filetype="pdf")
            document[0].get_pixmap(matrix=pymupdf.Matrix(2, 2)).save(target.with_suffix(".png"))
            png = target.with_suffix(".png").read_bytes()
            (ROOT / "figures" / target.with_suffix(".png").name).write_bytes(png)
            experiment_mirror = ROOT / "experiments" / "outputs" / "figures" / target.name
            if experiment_mirror.exists():
                experiment_mirror.write_bytes(result)
                experiment_mirror.with_suffix(".png").write_bytes(png)
            if target.stem == "fig_exp_ldm_latent_and_noise":
                for base in [ROOT / "figures", ROOT / "experiments" / "outputs" / "figures"]:
                    (base / "fig_exp_ldm_latent_and_cond.pdf").write_bytes(result)
                    (base / "fig_exp_ldm_latent_and_cond.png").write_bytes(png)
        print(f"Compiled {source} -> {destination}")

    pdf = pymupdf.open(ROOT / targets[-1][1])
    text_pages = [page.get_text() for page in pdf]
    references_page = next(i for i, text in enumerate(text_pages) if text.splitlines()[2:3] == ["References"] or "\nReferences\n" in text)
    appendix_page = next((i for i, text in enumerate(text_pages) if "Appendix A." in text), None)
    if not 6 <= references_page <= 10:
        raise AssertionError(f"Body is {references_page} pages; expected 6-10")
    body_text = "\n".join(text_pages[:references_page])
    for heading in ["Abstract", "Introduction", "Background and Paper Review", "Method and Worked Example", "Results and Discussion", "Conclusion"]:
        assert heading in body_text, heading
    for phrase in ["Author Name", "[Define", "complete information erasure", "entropy-packing", "deterministically dictates", "Cluster vs."]:
        assert phrase not in "\n".join(text_pages), phrase
    previews = ROOT / "tmp" / "pdfs" / "report-preview"
    previews.mkdir(parents=True, exist_ok=True)
    outside = []
    for index, page in enumerate(pdf):
        page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save(previews / f"page-{index + 1:02}.png")
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    x0, y0, x1, y1 = span["bbox"]
                    if x0 < -1 or y0 < -1 or x1 > page.rect.width + 1 or y1 > page.rect.height + 1:
                        outside.append((index + 1, span["text"]))
    assert not outside, outside
    summary = {
        "compiler_version": compiler.__version__,
        "total_pages": len(pdf),
        "body_pages": references_page,
        "references_start_page": references_page + 1,
        "appendix_start_page": appendix_page + 1 if appendix_page is not None else None,
        "compiler_warnings": [],
        "text_outside_page": outside,
    }
    (previews / "build_checks.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
