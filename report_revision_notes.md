# Report revision audit

The report remains in English, following the user's explicit instruction. The available assignment PDF is `notes/S06_Seminar_DIFF.pdf`; no `S06_Seminar_DIFF(1).pdf` was found in the project or Downloads search. Its Vietnamese-language instruction is therefore an intentional exception, not silently claimed as satisfied.

## Closure of the previous review

| Review item | Resolution |
|---|---|
| 1. Language | English retained by explicit user override. |
| 2. Length | 10 A4 body pages; references start on page 11; appendix starts on page 12. |
| 3. Empty Abstract/Conclusion | Both completed in a student academic tone consistent with the Introduction. |
| 4. Symbol definitions | Replaced placeholders; defined distributions, dimensions, units, norm and expectation. |
| 5. AI disclosure | Removed from the final report at the user's explicit request; reproducibility information retained. |
| 6. Attribution | Added local equation/paper citations, dataset references, figure authorship, scripts and evidence provenance. |
| 7. ELBO | Restored the timestep-dependent weight and explained deliberate reweighting. |
| 8. Scaling | Distinguished endpoint mapping from empirical centering and variance. |
| 9. Approximation claims | Corrected reverse-Gaussian and finite-terminal-noise statements; retained residual signal. |
| 10. Single-token attention | Derived the one-key softmax limitation and removed spatial-routing claims. |
| 11. Conditioning architectures | Both models correctly described as additive conditioning plus single-token attention. |
| 12. Shared noise | Described stochastic samples and differing later noise; no causal or deterministic claim. |
| 13. Comparable conditions | Disclosed different learning rates, channel counts, parameters and cached data. |
| 14. Training costs | Renamed epoch-time sums; excluded untimed preprocessing from end-to-end claims. |
| 15. Memory | Removed the unsupported VRAM comparison from table/chart; explained legacy fields in appendix. |
| 16. Extrapolation | Removed transfer of measured ratios and the claim that VAE costs do not scale. |
| 17. Loss interpretation | Removed entropy and variance explanations; corrected background value and target distribution. |
| 18. Literature column | Removed mixed-setting column; supplied a source-specific matched inpainting comparison in Background. |
| 19. FID variant | Identified guided LDM-4-G, guidance scale, steps and unguided FID; prohibited cross-dataset ranking. |
| 20. Ethics | Corrected LAION-400M attribution; removed unsupported demographic/audit statements; added extraction research and license source. |
| 21. Algorithms | Added complete training and ancestral-sampling steps, final-step behavior, clipping and index conventions. |
| 22. Reconstruction/generation | Separated encoding/reconstruction from decoding a generated latent. |
| 23. VAE implementation | Added coordinate-mean MSE/KL formula, deterministic means, scaling, clipping and frozen-stage details. |
| 24. Attention dimensions | Used consistent row-oriented matrix products and defined softmax axis/head width. |
| 25. Reconstruction evaluation | Identified the single test image; removed dataset-level quality and bottleneck claims. |
| 26. FLOPs | Defined counted MAC/FLOP scope and exclusions; made latency explanations hypotheses. |
| 27. Figure sequences | Explained coupled forward marginals; corrected reverse-state labels; separated generation from reconstruction. |
| 28. Worked example | Preserved verified arithmetic, specified zero-based coordinates/index mapping and sum-versus-mean reduction. |
| 29. Quality/diversity | Restricted conclusions to visible examples; no unmeasured distributional quality claims. |
| 30. Pipeline readability/history | Redrew readable vector pipeline with explicit iteration and historically accurate encoder discussion. |
| 31. Conditioning modalities | Separated token cross-attention from spatial concatenation; removed unsourced depth attribution. |
| 32. Paper synthesis | Condensed repetition; distinguished earlier diffusion framework, DDPM contribution and latent compression trade-off. |
| 33. Background sources | Added PixelRNN and a primary GAN optimization reference; qualified comparisons. |
| 34. Prose/flow | Edited grammar, punctuation and speculative language; maintained existing major-section structure. |
| 35. Metadata | Replaced placeholder author names in document metadata. |
| 36. Obsolete drafts | Converted duplicate Method/figure sources into compatibility entry points. |
| 37. Units/labels | Clarified epoch-time reduction versus throughput, MiB in raw records, and caption scope. |

## Verification

- `python tools/build_report.py`: Typst 0.15.2, no compiler warnings, 12 total pages (10 body + 1 references + 1 appendix).
- All rendered pages inspected; chart labels and captions enlarged; no out-of-page text detected.
- All six figures and three tables checked against the revised text. Figures 3-5 use losslessly extracted saved image panels with provenance hashes, not newly generated model output.
- Saved JSON experiment records and checkpoints remain unchanged. Experimental images retain the same sample content; labels/layout have been redrawn.
- Python syntax checks performed for edited scripts. Numerical schedule coefficients, example arithmetic, ratios and PSNR independently checked against saved values. No retraining or checkpoint inference was run during this revision.
- Evaluation-only latent sampling now restores recorded clipping, preventing the previous hardcoded-range substitution. Historical measurements remain historical; they were not silently replaced.

## Remaining constraints

The exact `(1)` assignment PDF is unavailable, so comparison used the supplied assignment version already in `notes/`. English is retained as explicitly requested despite that version's Vietnamese instruction. Unrecorded original dependency versions cannot be reconstructed; the reproducibility appendix discloses this. The AI-tool disclosure and its bibliography entry were removed at the user's request. The existing runs cannot establish matched latency/VRAM, distributional quality, or general text-prompt alignment; the report now states those limits rather than presenting unsupported findings. Slides, presentation and Q&A are outside the report artifact.
