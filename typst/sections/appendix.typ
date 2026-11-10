#set heading(numbering: none)
#set text(size: 10pt)
#set par(leading: 0.5em, spacing: 0.6em)

#heading(numbering: none)[Appendix A. Reproducibility]

== Experimental Records and Configuration

The sources are in `experiments/src/`, and executable entry points are in `experiments/Scripts/`. Run commands from `experiments/` in a Python environment with PyTorch, torchvision, NumPy, Pillow, and Matplotlib. Saved logs identify PyTorch `2.11.0+cu130`, CUDA 13.0, and NVIDIA RTX PRO 6000 Blackwell Server Edition. Versions of the other original dependencies were not recorded. Exact numerical repeatability across software versions and hardware is not guaranteed.

```text
python Scripts/step0_worked_example.py --device cpu
python Scripts/step1_forward_noise.py
python Scripts/step2_train_ddpm.py --full --epochs 50 --batch_size 64 --device cuda
python Scripts/step3_train_ldm.py --full --vae_epochs 30 --ldm_epochs 50 --batch_size 64 --device cuda
python Scripts/step4_compare_benchmark.py
```

These commands can overwrite generated outputs; preserve the submitted records before rerunning. Training uses the MNIST training split; the worked example and reconstruction use the first test image labeled 3, with zero-based selected pixel $(10,14)$. With `drop_last=True`, each training epoch contains 937 batches of 64. During pre-encoding, one shuffled pass likewise retains 59,968 encoder means. The 59,968 cached means form exactly 937 batches. Their retained subset is fixed after encoding, whereas pixel training reshuffles the full dataset each epoch and can omit different final samples.

The VAE minimizes coordinate-averaged reconstruction MSE plus $0.0005$ times coordinate-averaged KL. The recorded latent mean is 0.024823 and global standard deviation is 0.861757; means are multiplied by 1.160420 without subtracting that global mean. Latent clean estimates are clipped to $[-8.781976,8.781976]$ in the saved training-run sampler, then divided by the scale before decoding. The evaluation-only branch has been corrected to restore the clipping range from the saved latent run record rather than substituting a different constant.

Saved evidence is in `experiments/outputs/tables/`: `step0_worked_example.json`, `ddpm_pixel_stats.json`, `vae_stats.json`, and `ldm_latent_stats.json`. Checkpoints are in `experiments/outputs/checkpoints/`. The pixel run timestamp is 2026-11-08 17:39:41 UTC; the latent record is 2026-11-10 07:31:01 UTC. These are separate runs, not simultaneous measurements. Raw records and checkpoints are retained unchanged by the report revision.

== Measurement Definitions and Figure Provenance

Training times sum synchronized epoch intervals. They exclude model initialization, dataset download, latent-cache construction, and checkpoint writing. Pixel sampling times include intermediate snapshots copied to the CPU; latent sampling includes one VAE decode and does not request equivalent snapshots. Each reported trajectory has batch size one and uses 1,000 steps. These timings are descriptive observations with differing instrumentation; they are not isolated, repeated GPU microbenchmarks.

The arithmetic estimator hooks convolution and linear modules, counts MACs at batch size one, and reports two FLOPs per MAC. It omits other operators and the autoencoder. Memory values in the raw logs use MiB ($2^20$ bytes) and represent maximum allocated memory since a broad reset, not standardized training or inference peaks. They are preserved as historical fields but excluded from the report's comparative claims.

Figure 1 is an authors' conceptual diagram based on DDPM. Figure 2 is an authors' redrawn KL-regularized LDM schematic. Figures 3-5 retain the saved experimental image content. Figures 3-5 redraw readable labels around losslessly extracted, previously plotted image panels. A provenance manifest in typst/figures/data records source PDFs and image hashes. Figure 3 corrects reverse snapshot labels using the implementation's post-transition state convention; it does not regenerate a new trajectory. Figure 4 displays the channel mean of the latent, normalized separately to $[0,1]$. Figure 5 retains shared initial noise and independently drawn later reverse noise. Figure 6 and the benchmark table use the saved JSON values, with ambiguous memory results excluded. Dataset attribution is to MNIST @mnist.

== Report Build

From the repository root, `python tools/build_report.py` compiles the diagrams and `typst/main.typ`, writes `output/pdf/diffusion_models_report.pdf`, and renders page previews for inspection. The build requires the Python Typst compiler package and PyMuPDF. The source `typst/method.typ` is a compatibility entry point forwarding to the authoritative `typst/sections/method.typ`; it is not a separate draft.
