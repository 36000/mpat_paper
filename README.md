# Probabilistic mapping of brain connections without random sampling

Scripts to reproduce the figures of Kruper, Cieslak & Rokem, *Probabilistic mapping of brain
connections without random sampling* (MPAT: Markov probabilistic absorbing tractography).

MPAT itself is at [rokemlab/mpat](https://github.com/rokemlab/mpat). These scripts run it
(and pyAFQ, for comparison) on public data and make the figures.

> **Naming.** MPAT was developed as MCAT. The Python package is still imported as `mcat` and
> its command-line tool is `mcat`; MPAT writes `*_desc-mcat_*` file names. The figures label
> the method MPAT and pyAFQ as AFQ.

## Requirements

- Python ≥ 3.10
- [MPAT](https://github.com/rokemlab/mpat) (`pip install -e .` in a clone)
- [pyAFQ](https://github.com/tractometry/pyAFQ) 3.4, with its SynthSeg (ONNX) dependencies
- numpy, scipy, pandas, matplotlib, nibabel, scikit-image, Pillow
- Figure 1b (cortical endpoints) only: [neuropythy](https://github.com/noahbenson/neuropythy),
  k3d, selenium + Chrome/chromedriver, pyvirtualdisplay (for headless rendering), boto3 with
  HCP S3 credentials (an AWS profile named `hcp`)
- The large pyAFQ run in `fig1/seed_count_local.py` (5×10^7 seeds) runs on the CPU and takes
  over an hour; it needs disk space for the whole-brain tractogram (`TRX_TMPDIR`)

## Data

| Dataset | Used for | Access |
|---|---|---|
| HCP test–retest (44 subjects), HCP 1200 subject 627549 | Fig. 1 | [db.humanconnectome.org](https://db.humanconnectome.org); fetched with `AFQ.data.fetch.fetch_hcp` |
| IXI, IXI002-Guys-0828 | Fig. 2a | [brain-development.org/ixi-dataset](https://brain-development.org/ixi-dataset/) |
| Infant from Grotheer et al. 2022 | Fig. 2b | [figshare](https://figshare.com/ndownloader/files/38053692) |
| UCSF-PDGM, participant 0393 | Fig. 2c | The Cancer Imaging Archive |
| Healthy Brain Network, sub-NDARAA948VFH | Fig. 2d | [Child Mind Institute](https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/) |

## Layout

```
fig1/
  seed_count_local.py   pyAFQ whole-brain tractography of HCP 627549 (5e7 seeds), left CST recognized
  fig1b_mips.py         Fig. 1a: CST density, AFQ at 1e6 / 1e7 / 1e8 streamlines next to MPAT
  fig1c_sphere.py       Fig. 1b: cortical CST endpoints on the cortex, AFQ (1e8) next to MPAT
  run_hcp_mat_full.py   HCP test-retest: MPAT and pyAFQ on both sessions, fuzzy Dice per tract
  plot_fuzzydice.py     Fig. 1c-e: reliability bar plot, sampling fits and tests
  fig1_panels.py        assembles Figure 1
fig2/
  rgvp.py               the RGVP pipeline (eyes to the lateral geniculate region)
  mcat_rgvp_figure.py   Fig. 2d: the RGVP on an oblique slice
  mcat_ifof_tumor_figure.py  Fig. 2c: the IFOF through the tumor
  fig2_panels.py        assembles Figure 2
extras/
  fig1a_chain_t1.py     schematic of the chain on a T1 (not in the current figures)
```

Most scripts have constants at the top (subject IDs, paths, streamline counts) that need
setting for your machine; each script's docstring describes its options.

## Figure 1

### a, b: convergence of probabilistic tractography to MPAT (HCP 627549)

1. **MPAT**, with the corticospinal tract defined by SynthSeg endpoints only
   (`--synthseg-cst`; precentral, paracentral and postcentral cortex to the brainstem):

   ```bash
   mcat dwi.nii.gz dwi.bval dwi.bvec T1w.nii.gz hcp_out --n-proc 16 \
       --synthseg-cst --bundles "Left Corticospinal"
   ```

2. **pyAFQ**, one large run (5×10^7 random seeds, about 1.3×10^8 whole-brain streamlines),
   recognizing the same CST:

   ```bash
   python fig1/seed_count_local.py
   ```

   Then set `TOTAL_STREAMLINES` (in `fig1c_sphere.py`) and the counts in `PYAFQ_RUNS` (in
   `fig1b_mips.py`) to the run's whole-brain streamline count, and point both at its
   `*_desc-bundles_tractography.trx`.

3. **Panels.** Both scripts draw the same random subsets (`--rng-seed`):

   ```bash
   python fig1/fig1b_mips.py --bundle "Left Corticospinal" \
       --markov hcp_out/mcat/sub-627549_bundle-LeftCorticospinal_desc-density_map.nii.gz \
       --background sub-627549_desc-brain_mask.nii.gz --view coronal --out fig1b.png

   python fig1/fig1c_sphere.py 627549 \
       hcp_out/mcat/sub-627549_bundle-LeftCorticospinal_desc-endpoints_map.nii.gz \
       --hemi left --floor 1e-3 --elevation 25 --azimuth 0 -o fig1c
   ```

### c–e: test–retest reliability (44 HCP subjects)

```bash
python fig1/run_hcp_mat_full.py <subject IDs> --cst-fornix-hcc --pyafq
python fig1/run_hcp_mat_full.py --combine            # -> all_desc-fuzzydice.csv
python fig1/plot_fuzzydice.py all_desc-fuzzydice.csv -o fuzzydice
```

`plot_fuzzydice.py` prints the statistics reported in the text: MPAT vs AFQ per subject
(one-sided Wilcoxon signed-rank), and the sampling analysis (the per-tract fits
α + β/√N, the pooled fit a_t + b/√N, and a 5,000-resample subject bootstrap of the gap left
as N → ∞).

### Assembly

```bash
python fig1/fig1_panels.py fig1b.png fig1c.png fuzzydice.png -o fig1.png
```

## Figure 2

### a: IXI adult (1.5 T, 15 directions)

Preprocess with QSIPrep (26.1.1, defaults, 1.75 mm output), run MPAT on all bundles, then draw
the collage (SLF I–III left out):

```bash
mcat dwi.nii.gz dwi.bval dwi.bvec T1w.nii.gz ixi_out --n-proc 16
```

```python
import re
from mcat.collage import write_collage
from mcat.core import bundle_files

names = sorted(bundle_files("ixi_out/mcat"))
slf = re.compile(r"(SLF|SuperiorLongitudinal)_?(I{1,3}|[123])(?![a-z])", re.I)
write_collage("ixi_out/mcat", bundles=[n for n in names if not slf.search(n)],
              t1="<DWI-space image>", pad=0, fontsize=18)
```

### b: infant

MPAT with BabySeg tissue segmentation and BabyAFQ's bundle definitions, then the same
collage. *TODO: exact command.*

### c: tumor (UCSF-PDGM 0393)

MPAT with the enhancing tumor and edema/infiltration treated as white matter and the necrotic
core as CSF (never entered), then:

```bash
python fig2/mcat_ifof_tumor_figure.py tumor_out \
    --tumor sub-0393_space-diff_desc-tumor_label-1_mask.nii.gz
```

*TODO: exact MPAT command and tissue override.*

### d: retinogeniculate visual pathway (HBN)

```bash
python fig2/rgvp.py dwi.nii.gz dwi.bval dwi.bvec T1w.nii.gz rgvp_out \
    --n-proc 16 --eyes arc
python fig2/mcat_rgvp_figure.py rgvp_out/mcat-rgvp --t1 T1w.nii.gz -o rgvp
```

The T1 must be unmasked (the eyes are outside the brain mask). `--eyes arc` finds each eye by
fitting a sphere to the back wall of the globe (works on face-blurred or defaced images).

### Assembly

Edit the four image paths at the top of `fig2/fig2_panels.py`, then:

```bash
python fig2/fig2_panels.py
```

## Citation

*TODO: add once published.*
