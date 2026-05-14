# si110-reflection-holography

Reflection electron holography forward-simulation and reconstruction pipeline for the cleaved-Si[110] (1,1,-1) facet geometry, replicating the off-axis holographic step-phase measurement of Osakabe et al. (1993) and adapting it to the diamond-cubic structure factor of silicon.

> **Status: active research code.** Stage-1 of a larger PhD program on extending electron ptychography from transmission to reflection geometry (HF-3300 STEHM, CAMTEC, University of Victoria). Some scripts are validated; others have documented physics or implementation issues. See `docs/PIPELINE_LOG.md` for the script-by-script status table before relying on any output.

---

## 1. Scientific background

In off-axis reflection electron holography (Osakabe, Endo, Tonomura, Surface Science 1993), the height $h$ of a surface step is encoded in the phase difference $\Delta\varphi$ that the object beam acquires when its reflection point moves between two terraces. For a Bragg-reflected beam at angle $\theta_B$, the projected step-height formula reads

$$\Delta\varphi \;=\; \frac{4\pi}{\lambda}\,\sin\theta_B\,(\hat{\mathbf{g}}\cdot\hat{\mathbf{n}}_{\rm surf})\,h\;,$$

where $\lambda$ is the relativistic electron wavelength, $\hat{\mathbf{g}}$ the unit reciprocal-lattice vector of the active reflection, and $\hat{\mathbf{n}}_{\rm surf}$ the outward surface normal. The original Pt(111) experiment used the symmetric (666) condition for which $\hat{\mathbf{g}}\parallel\hat{\mathbf{n}}_{\rm surf}$, i.e. $\hat{\mathbf{g}}\cdot\hat{\mathbf{n}}_{\rm surf}=1$, so the formula reduces to $\Delta\varphi = (4\pi/\lambda)\sin\theta_B\,h$.

For cleaved silicon the symmetric form is not available: **Si(666) is structure-factor forbidden** in the diamond-cubic lattice ($F_{hkl}=0$ when $h+k+l \equiv 2\!\!\mod 4$ with mixed parity), so a *symmetric* specular reflection off a (1,1,-1) facet at the appropriate Bragg angle does not exist. The simulation therefore retargets to **Si(2,-2,0)**, an allowed asymmetric reflection accessible at $\theta_B\approx 6.53$ mrad at $200$ keV. Geometric projection of $\hat{\mathbf{g}}_{(2\bar{2}0)}$ onto $\hat{\mathbf{n}}_{(1\bar{1}1)}$ gives $\hat{\mathbf{g}}\cdot\hat{\mathbf{n}}_{\rm surf}=2/\sqrt{6}\approx 0.8165$, and one $d_{111}$ bilayer step ($a/\sqrt{3}\approx 3.136$ Å on Si) produces

$$\Delta\varphi_{(2\bar{2}0)} \;=\; \tfrac{4\pi}{\lambda}\sin\theta_B\,(2/\sqrt{6})\,d_{111} \;=\; 2\pi/3 \quad (\text{visible}).$$

By contrast, the geometric specular off the {111} facet — Si(1,1,-1) at the same crystal — yields $\Delta\varphi=2\pi\equiv 0$ (mod $2\pi$); the step is phase-invisible at that reflection, the same pathology that originally motivated Osakabe's choice of the *higher-order* (666) specular on Pt. The script names in `pipeline/` retain the historical `666` label for clarity with the literature, but the operative reflection on Si is Si(2,-2,0).

## 2. Pipeline architecture

```
  ┌──────────────────────────────────────────────┐
  │  sample_generators/si110_cleave_slab_         │
  │  generator.py                                 │
  │   Si[110] cleave-edge slab + meta.json        │
  │   (cleave-frame: z‖[110]beam, x‖[1,-1,1]n̂,    │
  │    y‖[1,-1,-2] step-normal-in-plane)          │
  └──────────────┬───────────────────────────────┘
                 │  .xyz (Prismatic 6-col) + meta.json
                 ▼
  ┌──────────────────────────────────────────────┐
  │  pipeline/multislice_forward_model.py         │
  │   Prismatique HRTEM wrapper. Single plane-    │
  │   wave run → complex exit wave + JSON dump    │
  └──────────────┬───────────────────────────────┘
                 │  hrtem_sim_wavefunction_*.h5
                 ▼
  ┌──────────────────────────────────────────────┐
  │  pipeline/multislice_tilt_series_runner.py    │
  │   Loops over beam tilts (angle list or        │
  │   start/stop/step). Tilt azimuth follows      │
  │   meta['tilt_azimuth_deg']                   │
  └──────────────┬───────────────────────────────┘
                 │  per-tilt dirs with .h5 + meta.json
                 ▼
  ┌──────────────────────────────────────────────┐
  │  pipeline/specular_666_filter.py              │
  │   Sideband isolation in k-space:              │
  │   FFT → notch (000) → mask peak (Hann/Hamming)│
  │   → recenter → IFFT → unwrap phase            │
  └──────────────┬───────────────────────────────┘
                 │  results_arrays.npz (phase_hann_flat,
                 │   phase_hamming_flat) per tilt
                 ▼
  ┌──────────────────────────────────────────────┐
  │  pipeline/step_height_reflection_formula.py   │
  │   h = Δφ·λ / (4π sin θ_B (ĝ·n̂_surf))          │
  └──────────────────────────────────────────────┘

  pipeline/process_all_tilts.py        — batch driver over a tilt-series dir
  pipeline/hdf5_output_validator.py    — minimal HDF5 layout probe
```

`meta.json` is the contract between stages. The slab generator writes the physics parameters ( `energy_keV`, `a_A`, `target_hkl`, `target_d_A`, `target_theta_B_mrad`, `g_dot_n_surf`, `tilt_azimuth_deg`, `bilayer_height_A`, terrace pattern, supercell dimensions, tilt sweep) once; every downstream stage reads from the same file. If a script drops `g_dot_n_surf` or `target_theta_B_mrad`, the recovered step height silently rescales — historical examples are flagged in `docs/PIPELINE_LOG.md` §2–4.

## 3. Quickstart

The commands below run a smoke test end-to-end. They assume `prismatique` (0.0.1) and `embeam` are importable in the active Python environment.

```bash
# 1. Generate a stepped Si[110] cleave-edge slab (~80 Å × 80 Å × 138 Å Si region)
python sample_generators/si110_cleave_slab_generator.py \
    --create-step \
    --outdir outputs/si110

# 2. Run one beam-tilt sweep through the forward multislice model
python pipeline/multislice_tilt_series_runner.py \
    outputs/si110/si110_object/si110_object.xyz \
    outputs/sim/

# 3. Apply the k-space sideband filter to every tilt directory
#    NOTE: process_all_tilts.py has a hardcoded base_dir near the top of main();
#    edit it to point at outputs/sim/ before running.
python pipeline/process_all_tilts.py

# 4. Invert the unwrapped phase to a step height for one tilt directory
python pipeline/step_height_reflection_formula.py \
    --results-dir outputs/sim/tilt_0650mrad/filtered \
    --meta       outputs/sim/tilt_0650mrad/meta.json
```

Single-tilt forward run (one angle, no sweep):

```bash
python pipeline/multislice_forward_model.py \
    outputs/si110/si110_object/si110_object.xyz outputs/sim/single \
    --production --timeout-min 30
```

Direct filter call on one exit wave (skipping the batch driver):

```bash
python pipeline/specular_666_filter.py \
    outputs/sim/tilt_0650mrad/hrtem_sim_wavefunction_output_of_subset_0.h5 \
    --meta outputs/sim/tilt_0650mrad/meta.json \
    --out-prefix outputs/sim/tilt_0650mrad/filtered/specular \
    --dx 0.5 --dy 0.5 --rin 8 --rout 14
```

HPC / scheduler hints:

- `multislice_forward_model.py` and `multislice_tilt_series_runner.py` cap `OMP_NUM_THREADS = MKL_NUM_THREADS = NUMBA_NUM_THREADS = 1` *before* importing `prismatique`; do not move those lines.
- GPU count is auto-detected from `CUDA_VISIBLE_DEVICES`, `SLURM_GPUS_*`, or via the override `HRTEM_NUM_GPUS`.
- Memory targeting: set `HRTEM_TARGET_MEM_GB` to auto-scale pixel size until estimated complex-wave memory fits.

## 4. Dependencies

| Package | Version | Role |
|---|---|---|
| `numpy` | `>=1.24` | Array math, FFT |
| `scipy` | `>=1.10` | Peak finding, signal utilities |
| `matplotlib` | `>=3.7` | Diagnostic plots |
| `h5py` | `>=3.8` | HDF5 wavefunction I/O |
| `ase` | `>=3.22` | Crystal building + CIF I/O for the slab generator |
| `prismatique` | `==0.0.1` | Python interface to the Prismatic multislice engine. The current code targets this exact release; later versions may break the `prismatique.hrtem.*` schema. |
| `embeam` | (compatible) | Gun/lens model objects passed to Prismatique |
| `cupy` | optional | GPU acceleration; auto-detected when installed |

The underlying Prismatic multislice/PRISM engine (C++/CUDA, distributed separately) is wrapped by `prismatique`. See the Acknowledgments section below.

## 5. Known-good / known-broken paths

See `docs/PIPELINE_LOG.md` for the authoritative status table. Headline items:

- `specular_666_filter.py` peak finder can lock onto the Si(220) sideband when run near $\theta_B$; pass `--manual-peak iy,ix` to override.
- `step_height_reflection_formula.py` now respects `g_dot_n_surf` from meta — older batch scripts that did not propagate it (e.g. the deprecated `batch_tilt_series_analysis.py`, not shipped here) returned $h\equiv 0$.
- Tilt-series rocking-curve interpretation: only **Regime A** (0 – 5.5 mrad, off-Bragg) has a fully validated peak tracker; Regime B (6 – 17.5 mrad) requires manual peak selection because the automatic tracker can hop to the Si(220) sideband.
- All historical script names containing `666` refer to the *sideband-filter step*, not the physical reflection; the physical reflection on Si is set by `meta['target_hkl']` (default Si(2,-2,0)).

## 6. Repository layout

```
si110-reflection-holography/
├── README.md
├── LICENSE                                MIT, © 2026 Hussien Ballouk
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── pipeline/
│   ├── multislice_forward_model.py        single-run plane-wave multislice
│   ├── multislice_tilt_series_runner.py   tilt-series driver
│   ├── specular_666_filter.py             k-space sideband isolation
│   ├── step_height_reflection_formula.py  phase → step height
│   ├── process_all_tilts.py               batch the filter over a tilt-series
│   ├── hdf5_output_validator.py           sanity check on Prismatique output
│   └── meta_tilt_examples.json            documented meta.json snippets
├── sample_generators/
│   └── si110_cleave_slab_generator.py     Si[110] cleave-edge slab + meta.json
└── docs/
    └── PIPELINE_LOG.md                    script-by-script status / physics notes
```

## 7. Citations

This codebase implements and extends physics from the following references. PDFs of the primary sources are held locally by the author; citations below are taken from the published metadata of each PDF and have not been independently expanded with arXiv/DOI numbers unless those are present in the PDF itself.

1. **Osakabe, N., Endo, J., Tonomura, A.**, "Observation of atomic steps by reflection electron holography", *Surface Science*, 1993. PII: `0039-6028(93)90047-N`. — Source paper for the off-axis reflection-holography step-phase formula; this repository replicates the geometry on cleaved Si.
2. **Rangel DaCosta, L., Brown, H. G., Pelz, P. M., Rakowski, A., Barber, N., O'Donovan, P., McBean, P., Jones, L., Ciston, J., Scott, M. C., Ophus, C.**, "Prismatic 2.0 – Simulation software for scanning and high resolution transmission electron microscopy (STEM and HRTEM)", *Micron*, **151** (2021) 103141. doi: `10.1016/j.micron.2021.103141`. — Multislice/PRISM engine wrapped by `prismatique`.
3. **Ophus, C.**, "A fast image simulation algorithm for scanning transmission electron microscopy", *Advanced Structural and Chemical Imaging*, **3** (2017) 13. — Original PRISM algorithm; cite if you re-use the PRISM path. (Verify the volume/article number against the open-access PDF before quoting.)
4. **Kirkland, E. J.**, *Advanced Computing in Electron Microscopy*, 3rd ed., Springer, 2020. — Reference text for multislice theory, transfer functions, FFT-friendly grid choices, and Debye–Waller treatments used throughout this code.
5. **Peng, L.-M., Whelan, M. J., Dudarev, S. L.**, *High-Energy Electron Diffraction and Microscopy*, Monographs on the Physics and Chemistry of Materials, Oxford University Press (ISBN 9780191001628). — Reference text for dynamical diffraction in transmission and reflection geometries; the physics underlying the asymmetric step-phase formula used here is developed in this monograph.
6. **Tomita, T., Shindo, D.**, *Material Characterization Using Electron Holography*, Wiley-VCH, 2022. — Modern reference for off-axis electron holography reconstruction (sideband filtering, phase unwrapping, ramp removal) implemented in `specular_666_filter.py`.

## 8. Acknowledgments

- **Prismatic** (C++/CUDA multislice/PRISM engine) — original PRISM algorithm by Ophus (2017); Prismatic 2.0 reimplementation by Rangel DaCosta et al. (Micron 2021). This codebase invokes Prismatic exclusively through its Python wrapper and does not redistribute the engine itself.
- **Prismatique** (Python wrapper) is the package this pipeline imports as `prismatique`; `embeam` provides the matched gun/lens model objects. Used here in version 0.0.1.
- **CAMTEC, University of Victoria** — instrument context for the broader thesis program (Hitachi HF-3300 STEHM, 200 keV).
- Supervision: **Dr. Arthur Blackburn** (UVic Physics, CAMTEC). Committee: **Prof. Tao Lu** (UVic ECE), **Prof. Magdalena Bazalova-Carter** (UVic Physics).
- The slab-construction approach (rotation matrix from the conventional cubic frame, bilayer-aware step cut at the wide-gap midline) draws on standard practice in the cleave-edge surface-science literature.

## 9. License

MIT License. Copyright (c) 2026 Hussien Ballouk. See `LICENSE`.

## 10. How to cite this repository

Use the metadata in `CITATION.cff` (rendered on GitHub via the "Cite this repository" button), or cite directly as:

> Ballouk, H. (2026). *si110-reflection-holography: Reflection electron holography simulation for cleaved-Si surface topography* (v0.1.0). University of Victoria. https://github.com/hussienba/si110-reflection-holography

This software accompanies the PhD program **"Advancement of Electron Ptychography towards Surface Topography and Semiconductor Device Inspection"** (Ballouk, supervisor Dr. A. Blackburn, University of Victoria) and represents *Stage 1 — simulation* of that program.
