# Plane-wave multislice simulation in reflection geometry using Prismatic (via the Prismatique Python wrapper), with off-axis electron-holographic reconstruction pipeline.

Reflection electron holography forward-simulation and reconstruction pipeline for the cleaved-Si[110] (1,1,-1) facet geometry, replicating the off-axis holographic step-phase measurement using diamond-cubic silicon.

---
## 2. Pipeline architecture

```
  ┌──────────────────────────────────────────────┐
  │  sample_generators/si110_cleave_slab_         │
  │  generator.py                                 │
  │   Si[110] cleave-edge slab + meta.json        │
  │   (cleave-frame: z‖[110]beam, x‖[1,-1,1]n̂,    │
  │    y‖[1,-1,-2] step-normal-in-plane)          │
  └──────────────┬───────────────────────────────┘
                 │  .xyz (Prismatic) + meta.json
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
  │   Loops over beam tilts               │
  └──────────────┬───────────────────────────────┘
                 │  .h5 + meta.json
                 ▼
  ┌──────────────────────────────────────────────┐
  │  pipeline/specular_filter.py              │
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
postprocessing>


  pipeline/process_all_tilts.py        — batch driver over a tilt-series dir
  pipeline/hdf5_output_validator.py    — minimal HDF5 layout probe, for checking the structure of data. 
```
## 4. Dependencies

| Package | Version | Role |
|---|---|---|
| `numpy` | `>=1.24` | Array math, FFT |
| `scipy` | `>=1.10` | Peak finding, signal utilities, signal proccesing |
| `matplotlib` | `>=3.7` | Diagnostic plots |
| `h5py` | `>=3.8` | HDF5 wavefunction I/O |
| `ase` | `>=3.22` | Crystal building + CIF I/O for the slab generator |
| `prismatique` | `==0.0.1` | Python interface to the Prismatic multislice engine. The current code targets this exact release; later versions may break the `prismatique.hrtem.*` schema. |
| `embeam` | `0.0.1`  | Gun/lens model objects passed to Prismatique |
| `cupy` | for HPC clsuter | GPU acceleration; auto-detected  |

The underlying Prismatic multislice/PRISM engine (C++/CUDA, distributed separately) is wrapped by `prismatique`. 

## 7. Citations

2. **Rangel DaCosta, L., Brown, H. G., Pelz, P. M., Rakowski, A., Barber, N., O'Donovan, P., McBean, P., Jones, L., Ciston, J., Scott, M. C., Ophus, C.**, "Prismatic 2.0 – Simulation software for scanning and high resolution transmission electron microscopy (STEM and HRTEM)", *Micron*, **151** (2021) 103141. doi: `10.1016/j.micron.2021.103141`. — Multislice/PRISM engine wrapped by `prismatique`.
3. https://github.com/mrfitzpa/prismatique 
4. **Ophus, C.**, "A fast image simulation algorithm for scanning transmission electron microscopy", *Advanced Structural and Chemical Imaging*, **3** (2017) 13. — Original PRISM algorithm; cite if you re-use the PRISM path. (Verify the volume/article number against the open-access PDF before quoting.)
5. **Kirkland, E. J.**, *Advanced Computing in Electron Microscopy*, 3rd ed., Springer, 2020. — Reference text for multislice theory, transfer functions, FFT-friendly grid choices, and Debye–Waller treatments used throughout this code.
6. **Peng, L.-M., Whelan, M. J., Dudarev, S. L.**, *High-Energy Electron Diffraction and Microscopy*, Monographs on the Physics and Chemistry of Materials, Oxford University Press (ISBN 9780191001628). — Reference text for dynamical diffraction in transmission and reflection geometries; 
7. **Tomita, T., Shindo, D.**, *Material Characterization Using Electron Holography*, Wiley-VCH, 2022. — Modern reference for off-axis electron holography reconstruction (sideband filtering, phase unwrapping, ramp removal) implemented in `specular_filter.py`.

## 8. Acknowledgments

- **Prismatique** (Python wrapper) is the package this pipeline imports as `prismatique`; `embeam` provides the matched gun/lens model objects. Used here in version 0.0.1. https://github.com/mrfitzpa/prismatique - Author: Matthew Fitzpatrick

## 9. License

MIT License.  See `LICENSE`.

