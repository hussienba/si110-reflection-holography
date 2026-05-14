# Holography Pipeline Log

**Project:** Electron holography simulation — replicating Osakabe et al.
**Author:** Hussien Ballouk, UVic (supervisor: Dr. Blackburn)
**Last updated:** 2026-04-04

---

## 1. Pipeline Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ELECTRON HOLOGRAPHY SIMULATION PIPELINE                │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────┐
  │  Virtual Si sample (CIF /    │
  │  si100_slab_generator.py)    │
  └──────────────┬───────────────┘
                 │  .xyz / HDF5 structure
                 ▼
  ┌──────────────────────────────────────────────────┐
  │         multislice_forward_model.py              │
  │  Prismatique HRTEM wrapper — STEM/HRTEM exit     │
  │  wavefunction via multislice propagation         │
  └──────────────────────┬───────────────────────────┘
                         │  exit wavefunction (complex array)
                         │  + beam-tilt metadata
                         ▼
  ┌──────────────────────────────────────────────────┐
  │       multislice_tilt_series_runner.py           │
  │  Loops over tilt angles; saves per-tilt HDF5;    │
  │  assembles rocking-curve dataset                 │
  └──────────────────────┬───────────────────────────┘
                         │  tilt-series HDF5 stack
                         ▼
  ┌──────────────────────────────────────────────────┐
  │           specular_666_filter.py                 │
  │  k-space sideband reconstruction; isolates       │
  │  target reflection; unwraps phase                │
  └──────────────────────┬───────────────────────────┘
                         │  phase map Δφ(x,y)
                         ▼
  ┌──────────────────────────────────────────────────┐
  │       step_height_reflection_formula.py          │
  │  h = Δφ λ / (4π sin θ_B)                        │
  │  Returns step height in Å                        │
  └──────────────────────┴───────────────────────────┘

  ─ ─ ─ ─ ─ ─ ─ ─ ─  Auxiliary / validation  ─ ─ ─ ─ ─ ─ ─ ─ ─

  ┌──────────────────────────┐   ┌──────────────────────────────┐
  │  hdf5_output_validator   │   │  batch_tilt_series_analysis  │
  │  .py                     │   │  .py  [BROKEN — see §4]      │
  └──────────────────────────┘   └──────────────────────────────┘

  ┌──────────────────────────┐   ┌──────────────────────────────┐
  │  measure_all_steps.py    │   │  figure_tilt_series_v1/v2/v3 │
  │  [BROKEN — see §4]       │   │  .py  +  figure_hologram_    │
  └──────────────────────────┘   │  publication.py              │
                                  └──────────────────────────────┘
```

---

## 2. Script Inventory

| Script | Role | Status | Key issue |
|---|---|---|---|
| `multislice_forward_model.py` | Prismatique HRTEM wrapper | correct | Focuses to mid-slab |
| `multislice_tilt_series_runner.py` | Tilt series runner | correct | 128-px floor limits resolution |
| `specular_666_filter.py` | k-space sideband reconstruction | **errors** | (666) forbidden in Si (F=0); peak picker fails at 6 mrad |
| `step_height_reflection_formula.py` | Step height from phase | correct | Uses h = Δφλ / 4π sin(θ) |
| `step_height_mip_formula_DEPRECATED.py` | DEPRECATED height formula | **wrong** | MIP formula; systematic error ×35800 |
| `batch_tilt_series_analysis.py` | Batch tilt analysis | **wrong** | `alpha_deg` never propagated → all h = 0 |
| `figure_tilt_series_v1/v2/v3.py` | Publication figures | correct | — |
| `figure_hologram_publication.py` | Main publication figure | correct | Foreshortening correction not applied |
| `hdf5_output_validator.py` | HDF5 data validator | correct | — |
| `measure_all_steps.py` | Step measurement script | **wrong** | h_theo uses a/4 not d₁₁₁ |

---

## 3. Tilt Series Rocking Curve Summary

```
  Beam tilt θ (mrad)
  ──────────────────────────────────────────────────────────────────
   0         5        10        15        20        25
   │                                                  │
   ├──── Regime A ────┼──────── Regime B ────────┼── Regime C ──┤   ►  >22 mrad
   │  (0 – 5.5 mrad)  │     (6 – 17.5 mrad)      │(18 – 22 mrad)│    REMOVED
   │                  │                          │              │
   │  off-Bragg       │  near-Bragg              │  strong tilt │
   │  amp ≈ 0.128     │  amp ≈ 0.820             │  amp falling │
   │  peak tracker    │  peak tracker INVALID     │  partially   │
   │  VALID           │  (jumped to Si(220)       │  meaningful  │
   │  12 datasets     │  artefact)               │  ramp        │
   │                  │  23 datasets             │  11 datasets │
   └──────────────────┴──────────────────────────┴──────────────┘
                                                        16 dirs removed
                                                        (Nyquist exceeded)
```

| Regime | θ range (mrad) | Sideband amplitude | Peak tracker | N datasets |
|---|---|---|---|---|
| A — off-Bragg | 0 – 5.5 | ~0.128 | VALID | 12 |
| B — near-Bragg | 6 – 17.5 | ~0.820 | **INVALID** (Si(220) artefact) | 23 |
| C — strong tilt | 18 – 22 | falling | partially meaningful | 11 |
| — | >22 | N/A | N/A | 16 removed (Nyquist) |

---

## 4. Critical Physics Issues

- **Si(666) systematically absent.** In the diamond-cubic structure factor F(hkl)=0 for all (h+k+l)=4n+2 with mixed parity; Si(666) is a forbidden reflection. The nearest accessible reflection producing usable holographic contrast is **Si(220) at θ_B = 6.53 mrad** (200 kV, d₂₂₀ = 1.920 Å). The filter must be retargeted.

- **Si(111) monoatomic steps invisible at the (666) condition.** A single [111] bilayer step produces a phase shift Δφ = 2π × (2 × step_height / d₁₁₁) = 12π ≡ 0 (mod 2π). The step is completely phase-invisible at this reflection regardless of filter quality.

- **No surface step present in any simulated sample.** The torus groove script (`si100_torus_groove_model.py`) constructs a cylindrical trench geometry but never calls the step-generator routine. All x-column z_max values are uniform. No step signal is possible until a two-terrace slab with explicit step height a₀/4 = 1.36 Å is introduced.
