#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REM Holography Simulation - FINAL (Prismatique 0.0.1)
- Single-run, plane-wave multislice for Osakabe-style reflection holography
- Reads meta.json (next to coords) to derive grid and angles
- Saves complex exit wave; logs absorber window (bulk side)

Usage:
  python rem_holography_final.py <coords_file> [output_dir] [--production] [--timeout-min 30]

--- SCRIPT MODIFIED AND VERIFIED ---
Key changes:
1.  Fixed critical bug where discretization/image params were not passed to sim.
2.  Added HPC-aware CPU thread management (reads SLURM_CPUS_PER_TASK).
3.  Moved hardcoded 'dz' and 'z_supersampling' to meta.json.
4.  Added correct logging via disc_params.get_core_attrs().
5.  Removed unused 'box_A' from meta.json required keys.
6.  Fixed SyntaxError typo ('7key' -> '70').
7.  (v7) Fixed AttributeError for image_params logging (using get_core_attrs).
------------------------------------
"""
import os
import sys
import json
import time
import signal
import math
import shutil
from datetime import datetime
from contextlib import contextmanager

# Thread caps BEFORE imports that spin BLAS/FFT threads
# This is crucial on HPC to stop Python fighting the scheduler
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'

import prismatique
import embeam

try:
    import cupy as cp  # noqa: F401
    print("CuPy detected - optional GPU helpers available.")
except Exception:
    print("CuPy not available; continuing without optional CuPy helpers.")


def env_float(name: str, default: float | None = None) -> float | None:
    """Read an environment variable as a float."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"Warning: ignoring {name}={raw!r} (not a float).")
        return default


def env_int(name: str, default: int | None = None) -> int | None:
    """Read an environment variable as an int."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"Warning: ignoring {name}={raw!r} (not an int).")
        return default


def _count_visible_devices(spec: str) -> int:
    tokens = [tok.strip() for tok in spec.split(",")]
    tokens = [tok for tok in tokens if tok and tok != "-1"]
    return len(tokens)


def _parse_gpu_env_value(value: str) -> int:
    total = 0
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        token = token.split("=")[-1]
        if ":" in token:
            parts = token.split(":")
            try:
                total += int(parts[-1])
                continue
            except ValueError:
                pass
        try:
            total += int(token)
        except ValueError:
            continue
    return total


def detect_available_gpus(fallback: int = 0) -> int:
    """Best-effort detection of GPUs assigned to the job."""
    override = env_int("HRTEM_NUM_GPUS")
    if override is not None:
        return max(0, override)

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        count = _count_visible_devices(visible)
        if count:
            return count

    for env_name in ("SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE", "SLURM_GPUS"):
        value = os.environ.get(env_name)
        if not value:
            continue
        parsed = _parse_gpu_env_value(value)
        if parsed:
            return parsed

    return fallback


def normalize_transfer_mode(raw: str | None) -> str:
    if not raw:
        return "auto"
    value = raw.strip().lower()
    if value in {"auto", "streaming"}:
        return value
    if value in {"single", "single-transfer", "single_transfer", "singlexfer"}:
        return "single-transfer"
    return "auto"


def estimate_wave_memory_gb(
    nx: int,
    ny: int,
    num_slices: int,
    z_supersampling: int,
    dtype_bytes: int = 16,
    safety_factor: float = 2.0,
) -> float:
    """Approximate memory footprint for storing complex exit waves."""
    total_voxels = (
        max(1, nx)
        * max(1, ny)
        * max(1, num_slices)
        * max(1, z_supersampling)
    )
    bytes_needed = total_voxels * dtype_bytes * max(1.0, safety_factor)
    return bytes_needed / (1024.0 ** 3)


@contextmanager
def timeout(seconds: int):
    """Context manager for enforcing a wall clock timeout."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)


def read_meta(coords_path: str) -> dict:
    """Reads and validates the required meta.json file."""
    meta_path = os.path.join(os.path.dirname(coords_path), "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found next to coords: {meta_path}")
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    for k in ["advisory_pixel_size_A", "alpha_deg", "energy_keV"]:
        if k not in meta:
            raise KeyError(f"meta.json missing required key: {k}")
    return meta


def analyze_prismatic_xyz(coords_path: str) -> dict:
    """Quickly parses the header of a .xyz file for box dimensions."""
    with open(coords_path, "r") as f:
        header = f.readline().strip()
        dims_line = f.readline().strip()
    try:
        Lx, Ly, Lz = [float(x) for x in dims_line.split()]
    except Exception as exc:
        raise ValueError(f"Malformed prismatic XYZ header: {dims_line}") from exc
    return {"Lx": Lx, "Ly": Ly, "Lz": Lz, "header": header}


def derive_grid_from_meta(Lx: float, Ly: float, px_A: float) -> tuple[int, int]:
    """Derives an FFT-friendly grid size from box dims and target pixel size."""
    def _nearest_fft_friendly(n: int) -> int:
        # prefer even sizes with small prime factors (2,3,5)
        def is_235_only(x: int) -> bool:
            y = x
            for p in (2, 3, 5):
                while y % p == 0 and y > 1:
                    y //= p
            return y == 1
        search_spans = [0.05, 0.1, 0.2]
        best = None
        for span in search_spans:
            lo = max(128, int((1 - span) * n))
            hi = max(lo, int((1 + span) * n))
            for k in range(lo, hi + 1):
                if k % 4 != 0:
                    continue
                if not is_235_only(k):
                    continue
                if best is None or abs(k - n) < abs(best - n) or (abs(k - n) == abs(best - n) and k > best):
                    best = k
            if best is not None:
                break
        if best is None:
            best = ((max(128, n) + 3) // 4) * 4
        return best
    # use ceil to ensure Δx ≤ advisory px, then snap to FFT-friendly sizes
    gx_raw = max(128, int(math.ceil(Lx / px_A)))
    gy_raw = max(128, int(math.ceil(Ly / px_A)))
    gx = _nearest_fft_friendly(gx_raw)
    gy = _nearest_fft_friendly(gy_raw)
    return gx, gy


def relativistic_lambda_A(E_keV: float) -> float:
    """Calculates the relativistic electron wavelength in Angstroms."""
    h = 6.62607015e-34
    e = 1.602176634e-19
    m0 = 9.10938356e-31
    c = 299792458.0
    V = E_keV * 1e3 * e
    p = (2 * m0 * V * (1 + V / (2 * m0 * c * c))) ** 0.5
    return (h / p) * 1e10


def quick_phase_report(meta: dict) -> None:
    """Prints a quick check of the expected phase shift."""
    try:
        E = float(meta["energy_keV"])
        lam = relativistic_lambda_A(E)
        a = float(meta["a_A"]) if "a_A" in meta else None
        d111 = float(meta["d111_A"]) if "d111_A" in meta else (a / math.sqrt(3.0) if a else None)
        alpha = math.radians(float(meta["alpha_deg"]))
        if d111 is None:
            return
        k = 2 * math.pi / lam
        Kperp = k * math.sin(alpha)
        dphi = 2 * Kperp * d111
        print(f"λ={lam:.6f} Å | θ_B={math.degrees(alpha):.3f}° | d111={d111:.3f} Å | Δφ≈{dphi:.3f} rad")
    except Exception:
        pass


def create_system_params(coords_path: str, energy_keV: float, disc_params_obj=None, absorber: dict | None = None):
    """Builds the hrtem.system.ModelParams object."""
    # Early format check (optional but helpful)
    try:
        prismatique.sample.check_atomic_coords_file_format(coords_path)
    except Exception:
        pass
    # Build a sample.ModelParams
    sample_kwargs = {
        "atomic_coords_filename": coords_path,
        "unit_cell_tiling": (1, 1, 1),
    }
    # Attach discretization at the sample level if provided (schema used by 0.0.2)
    if disc_params_obj is not None:
        sample_kwargs["discretization_params"] = disc_params_obj
    
    # Advisory atomic potential extent (Å)
    sample_kwargs["atomic_potential_extent"] = 8.0
    
    if absorber is not None:
        sample_kwargs["absorbing_layers"] = [{
            "zmin_A": float(absorber["zmin_A"]),
            "zmax_A": float(absorber["zmax_A"]),
            "eta": float(absorber.get("eta", 0.5)),
        }]
    
    try:
        sample_model_params = prismatique.sample.ModelParams(**sample_kwargs)
    except Exception:
        # Retry without absorber if schema unsupported
        sample_kwargs.pop("absorbing_layers", None)
        sample_model_params = prismatique.sample.ModelParams(**sample_kwargs)

    # Build embeam gun/lens model objects
    try:
        gun_model_params = embeam.gun.ModelParams(
            mean_beam_energy=float(energy_keV),
            intrinsic_energy_spread=0.5e-3,
        )
    except Exception:
        gun_model_params = None
    try:
        lens_model_params = embeam.lens.ModelParams(
            coherent_aberrations=(),
            chromatic_aberration_coef=0.0,
        )
    except Exception:
        lens_model_params = None

    kwargs = {
        "sample_specification": sample_model_params,
        "gun_model_params": gun_model_params,
        "lens_model_params": lens_model_params,
    }
    return prismatique.hrtem.system.ModelParams(**kwargs)


def create_output_params(output_dir: str, image_params_obj=None):
    """Builds the hrtem.output.Params object."""
    os.makedirs(output_dir, exist_ok=True)
    kwargs = {
        "output_dirname": output_dir,
    }
    # Attach nested image params if provided (schema used by 0.0.2)
    if image_params_obj is not None:
        kwargs["image_params"] = image_params_obj
    
    # Provide a reasonable max data size cap
    kwargs["max_data_size"] = 8_000_000_000  # 2 GB
    
    # Do not save potential slices by default
    kwargs["save_potential_slices"] = False
    
    return prismatique.hrtem.output.Params(**kwargs)


def create_worker_params():
    """Builds the worker.Params, dynamically setting CPU threads for HPC."""
    try:
        # --- MODIFIED: Dynamically set CPU threads for HPC ---
        try:
            # Best for Slurm jobs (e.g., Narval)
            num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
        except ValueError:
            num_cpus = 1
        
        # Fallback for local testing (use all cores)
        if num_cpus == 1 and 'SLURM_CPUS_PER_TASK' not in os.environ:
            try:
                num_cpus = os.cpu_count() or 1
            except Exception:
                num_cpus = 1 # Failsafe
        
        gpu_count = detect_available_gpus()
        gpu_streams = max(1, env_int("HRTEM_GPU_STREAMS_PER_DEVICE") or 1)
        gpu_batch_size = max(1, env_int("HRTEM_GPU_BATCH_SIZE") or 1)
        gpu_transfer_mode = normalize_transfer_mode(os.environ.get("HRTEM_GPU_TRANSFER_MODE"))

        cpu_override = env_int("HRTEM_CPU_WORKERS")
        cpu_with_gpu = max(1, env_int("HRTEM_CPU_WORKERS_WITH_GPU") or 1)
        if cpu_override and cpu_override > 0:
            num_cpu_workers = cpu_override
        else:
            num_cpu_workers = cpu_with_gpu if gpu_count > 0 else num_cpus
        
        print(f"Configuring CPU worker pool for {num_cpu_workers} thread(s).")
        # --- END MODIFIED ---

        cpu_params = prismatique.worker.cpu.Params(
            enable_workers=True,
            num_worker_threads=num_cpu_workers,
            batch_size=max(1, num_cpu_workers // 4 if gpu_count == 0 else 1),
        )
        worker_kwargs = {"cpu_params": cpu_params}
        
        try:
            gpu_params = prismatique.worker.gpu.Params(
                num_gpus=max(0, gpu_count),
                batch_size=gpu_batch_size,
                data_transfer_mode=gpu_transfer_mode,
                num_streams_per_gpu=gpu_streams,
            )
            worker_kwargs["gpu_params"] = gpu_params
            if gpu_count > 0:
                print(
                    f"Configuring GPU worker pool for {gpu_count} device(s) "
                    f"(transfer={gpu_transfer_mode}, streams={gpu_streams})."
                )
            else:
                print("GPU workers disabled (num_gpus=0).")
        except Exception as exc:
            print(f"Warning: Failed to configure GPU workers ({exc}).")
        
        return prismatique.worker.Params(**worker_kwargs)
    
    except Exception as exc:
        print(f"Warning: Falling back to default worker parameters ({exc}).")
        return None


def create_discretization_and_image_params(
    meta: dict, Lx: float, Ly: float, Lz: float, gx: int, gy: int,
    dz_A: float, z_supersampling_factor: int
):
    """Builds the discretization and image parameter objects."""
    disc_params = None
    image_params = None
    
    def _reduced_dim(dim: int) -> int:
        if dim % 4 != 0:
            raise ValueError(f"Grid dimension {dim} must be divisible by 4.")
        return dim // 4
    
    reduced_dims = (_reduced_dim(int(gx)), _reduced_dim(int(gy)))
    
    # z-plane to save complex wave (vacuum sampling plane)
    z_sampling = float(meta.get("z_sampling_A", Lz))
    # clamp sampling plane to [0, Lz]
    if z_sampling < 0.0:
        z_sampling = 0.0
    if z_sampling > Lz:
        z_sampling = Lz
    
    dz = dz_A  # Use value from meta
    num_slices = max(1, int(math.ceil(Lz / dz)))
    z_ss = z_supersampling_factor # Use value from meta

    try:
        disc_params = prismatique.discretization.Params(
            sample_supercell_reduced_xy_dims_in_pixels=reduced_dims,
            num_slices=int(num_slices),
            z_supersampling=int(z_ss),
            interpolation_factors=(1, 1),
        )
    except Exception as e:
        print(f"Warning: Discretization params not applied: {e}")
        disc_params = None
    
    try:
        image_params = prismatique.hrtem.image.Params(
            postprocessing_seq=(),
            avg_num_electrons_per_postprocessed_image=1.0,
            apply_shot_noise=False,
            save_wavefunctions=True,
            save_final_intensity=True,
        )
    except Exception as e:
        print(f"Warning: Image params not applied: {e}")
        image_params = None
        
    return disc_params, image_params


def run_sim(coords_file: str, output_dir: str, timeout_min: int, production: bool) -> bool:
    """Main simulation driver function."""
    print("="*70)
    print("REM HOLOGRAPHY SIMULATION - FINAL (v7-corrected)")
    print("="*70)
    print(f"Start: {datetime.now()}")
    print(f"Mode: {'production' if production else 'test'}")
    print(f"Input: {coords_file}")
    print(f"Output: {output_dir}\n")
    
    if not os.path.exists(coords_file):
        raise FileNotFoundError(coords_file)
        
    meta = read_meta(coords_file)
    head = analyze_prismatic_xyz(coords_file)
    Lx, Ly, Lz = head["Lx"], head["Ly"], head["Lz"]
    
    px_advisory = float(meta["advisory_pixel_size_A"])
    px_scale = env_float("HRTEM_PIXEL_SCALE")
    if px_scale and px_scale > 0 and px_scale != 1.0:
        px_advisory *= px_scale
        print(f"Applied HRTEM_PIXEL_SCALE={px_scale:.3f} -> target px≈{px_advisory:.3f} Å")

    px_absolute = env_float("HRTEM_PIXEL_SIZE_A")
    if px_absolute and px_absolute > 0:
        px_advisory = px_absolute
        print(f"Applied explicit HRTEM_PIXEL_SIZE_A={px_advisory:.3f} Å")

    max_px = env_float("HRTEM_MAX_PIXEL_SIZE_A")
    if max_px and px_advisory > max_px:
        print(f"Clamping pixel size to HRTEM_MAX_PIXEL_SIZE_A={max_px:.3f} Å")
        px_advisory = max_px

    def refresh_grid(px_value: float) -> tuple[int, int, float, float]:
        gx_loc, gy_loc = derive_grid_from_meta(Lx, Ly, px_value)
        return gx_loc, gy_loc, Lx / gx_loc, Ly / gy_loc

    gx, gy, dx, dy = refresh_grid(px_advisory)
    energy_keV = float(meta.get("energy_keV", 200.0))
    alpha_deg = float(meta["alpha_deg"])  # geometry already applied in coords
    theta_max_deg = float(meta.get("advisory_theta_max_deg", 1.3 * alpha_deg))
    absorb_min = meta.get("z_absorb_min_A", None)
    absorb_max = meta.get("z_absorb_max_A", None)

    # Use 1.0 Å and 4x as sane defaults if not in meta.json
    dz_A = float(meta.get("advisory_dz_A", 1.0))
    dz_scale = env_float("HRTEM_DZ_SCALE")
    if dz_scale and dz_scale > 0 and dz_scale != 1.0:
        dz_A *= dz_scale
        print(f"Applied HRTEM_DZ_SCALE={dz_scale:.3f} -> dz≈{dz_A:.3f} Å")

    dz_absolute = env_float("HRTEM_DZ_A")
    if dz_absolute and dz_absolute > 0:
        dz_A = dz_absolute
        print(f"Applied explicit HRTEM_DZ_A={dz_A:.3f} Å")

    z_supersampling = int(meta.get("z_supersampling", 4))
    z_sup_override = env_int("HRTEM_Z_SUPERSAMPLING")
    if z_sup_override and z_sup_override > 0:
        z_supersampling = z_sup_override
        print(f"Applied HRTEM_Z_SUPERSAMPLING={z_supersampling}")

    print(f"Cell: {Lx:.1f} × {Ly:.1f} × {Lz:.1f} Å")
    print(f"Grid (effective): {gx} × {gy}   px≈{min(dx, dy):.3f} Å")
    print(f"Energy: {energy_keV:.1f} keV   alpha (deg): {alpha_deg:.3f}   theta_max (deg): {theta_max_deg:.3f}")
    print(f"Slicing: dz={dz_A:.2f} Å, z_supersampling={z_supersampling}x")
    
    if absorb_min is not None and absorb_max is not None:
        print(f"Absorber window (z, Å): [{absorb_min:.1f}, {absorb_max:.1f}] (bulk side)")
        
    # Nyquist self-check
    qmax = math.pi / min(dx, dy)
    a_meta = float(meta["a_A"]) if "a_A" in meta else None
    g_max = 2 * math.pi / (a_meta / math.sqrt(2)) if a_meta else None
    print(
        f"Δx={dx:.3f} Å, Δy={dy:.3f} Å | q_max(Nyquist)={qmax:.2f} Å⁻¹"
        + (f" | est |g|max≈{g_max:.2f} Å⁻¹" if g_max else "")
    )
    if g_max and qmax < g_max:
        print("WARNING: grid may undersample high-q scattering; consider smaller px (larger gx, gy).")
        
    # Quick physics report
    quick_phase_report(meta)

    # Memory estimation + optional auto-scaling
    preview_slices = max(1, int(math.ceil(Lz / dz_A)))
    mem_safety = env_float("HRTEM_MEMORY_SAFETY_FACTOR", 2.0) or 2.0
    mem_gb = estimate_wave_memory_gb(gx, gy, preview_slices, z_supersampling, safety_factor=mem_safety)
    print(
        f"Estimated complex-wave memory ≈ {mem_gb:.1f} GB "
        f"(safety factor {mem_safety:.1f}×)."
    )
    target_mem_gb = env_float("HRTEM_TARGET_MEM_GB")
    original_px = px_advisory
    if target_mem_gb and mem_gb > target_mem_gb:
        scale = math.sqrt(mem_gb / target_mem_gb)
        px_advisory *= scale
        print(
            f"HRTEM_TARGET_MEM_GB={target_mem_gb:.1f} requested; "
            f"increasing px by {scale:.2f}× (from {original_px:.3f} Å to {px_advisory:.3f} Å)."
        )
        if max_px and px_advisory > max_px:
            print("Pixel size capped by HRTEM_MAX_PIXEL_SIZE_A.")
            px_advisory = max_px
        gx, gy, dx, dy = refresh_grid(px_advisory)
        mem_gb = estimate_wave_memory_gb(gx, gy, preview_slices, z_supersampling, safety_factor=mem_safety)
        print(f"Recomputed grid: {gx} × {gy}   px≈{min(dx, dy):.3f} Å | est mem {mem_gb:.1f} GB")
        if mem_gb > target_mem_gb:
            print("Warning: Unable to meet memory target exactly; consider larger HRTEM_PIXEL_SCALE or HRTEM_DZ_SCALE.")

    # 1. Create discretization and image params FIRST
    print("\nDeriving simulation grid and slice parameters...")
    disc_params, image_params = create_discretization_and_image_params(
        meta, Lx, Ly, Lz, gx, gy, dz_A, z_supersampling
    )
    
    # Use the get_core_attrs() method to correctly read back parameters
    # for logging. Fall back to printing local vars if this fails.
    if disc_params:
        try:
            dcore = disc_params.get_core_attrs()
            reduced = dcore['sample_supercell_reduced_xy_dims_in_pixels']
            full_dims = (int(reduced[0]) * 4, int(reduced[1]) * 4)
            print(f"  Grid: {full_dims} (reduced={tuple(reduced)})")
            print(f"  Slices: {dcore['num_slices']} (dz={dz_A:.2f} Å)")
            print(f"  Z-Supersampling: {dcore['z_supersampling']}x")
        except Exception as e:
            print(f"  Could not read back disc_params (using local vars for log): {e}")
            print(f"  Grid set to: ({int(gx)}, {int(gy)})")
            print(f"  Slicing params sent: dz={dz_A:.2f} Å, z_supersampling={z_supersampling}x")
    else:
        print("  Warning: Discretization parameter object was NOT created.")
        
    # --- MODIFIED: CRITICAL BUG FIX (v7) ---
    # Applied the same get_core_attrs() fix to image_params logging
    if image_params:
        try:
            icore = image_params.get_core_attrs()
            print(f"  Will save complex exit wave: {icore['save_wavefunctions']}")
        except Exception:
            print(f"  Will save complex exit wave: True (readback failed)")
    # --- END MODIFIED ---

    # 2. Setup absorber
    absorber = None
    if absorb_min is not None and absorb_max is not None:
        absorber = {"zmin_A": float(absorb_min), "zmax_A": float(absorb_max), "eta": 0.5}

    # 3. NOW create system and output params, PASSING IN the objects
    system_params = create_system_params(
        coords_file, 
        energy_keV, 
        disc_params_obj=disc_params,
        absorber=absorber
    )
    output_params = create_output_params(
        output_dir, 
        image_params_obj=image_params
    )
    worker_params = create_worker_params()
    
    # Assemble final simulation parameters
    sim_kwargs = {
        "hrtem_system_model_params": system_params,
        "output_params": output_params,
    }
    if worker_params is not None:
        sim_kwargs["worker_params"] = worker_params
    
    sim_params = prismatique.hrtem.sim.Params(**sim_kwargs)
    
    try:
        sim_params.dump(os.path.join(output_dir, "simulation_parameters.json"), overwrite=True)
    except Exception:
        pass
        
    # Save a copy of meta for provenance
    try:
        shutil.copyfile(os.path.join(os.path.dirname(coords_file), "meta.json"),
                        os.path.join(output_dir, "meta.json"))
    except Exception:
        pass

    print("\nStarting simulation...")
    t0 = time.time()
    try:
        with timeout(int(timeout_min * 60)):
            prismatique.hrtem.sim.run(sim_params=sim_params)
    except TimeoutError:
        print(f"\nTimeout after {timeout_min} minutes")
        return False
    except Exception as e:
        print(f"\nSimulation failed: {e} ({type(e).__name__})")
        return False
        
    dt = time.time() - t0
    print(f"\n✓ Done in {dt:.1f} s ({dt/60:.1f} min)")
    print("Saved: intensity + complex exit wave (check output dir).")
    
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python rem_holography_final.py <coords_file> [output_dir] [--production] [--timeout-min N]")
        sys.exit(1)
        
    coords_file = sys.argv[1]
    output_dir = "./rem_holo_output"
    production = False
    timeout_min = 5
    output_set = False
    
    # Simple argument parsing
    for i, arg in enumerate(sys.argv[2:], start=2):
        if arg == "--production":
            production = True
        elif arg == "--timeout-min" and i + 1 < len(sys.argv):
            try:
                timeout_min = int(sys.argv[i + 1])
            except Exception:
                pass
        elif not arg.startswith("--") and not output_set:
            output_dir = arg
            output_set = True
            
    ok = run_sim(coords_file, output_dir, timeout_min=(30 if production else timeout_min), production=production)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()