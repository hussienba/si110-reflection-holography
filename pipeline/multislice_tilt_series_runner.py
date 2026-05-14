#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REM Holography Simulation - UNIFIED + SNAPPED (Prismatique 0.0.1)
- Single-run, plane-wave multislice for Osakabe-style reflection holography
- Reads meta.json (next to coords) to derive grid and angles
- Saves complex exit wave; logs absorber window (bulk side)
- MERGED: Unified simulation engine (fast) + Tilt Snapping logic (precise)

Usage:
  python holography_unified_snapped.py <coords_file> [output_dir] [--production] [--timeout-min 30]
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


def parse_tilt_angles(meta: dict) -> list[float]:
    """Parse tilt angles from meta.json, supporting both step-based and explicit array methods.
    
    Supports three methods:
    1. Step-based: tilt_start_mrad, tilt_end_mrad, tilt_step_mrad
    2. Explicit array: tilt_angles_mrad (list or single value)
    3. Default: [0.0] if none specified (backward compatible)
    
    Returns:
        List of tilt angles in milliradians
    """
    # Method 1: Step-based range
    if all(k in meta for k in ["tilt_start_mrad", "tilt_end_mrad", "tilt_step_mrad"]):
        start = float(meta["tilt_start_mrad"])
        end = float(meta["tilt_end_mrad"])
        step = float(meta["tilt_step_mrad"])
        if step <= 0:
            raise ValueError("tilt_step_mrad must be positive")
        # Use numpy-style arange logic
        n_steps = int(round((end - start) / step)) + 1
        angles = [start + i * step for i in range(n_steps)]
        # Ensure we don't exceed end due to floating point errors
        angles = [a for a in angles if a <= end + 1e-9]
        return angles
    
    # Method 2: Explicit array
    if "tilt_angles_mrad" in meta:
        angles = meta["tilt_angles_mrad"]
        # Handle single value
        if isinstance(angles, (int, float)):
            return [float(angles)]
        # Handle array
        if isinstance(angles, list):
            return [float(a) for a in angles]
        raise ValueError("tilt_angles_mrad must be a number or list of numbers")
    
    # Method 3: Default (backward compatible)
    return [0.0]


def format_tilt_angle(angle_mrad: float) -> str:
    """Format tilt angle for use in output directory names.
    
    Args:
        angle_mrad: Tilt angle in milliradians
    
    Returns:
        Formatted string like 'tilt_0000mrad' or 'tilt_2400mrad'
    """
    # Convert to integer milliradians for cleaner naming
    angle_int = int(round(angle_mrad * 100))  # Store as 0.01 mrad precision
    return f"tilt_{angle_int:04d}mrad"


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


def create_system_params(coords_path: str, energy_keV: float, disc_params_obj=None, absorber: dict | None = None, tilt_angles_mrad: list[float] = None, cell_dim_A: list[float] | None = None, tilt_azimuth_deg: float = 0.0):
    """Builds the hrtem.system.ModelParams object.

    Args:
        coords_path: Path to atomic coordinates file
        energy_keV: Beam energy in keV
        disc_params_obj: Discretization parameters object
        absorber: Absorber layer specification dict
        tilt_angles_mrad: List of beam tilt angles in mrad (ALL simulated together!)
        cell_dim_A: Unit cell dimensions [x, y, z] in Angstroms (optional, for tilt snapping)
        tilt_azimuth_deg: Azimuth (deg from +k_x) along which to apply the tilt sweep.
            For Si(2,-2,0) with the cleave-edge slab frame this is ~35.26 deg.
            Defaults to 0.0 (legacy: tilt along +k_x), used when meta does not
            specify it (e.g. the old Si(100) slabs).
    """
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
    
    # Build tilt parameters for prismatique
    # NOTE: tilt_params simulates ALL angles in ONE run!
    # The window selects points from the discretized angular grid.
    tilt_params = None
    if tilt_angles_mrad is None or len(tilt_angles_mrad) == 0:
        tilt_angles_mrad = [0.0]
    
    try:
        # --- MERGED: TILT SNAPPING LOGIC ---
        # Snap the center tilt to the grid to minimize artifacts
        min_tilt = min(tilt_angles_mrad)
        max_tilt = max(tilt_angles_mrad)
        center_tilt = (min_tilt + max_tilt) / 2.0
        radial_span = max_tilt - min_tilt
        
        snapped_center_tilt = center_tilt
        if cell_dim_A is not None and abs(center_tilt) > 1e-6:
            try:
                Lx = cell_dim_A[0]
                # Relativistic wavelength calculation
                V = energy_keV * 1000.0
                wavelength_A = 12.2643 / math.sqrt(V * (1.0 + V * 0.978476e-6))
                
                # Grid spacing in mrad
                delta_theta_x_rad = wavelength_A / Lx
                delta_theta_x_mrad = delta_theta_x_rad * 1000.0
                
                # Snap to nearest integer multiple
                n_x = round(center_tilt / delta_theta_x_mrad)
                snapped_center_tilt = n_x * delta_theta_x_mrad
                
                if abs(snapped_center_tilt - center_tilt) > 1e-4:
                    print(f"  Snapping CENTER tilt: {center_tilt:.4f} -> {snapped_center_tilt:.4f} mrad (n={n_x}, grid={delta_theta_x_mrad:.4f} mrad)")
            except Exception as e:
                print(f"  Warning: Could not snap tilt angle ({e}). Using requested angle.")
        
        # Place the tilt offset along the in-plane direction of the target g-vector.
        # For Si(2,-2,0) on the cleave-edge slab the meta-supplied azimuth puts the
        # offset along the (220) k-vector; legacy slabs default to +k_x.
        az_rad = math.radians(float(tilt_azimuth_deg))
        offset_kx = float(snapped_center_tilt) * math.cos(az_rad)
        offset_ky = float(snapped_center_tilt) * math.sin(az_rad)
        tilt_params = prismatique.tilt.Params(
            offset=[offset_kx, offset_ky],
            window=[0.0, float(radial_span / 2.0 + 0.1)],  # +0.1 for safety
            spread=0.0,
        )
        print(f"Configured tilt series: offset=[{offset_kx:.3f}, {offset_ky:.3f}] mrad "
              f"(|θ|={snapped_center_tilt:.3f} mrad, az={tilt_azimuth_deg:.2f}°), "
              f"window=[0, {radial_span/2.0+0.1:.1f}] mrad")
    except Exception as e:
        print(f"Warning: Could not create tilt_params ({e}). Continuing without explicit tilt.")

    kwargs = {
        "sample_specification": sample_model_params,
        "gun_model_params": gun_model_params,
        "lens_model_params": lens_model_params,
    }
    if tilt_params is not None:
        kwargs["tilt_params"] = tilt_params
    
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
    kwargs["max_data_size"] = 64_000_000_000  # 64 GB
    
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
            interpolation_factors=(4, 4),
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
            save_final_intensity=False,
            save_probe_complex=False,
            # Try to restrict to exit plane only to save space
            wavefunction_z_planes=[float(z_sampling)],
        )
    except Exception as e:
        print(f"Warning: Image params not applied: {e}")
        image_params = None
        
    return disc_params, image_params


def _read_prismatic_xyz(path: str):
    """Read a Prismatic-format .xyz: header line, 'Lx Ly Lz' line, then atom rows
    'Z x y z occ sigma', terminated by '-1'. Returns (header, [Lx,Ly,Lz], atoms_array).

    atoms_array shape (N, 6): columns Z, x, y, z, occ, sigma.
    """
    with open(path, "r") as f:
        header = f.readline().rstrip("\n")
        Lx, Ly, Lz = (float(v) for v in f.readline().split())
        rows = []
        for line in f:
            line = line.strip()
            if not line or line == "-1":
                break
            parts = line.split()
            if len(parts) < 6:
                continue
            rows.append([float(p) for p in parts[:6]])
    import numpy as _np
    return header, [Lx, Ly, Lz], _np.array(rows, dtype=float)


def _write_prismatic_xyz(path: str, header: str, dims, atoms_array):
    with open(path, "w") as f:
        f.write(header.rstrip("\n") + "\n")
        f.write(f"{dims[0]:.6f} {dims[1]:.6f} {dims[2]:.6f}\n")
        for row in atoms_array:
            Z = int(row[0])
            x, y, z, occ, sigma = row[1], row[2], row[3], row[4], row[5]
            f.write(f"{Z} {x:.6f} {y:.6f} {z:.6f} {occ:.3f} {sigma:.3f}\n")
        f.write("-1")


def _build_sample_tilt_variant(coords_file: str, meta: dict, output_dir: str) -> str | None:
    """
    Create a pre-rotated .xyz + meta.json that, when run with zero beam tilt,
    excites the same Bragg condition as a beam tilt of theta_B along the
    target azimuth. Returns the path to the new .xyz, or None on failure.

    Rotation: a sample tilt by theta_B around axis (sin(az), -cos(az), 0) in
    the slab frame is the inverse of a beam tilt by theta_B along
    (cos(az), sin(az)) within the small-angle approximation.
    """
    import numpy as _np
    theta_b_mrad = float(meta.get("target_theta_B_mrad", 0.0))
    az_deg = float(meta.get("tilt_azimuth_deg", 0.0))
    if theta_b_mrad <= 0.0:
        print("  [sample-tilt] meta is missing target_theta_B_mrad; skipping validation.")
        return None

    theta_rad = theta_b_mrad * 1e-3
    az_rad = math.radians(az_deg)
    # Rotation axis in the slab (x,y,z) frame:
    k = _np.array([math.sin(az_rad), -math.cos(az_rad), 0.0], dtype=float)
    nk = _np.linalg.norm(k)
    if nk == 0.0:
        k = _np.array([0.0, -1.0, 0.0])
    else:
        k /= nk

    # Rodrigues rotation matrix for angle theta_rad around k
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    K = _np.array([[0.0, -k[2], k[1]],
                   [k[2], 0.0, -k[0]],
                   [-k[1], k[0], 0.0]])
    R = _np.eye(3) + s * K + (1.0 - c) * (K @ K)

    # Load + rotate coordinates around the box centre so the slab stays in-cell.
    header, dims, atoms = _read_prismatic_xyz(coords_file)
    centre = _np.array([dims[0] / 2.0, dims[1] / 2.0, dims[2] / 2.0])
    pos = atoms[:, 1:4]
    rotated = (pos - centre) @ R.T + centre
    atoms[:, 1:4] = rotated

    # Write the pre-rotated xyz alongside a single-tilt meta.json.
    variant_dir = os.path.join(output_dir, "sample_tilt_validation")
    os.makedirs(variant_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(coords_file))[0]
    new_xyz = os.path.join(variant_dir, f"{base}_pretilted.xyz")
    _write_prismatic_xyz(
        new_xyz,
        header + " | pre-rotated for sample-tilt validation",
        dims,
        atoms,
    )

    new_meta = dict(meta)
    # Run at zero beam tilt; bake the sample rotation into the coordinates.
    new_meta["tilt_angles_mrad"] = [0.0]
    new_meta.pop("tilt_start_mrad", None)
    new_meta.pop("tilt_end_mrad", None)
    new_meta.pop("tilt_step_mrad", None)
    new_meta["sample_tilt_mrad"] = theta_b_mrad
    new_meta["sample_tilt_axis_slab"] = list(map(float, k.tolist()))
    new_meta["sample_tilt_origin"] = "pre-rotated for sample-tilt validation cross-check"
    with open(os.path.join(variant_dir, "meta.json"), "w") as f:
        json.dump(new_meta, f, indent=2)

    print(f"  [sample-tilt] wrote pre-rotated slab: {new_xyz}")
    print(f"  [sample-tilt] axis (slab frame): {k.tolist()}, angle = {theta_b_mrad:.3f} mrad")
    return new_xyz


def run_sim(coords_file: str, output_dir: str, timeout_min: int, production: bool, sample_tilt_validation: bool = False) -> bool:
    """Main simulation driver function with multi-angle tilt support."""
    print("="*70)
    print("REM HOLOGRAPHY SIMULATION - UNIFIED + SNAPPED (v9)")
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
    
    # Parse tilt angles (supports step-based, explicit array, or default [0.0])
    tilt_angles = parse_tilt_angles(meta)
    print(f"Tilt angles to simulate: {tilt_angles} mrad")
    print(f"Total number of angles: {len(tilt_angles)}\n")
    
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

    # Reverted to 4 as user requested to use more RAM
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

    print("="*70)
    print("UNIFIED MULTI-TILT SIMULATION")
    print("="*70)
    print(f"Running ONE simulation with {len(tilt_angles)} tilt angles")
    print(f"All angles will be output together in: {output_dir}\n")
    
    # Create discretization and image params
    disc_params, image_params = create_discretization_and_image_params(
        meta, Lx, Ly, Lz, gx, gy, dz_A, z_supersampling
    )
    
    # Tilt azimuth from meta (degrees from +k_x); set by the slab generator
    # for the target reflection. Si[110] cleave + (2,-2,0) -> ~35.26°.
    tilt_az_deg = float(meta.get("tilt_azimuth_deg", 0.0))

    # Create system params with ALL tilt angles AND cell dimensions for snapping
    system_params = create_system_params(
        coords_file,
        energy_keV,
        disc_params_obj=disc_params,
        tilt_angles_mrad=tilt_angles,
        cell_dim_A=[Lx, Ly, Lz],  # Passed for snapping
        tilt_azimuth_deg=tilt_az_deg,
    )
    if system_params is None:
        print("Failed to create system parameters.")
        return False
    
    # Create output params
    output_params = create_output_params(output_dir, image_params)
    if output_params is None:
        print("Failed to create output parameters.")
        return False
    
    # Create worker params
    worker_params = create_worker_params()
    
    # Build simulation parameters
    sim_params = prismatique.hrtem.sim.Params(
        hrtem_system_model_params=system_params,
        output_params=output_params,
        worker_params=worker_params,
    )
    
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

    print(f"\nStarting unified simulation...")
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
    print(f"Saved: intensity + complex exit wave (check {output_dir}).")

    # Optional cross-check: re-run once with pre-rotated coordinates at zero
    # beam tilt, to validate that the beam-tilt approximation agrees with the
    # sample-tilt result at theta_B.
    if sample_tilt_validation:
        print("\n" + "=" * 70)
        print("SAMPLE-TILT VALIDATION CROSS-CHECK")
        print("=" * 70)
        new_xyz = _build_sample_tilt_variant(coords_file, meta, output_dir)
        if new_xyz is not None:
            variant_out = os.path.join(output_dir, "sample_tilt_validation_output")
            print(f"  [sample-tilt] running forward sim in: {variant_out}\n")
            ok = run_sim(
                coords_file=new_xyz,
                output_dir=variant_out,
                timeout_min=timeout_min,
                production=production,
                sample_tilt_validation=False,  # never recurse
            )
            if not ok:
                print("  [sample-tilt] cross-check run failed.")

    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="REM Holography Simulation (Unified+Snapped)")
    parser.add_argument("coords_file", help="Path to .xyz file")
    parser.add_argument("output_dir", nargs="?", default="output", help="Output directory")
    parser.add_argument("--production", action="store_true", help="Production mode (logging)")
    parser.add_argument("--timeout-min", type=int, default=30, help="Timeout in minutes")
    parser.add_argument(
        "--sample-tilt-validation",
        action="store_true",
        help="After the main beam-tilt sweep, also run one forward simulation "
             "with the slab pre-rotated by theta_B (around (sin(az),-cos(az),0)) "
             "at zero beam tilt, as a cross-check against the small-angle "
             "beam-tilt approximation.",
    )

    args = parser.parse_args()

    success = run_sim(
        coords_file=args.coords_file,
        output_dir=args.output_dir,
        timeout_min=args.timeout_min,
        production=args.production,
        sample_tilt_validation=args.sample_tilt_validation,
    )
    
    sys.exit(0 if success else 1)
