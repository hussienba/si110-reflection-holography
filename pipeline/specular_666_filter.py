
#!/usr/bin/env python3
"""
specular_666_filter.py
----------------------
Fourier-select a single diffraction spot (specified by --meta or --target-radius,
default Si(2,-2,0) on the Si[110] cleave-edge slab) from a complex exit wave
stored in HDF5 and reconstruct a "virtual dark-field" wave using both Hann and
Hamming raised-cosine masks.

This script performs the following steps:
1.  **Load**: Reads a complex 2D wavefunction from an HDF5 file.
2.  **Pre-process**: Applies real-space apodization (Tukey window) and zero-padding.
3.  **FFT**: Computes the 2D Fast Fourier Transform to move to reciprocal space (k-space).
4.  **(000) suppression**: Applies a super-Gaussian k-space notch to remove the
    direct-beam contribution before sideband selection. This is the
    signal-processing analogue of physically absorbing the transmitted beam.
5.  **Peak Finding**: Identifies the target diffraction spot (sideband).
6.  **Filter & Reconstruct**:
    - Applies a soft circular mask (Hann/Hamming) around the selected peak.
    - Shifts the selected peak to the center of k-space (removing the carrier frequency).
    - Performs Inverse FFT (IFFT) to obtain the complex "virtual dark-field" image.
7.  **Post-process**:
    - Unwraps the phase of the reconstructed wave.
    - Detrends the phase (removes linear phase ramps).
8.  **Output**: Saves results as PNG images, NPZ arrays, and an HDF5 file.

INPUT:
  - HDF5 file with either:
      * complex dataset 'psi'  (shape: [Ny, Nx], dtype complex)
    OR
      * real datasets 'amplitude' and 'phase' (radians), same shape
  - Optional --meta meta.json: when given, target_radius (= 1/target_d_A) is
    read from meta and the manual --target-radius flag is unnecessary.

OUTPUTS (prefix derived from input filename unless --out-prefix given):
  - PNGs: k-space magnitude, masks, reconstructed amplitude/phase (for Hann/Hamming)
  - NPZ: arrays with amplitude and phase (wrapped and flattened)
  - HDF5: '/hann/amp', '/hann/phase', '/hamming/amp', '/hamming/phase'

USAGE (typical):
  python specular_666_filter.py exit_wave.h5 --meta meta.json --dx 0.5 --dy 0.5 --rin 8 --rout 14
"""

import argparse
import os
import json
import logging
from typing import Tuple, Optional, Any, List, Dict

import h5py
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ----------------- Utilities -----------------

def _pick_first_2d_slice(dset: h5py.Dataset) -> np.ndarray:
    """
    Return the first 2D slice from an HDF5 dataset by fixing all leading
    dimensions to 0. This matches how the Prismatique/HR-TEM outputs store
    wavefunctions, e.g. 'data/image_wavefunctions'.
    
    Args:
        dset: The HDF5 dataset to slice.
        
    Returns:
        A 2D numpy array (complex or real).
    """
    if dset.ndim <= 2:
        return dset[()]
    # Create a tuple of 0s for the leading dimensions
    idx = (0,) * (dset.ndim - 2) + (slice(None), slice(None))
    return dset[idx]


def load_complex_h5(path: str) -> np.ndarray:
    """
    Load a complex 2D wavefunction ψ(y,x) from common HDF5 layouts.

    Tries, in order:
      1) 'data/image_wavefunctions' (Prismatique / HR-TEM output; first 2D slice)
      2) root dataset 'psi' (complex 2D)
      3) root datasets 'amplitude' and 'phase' (combined to complex)
      4) fallback: first complex dataset found anywhere (preferring names with 'wave')
      
    Args:
        path: Path to the HDF5 file.
        
    Returns:
        A 2D complex numpy array representing the wavefunction.
        
    Raises:
        ValueError: If found datasets are not complex or compatible.
        KeyError: If no suitable dataset is found.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    with h5py.File(path, "r") as f:
        # 1) Prismatique / HR-TEM layout
        if "data/image_wavefunctions" in f:
            logging.info("Found 'data/image_wavefunctions'.")
            dset = f["data/image_wavefunctions"]
            psi = _pick_first_2d_slice(dset)
            if not np.iscomplexobj(psi):
                raise ValueError("'data/image_wavefunctions' exists but is not complex.")
            return np.asarray(psi)

        # 2) Simple complex dataset at root
        if "psi" in f:
            logging.info("Found 'psi' dataset.")
            psi = f["psi"][()]
            if not np.iscomplexobj(psi):
                raise ValueError("'psi' exists but is not complex.")
            return psi

        # 3) Amplitude / phase pair at root
        if "amplitude" in f and "phase" in f:
            logging.info("Found 'amplitude' and 'phase' datasets.")
            amp = f["amplitude"][()]
            ph  = f["phase"][()]
            psi = amp * np.exp(1j * ph)
            return psi

        # 4) Fallback: search for any complex dataset in the file
        logging.info("Searching for any complex dataset...")
        candidates: List[Tuple[str, np.ndarray]] = []

        def _visitor(name: str, obj: Any):
            if isinstance(obj, h5py.Dataset):
                # Read the data to check type (careful with large files, but usually wavefunctions fit in RAM)
                # For very large files, check dtype first if possible, but h5py dtype is complex128 etc.
                if np.issubdtype(obj.dtype, np.complexfloating):
                     arr = obj[...]
                     candidates.append((name, arr))

        f.visititems(_visitor)

        if candidates:
            # Prefer shallower paths and names containing "wave"
            candidates.sort(key=lambda na: (len(na[0].split("/")), "wave" not in na[0]))
            name, arr = candidates[0]
            logging.info(f"Selected fallback dataset: '{name}'")
            psi = np.squeeze(arr)
            if psi.ndim != 2:
                 # Try to take a slice if it's higher dim
                 if psi.ndim > 2:
                     psi = _pick_first_2d_slice(psi) # This won't work directly on array, need logic
                     # Re-implement slice logic for array
                     idx = (0,) * (psi.ndim - 2) + (slice(None), slice(None))
                     psi = psi[idx]
            return psi

        raise KeyError(
            f"Could not find a complex wavefunction dataset in file. "
            f"Top-level keys: {list(f.keys())}"
        )

def tukey2d(ny: int, nx: int, alpha: float = 0.2) -> np.ndarray:
    """
    Generate a 2D Tukey window (tapered cosine).
    
    Args:
        ny: Height of the window.
        nx: Width of the window.
        alpha: Shape parameter. 0 -> Rectangular, 1 -> Hann.
        
    Returns:
        2D numpy array with the window.
    """
    def tukey(n: int, a: float) -> np.ndarray:
        if a <= 0: return np.ones(n)
        if a >= 1: a = 1
        w = np.ones(n)
        edge = int(a * (n - 1) / 2)
        if edge > 0:
            x = np.linspace(0, np.pi, edge, endpoint=False)
            rise = 0.5 * (1 - np.cos(x))
            w[:edge]  = rise
            w[-edge:] = rise[::-1]
        return w
    return np.outer(tukey(ny, alpha), tukey(nx, alpha))

def raised_cosine_disk(shape: Tuple[int, int], center: Tuple[int, int], 
                       R_in: float, R_out: float, window: str = "hann") -> np.ndarray:
    """
    Create a 2D soft circular mask (filter).
    
    The mask is 1.0 inside R_in, and tapers to 0.0 at R_out using a raised cosine profile.
    
    Args:
        shape: (ny, nx) of the array.
        center: (cy, cx) center of the mask.
        R_in: Inner radius (passband) in pixels.
        R_out: Outer radius (stopband start) in pixels.
        window: Taper shape, "hann" or "hamming".
        
    Returns:
        2D numpy array (float) with values in [0, 1].
    """
    ny, nx = shape
    cy, cx = center
    y = np.arange(ny) - cy
    x = np.arange(nx) - cx
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)
    
    m = np.zeros_like(r, dtype=float)
    inside = r <= R_in
    taper  = (r > R_in) & (r < R_out)
    
    m[inside] = 1.0
    
    if np.any(taper):
        # Normalized distance in the taper region (0 at R_in, 1 at R_out)
        t = (r[taper] - R_in) / (R_out - R_in)
        
        if window.lower().startswith("ham"):
            # Hamming window: 0.54 - 0.46 * cos(...)
            # Here we want it to go from 1 down to 0.
            # Standard Hamming is on [-pi, pi] or [0, 2pi].
            # We map t=[0,1] to the falling edge of the window.
            # A standard Hamming doesn't go exactly to zero.
            # We scale it to force it to 0 at the edge for a true mask.
            a = 0.54
            # w ranges from 1 (at t=0) to b (at t=1)
            w_raw = a + (1 - a) * np.cos(np.pi * t) 
            b = 2 * a - 1  # Value at t=1 (approx 0.08)
            # Normalize so it hits exactly 0 at R_out
            w = (w_raw - b) / (1 - b)
        else:
            # Hann window: 0.5 * (1 + cos(...))
            # Ranges from 1 (at t=0) to 0 (at t=1)
            w = 0.5 * (1 + np.cos(np.pi * t))
            
        w = np.clip(w, 0.0, 1.0)
        m[taper] = w
        
    return m

def roll2(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """
    Circularly shift a 2D array.
    
    Args:
        a: Input 2D array.
        dy: Shift in y (rows).
        dx: Shift in x (columns).
        
    Returns:
        Shifted array.
    """
    return np.roll(np.roll(a, dy, axis=0), dx, axis=1)

def unwrap2d(phase: np.ndarray) -> np.ndarray:
    """
    Unwrap 2D phase array.
    
    Args:
        phase: Wrapped phase in radians.
        
    Returns:
        Unwrapped phase.
    """
    return np.unwrap(np.unwrap(phase, axis=0), axis=1)

def detrend_plane(phi: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    """
    Remove a linear phase ramp (plane) from the phase image using least squares.
    phi(y,x) ~ a*x + b*y + c
    
    Args:
        phi: 2D phase array.
        
    Returns:
        Tuple of (detrended_phase, [a, b, c]).
    """
    ny, nx = phi.shape
    Y, X = np.mgrid[0:ny, 0:nx]
    # Flatten arrays for regression
    # We solve A * [a, b, c]^T = phi
    A = np.c_[X.ravel(), Y.ravel(), np.ones(phi.size)]
    c, *_ = np.linalg.lstsq(A, phi.ravel(), rcond=None)
    
    plane = (c[0]*X + c[1]*Y + c[2])
    return phi - plane, list(c)

def save_png(fn: str, arr: np.ndarray, cmap: str = "gray", 
             vmin: Optional[float] = None, vmax: Optional[float] = None,
             title: Optional[str] = None, pixel_size: Optional[float] = None,
             unit: str = "px", colorbar_label: Optional[str] = None) -> None:
    """
    Save a 2D array as a PNG image with scientific annotations.
    """
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    
    ny, nx = arr.shape
    extent = None
    if pixel_size is not None:
        # extent = [left, right, bottom, top]
        # Assuming center is (0,0) for k-space if shifted, or just 0..L for real space
        # For simplicity in this script, let's stick to 0..L for real space
        # and centered for k-space if implied by the caller.
        # But here we just map pixels to physical units starting at 0.
        extent = [0, nx * pixel_size, ny * pixel_size, 0]

    plt.figure(figsize=(8, 6), dpi=150)
    
    im = plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
    
    if title:
        plt.title(title)
        
    if pixel_size:
        plt.xlabel(f"x ({unit})")
        plt.ylabel(f"y ({unit})")
    else:
        plt.axis("off")
        
    if colorbar_label:
        cbar = plt.colorbar(im)
        cbar.set_label(colorbar_label)
        
    plt.tight_layout()
    plt.savefig(fn, bbox_inches="tight")
    plt.close()

# ----------------- Core pipeline -----------------

def find_peak(Psi: np.ndarray, 
              exclude_center_px: int = 20, 
              K: Optional[np.ndarray] = None, 
              target_radius: Optional[float] = None, 
              ring_frac: float = 0.03, 
              manual: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    """
    Find the brightest non-central peak in the magnitude of the FFT |Psi|.
    
    Args:
        Psi: Complex FFT array (shifted, so DC is at center).
        exclude_center_px: Radius in pixels to mask out around the DC component.
        K: 2D array of spatial frequencies (cycles/Å). Required if target_radius is used.
        target_radius: Expected |k| of the target reflection (cycles/Å).
        ring_frac: Fractional tolerance for the target radius.
        manual: Optional (iy, ix) tuple to force a specific peak location.
        
    Returns:
        (iy, ix) indices of the peak.
    """
    mag = np.abs(Psi).astype(np.float64)
    ny, nx = mag.shape
    cy, cx = ny // 2, nx // 2

    if manual is not None:
        iy, ix = manual
        logging.info(f"Using manual peak selection: ({iy}, {ix})")
        return int(iy), int(ix)

    # Create a working copy to mask out unwanted regions
    mag2 = mag.copy()
    
    # Mask out the central beam (DC component)
    # It's usually the brightest, so we must suppress it to find diffraction spots.
    mag2[cy-exclude_center_px : cy+exclude_center_px, 
         cx-exclude_center_px : cx+exclude_center_px] = 0.0

    # If a target radius (scattering vector length) is specified, mask everything else
    if target_radius is not None and K is not None:
        ring_tol = ring_frac * target_radius
        ring_mask = (np.abs(K - target_radius) <= ring_tol)
        mag2[~ring_mask] = 0.0
        logging.info(f"Searching for peak within |k| = {target_radius} ± {ring_tol*100/target_radius:.1f}%")

    # Find the max value index
    iy, ix = np.unravel_index(np.argmax(mag2), mag2.shape)
    logging.info(f"Found peak at ({iy}, {ix}) with magnitude {mag2[iy, ix]:.2e}")
    return int(iy), int(ix)

def k_axes(ny: int, nx: int, dy: float, dx: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate frequency axes for the FFT.
    
    Args:
        ny, nx: Dimensions of the array.
        dy, dx: Real-space sampling intervals (Å/pixel).
        
    Returns:
        (KX, KY) meshgrid of frequencies (cycles/Å), shifted to center.
    """
    # fftfreq returns [0, 1, ..., -n/2, ...], fftshift moves 0 to center
    ky = fftshift(fftfreq(ny, d=dy))
    kx = fftshift(fftfreq(nx, d=dx))
    return np.meshgrid(kx, ky)

def reconstruct_for_window(Psi: np.ndarray, 
                           center_idx: Tuple[int, int], 
                           rin: int, 
                           rout: int, 
                           window_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct the real-space image from a specific diffraction spot.
    
    Method:
    1. Create a mask centered at the chosen peak `center_idx`.
    2. Multiply FFT `Psi` by this mask.
    3. Shift the masked array so the peak moves to the DC (center) position.
       - This demodulates the carrier frequency, resulting in the complex envelope
         (amplitude and phase) of that specific periodicity.
    4. Inverse FFT.
    
    Args:
        Psi: Centered FFT array.
        center_idx: (iy, ix) of the peak to filter.
        rin: Passband radius (px).
        rout: Stopband radius (px).
        window_name: "hann" or "hamming".
        
    Returns:
        (psi_rec, mask) where psi_rec is the complex reconstructed wave.
    """
    ny, nx = Psi.shape
    cy, cx = ny // 2, nx // 2
    
    # 1. Create mask at the peak location
    mask = raised_cosine_disk(Psi.shape, center_idx, rin, rout, window=window_name)
    
    # 2. Apply mask
    Psi_masked = Psi * mask
    
    # 3. Shift the peak to the center (DC)
    #    We want the pixel at center_idx (iy0, ix0) to move to (cy, cx).
    #    Shift amount: dy = cy - iy0, dx = cx - ix0
    iy0, ix0 = center_idx
    dy = cy - iy0
    dx = cx - ix0
    
    # roll2 performs a circular shift. Since we padded the image, edge effects 
    # should be minimal if the padding was sufficient.
    Psi_centered = roll2(Psi_masked, dy, dx)
    
    # 4. Inverse FFT
    #    ifftshift is needed because Psi_centered has DC at (cy, cx), 
    #    but standard IFFT expects DC at (0, 0).
    psi_rec = ifft2(ifftshift(Psi_centered))
    
    return psi_rec, mask

def main():
    parser = argparse.ArgumentParser(description="Filter and reconstruct from a specific diffraction spot.")
    parser.add_argument("input", type=str, help="Input .h5 file with 'psi' or ('amplitude','phase')")
    parser.add_argument("--out-prefix", type=str, default=None, help="Prefix for output files")
    parser.add_argument("--dx", type=float, default=0.5, help="Sampling in x (Å/px)")
    parser.add_argument("--dy", type=float, default=0.5, help="Sampling in y (Å/px)")
    parser.add_argument("--pad", type=int, default=2, help="Zero-padding factor (e.g., 2 for 2x size)")
    parser.add_argument("--tukey-alpha", type=float, default=0.3, help="Tukey window alpha (0=rect, 1=Hann)")
    parser.add_argument("--exclude-center", type=int, default=20, help="Pixels to ignore around DC for peak finding")
    parser.add_argument("--target-radius", type=float, default=None, help="Expected |k| (cycles/Å) of target spot. If omitted and --meta given, read 1/target_d_A from meta.")
    parser.add_argument("--meta", type=str, default=None, help="Path to meta.json to read target_d_A (sets target-radius automatically)")
    parser.add_argument("--ring-frac", type=float, default=0.03, help="Tolerance fraction for target radius")
    parser.add_argument("--manual-peak", type=str, default=None, help="Force peak at 'iy,ix' (padded coords)")
    parser.add_argument("--rin", type=int, default=8, help="Mask inner radius (px)")
    parser.add_argument("--rout", type=int, default=14, help="Mask outer radius (px)")
    parser.add_argument("--no-unwrap", action="store_true", help="Skip phase unwrapping")
    parser.add_argument("--no-detrend", action="store_true", help="Skip phase detrending")
    parser.add_argument("--notch-sigma-inv-A", type=float, default=0.04,
                        help="Super-Gaussian (000) notch width (cycles/Å). Default 0.04. Set <=0 to disable.")
    parser.add_argument("--notch-order", type=int, default=4,
                        help="Super-Gaussian notch order n (sharper roll-off for larger n). Default 4.")

    args = parser.parse_args()

    # If meta was given and target-radius was not, derive target-radius from meta
    if args.target_radius is None and args.meta:
        try:
            with open(args.meta, "r") as _f:
                _meta = json.load(_f)
            _d = _meta.get("target_d_A")
            if _d:
                args.target_radius = 1.0 / float(_d)
                logging.info(f"target-radius set from meta: 1/{_d:.4f} A = {args.target_radius:.4f} cycles/A")
        except Exception as _exc:
            logging.warning(f"Could not derive target-radius from --meta: {_exc}")
    
    inpath = args.input
    if args.out_prefix:
        out_prefix = args.out_prefix
    else:
        out_prefix = os.path.splitext(os.path.basename(inpath))[0]
        
    outdir = out_prefix + "_bragg_filter"
    os.makedirs(outdir, exist_ok=True)
    logging.info(f"Output directory: {outdir}")

    # --- 1. Load Data ---
    try:
        psi0 = load_complex_h5(inpath)
    except Exception as e:
        logging.error(f"Failed to load input file: {e}")
        return

    Ny, Nx = psi0.shape
    logging.info(f"Loaded wavefunction: {Ny}x{Nx}")

    # --- 2. Pre-processing (Apodization + Padding) ---
    # Apodization reduces edge artifacts in FFT
    apo = tukey2d(Ny, Nx, alpha=args.tukey_alpha)
    psi_apod = psi0 * apo

    # Zero-padding improves k-space sampling density
    pad_factor = max(1, int(args.pad))
    pad_y = Ny * (pad_factor - 1) // 2
    pad_x = Nx * (pad_factor - 1) // 2
    psi_pad = np.pad(psi_apod, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant")
    
    logging.info(f"Padded shape: {psi_pad.shape} (Factor: {pad_factor})")

    # --- 3. FFT ---
    # fftshift moves DC to the center of the image
    Psi = fftshift(fft2(psi_pad))
    Ny2, Nx2 = Psi.shape

    # Calculate frequency axes
    KX, KY = k_axes(Ny2, Nx2, args.dy, args.dx)
    K = np.sqrt(KX**2 + KY**2)

    # Save raw (pre-notch) k-space magnitude for inspection
    save_png(os.path.join(outdir, "kspace_mag_raw.png"),
             np.log1p(np.abs(Psi)), cmap="gray",
             title="Log |FFT| (pre (000) notch)", colorbar_label="Log Intensity")

    # --- 3b. Super-Gaussian (000) notch ---
    # Suppress the transmitted (direct) beam before sideband selection. With a
    # symmetric small slab we do not absorb the direct beam physically, so we
    # remove it here in k-space. The notch must be smoothly tapered (super-
    # Gaussian) to avoid Gibbs ringing under the subsequent IFFT.
    notch_sigma = float(args.notch_sigma_inv_A)
    if notch_sigma > 0.0:
        n_order = int(max(1, args.notch_order))
        notch = 1.0 - np.exp(-(K / notch_sigma) ** (2 * n_order))
        Psi = Psi * notch
        logging.info(f"Applied (000) super-Gaussian notch: sigma={notch_sigma:.4f} cycles/A, order={n_order}")
        save_png(os.path.join(outdir, "notch.png"), notch, cmap="gray",
                 title=f"(000) notch (σ={notch_sigma:.3f} cyc/Å, n={n_order})",
                 colorbar_label="transmission")
    else:
        logging.info("(000) notch disabled (sigma <= 0).")

    # Save k-space magnitude (post-notch) for inspection
    # Note: K-space axes are frequency, not space. We don't pass pixel_size here in the same way.
    # We could construct extent manually from KX, KY limits.
    # Actually imshow extent is [left, right, bottom, top]. KY.min is negative (top in freq?), KY.max positive.
    # Standard image origin is top-left.
    # Let's just label it "pixels" for k-space or skip physical units for now to avoid confusion,
    # or implement proper k-extent.
    save_png(os.path.join(outdir, "kspace_mag.png"),
             np.log1p(np.abs(Psi)), cmap="gray",
             title="Log Magnitude of FFT", colorbar_label="Log Intensity")

    # --- 4. Peak Selection ---
    manual_peak = None
    if args.manual_peak:
        try:
            iy, ix = map(int, args.manual_peak.split(","))
            manual_peak = (iy, ix)
        except ValueError:
            logging.error("Bad --manual-peak format. Use 'iy,ix'.")
            return

    iy0, ix0 = find_peak(Psi,
                         exclude_center_px=args.exclude_center,
                         K=K,
                         target_radius=args.target_radius,
                         ring_frac=args.ring_frac,
                         manual=manual_peak)

    # Visual check: mark the selected peak on the k-space image
    kvis = np.log1p(np.abs(Psi))
    mark = kvis.copy()
    rmark = 6
    yy, xx = np.ogrid[:Ny2, :Nx2]
    circ = (yy - iy0)**2 + (xx - ix0)**2 <= rmark**2
    mark[circ] = mark.max() # Draw a bright spot
    save_png(os.path.join(outdir, "kspace_mag_marked.png"), mark, cmap="gray",
             title=f"Peak Selection at ({iy0}, {ix0})")

    # --- 5. Reconstruction ---
    # We use two different window shapes for comparison
    
    # Hann Window
    psi_hann, mask_hann = reconstruct_for_window(Psi, (iy0, ix0), args.rin, args.rout, "hann")
    amp_hann = np.abs(psi_hann)
    ph_hann  = np.angle(psi_hann)

    # Hamming Window
    psi_hamm, mask_hamm = reconstruct_for_window(Psi, (iy0, ix0), args.rin, args.rout, "hamming")
    amp_hamm = np.abs(psi_hamm)
    ph_hamm  = np.angle(psi_hamm)

    # Save mask visualizations
    save_png(os.path.join(outdir, "mask_hann.png"), mask_hann, 
             title="Hann Mask", colorbar_label="Transmission")
    save_png(os.path.join(outdir, "mask_hamming.png"), mask_hamm,
             title="Hamming Mask", colorbar_label="Transmission")

    # --- 6. Post-processing (Unwrap + Detrend) ---
    
    meta = {
        "input": inpath, "Ny": Ny, "Nx": Nx, "pad": pad_factor,
        "dx_A_per_px": args.dx, "dy_A_per_px": args.dy,
        "iy_ix_peak": [int(iy0), int(ix0)],
        "rin_px": args.rin, "rout_px": args.rout,
        "tukey_alpha": args.tukey_alpha,
        "amplitude_threshold_fraction": 0.1
    }

    # Create amplitude masks (threshold at 10% of max amplitude) to avoid unwrapping noise
    amp_threshold_hann = 0.1 * amp_hann.max()
    amp_threshold_hamm = 0.1 * amp_hamm.max()
    mask_valid_hann = amp_hann > amp_threshold_hann
    mask_valid_hamm = amp_hamm > amp_threshold_hamm

    # Unwrapping
    if not args.no_unwrap:
        logging.info("Unwrapping phase...")
        ph_hann_u = np.full_like(ph_hann, np.nan)
        ph_hamm_u = np.full_like(ph_hamm, np.nan)
        
        if mask_valid_hann.any():
            # Only unwrap valid pixels. We replace invalid ones with 0 for the unwrap function,
            # then mask them back to NaN.
            ph_hann_u[mask_valid_hann] = unwrap2d(np.where(mask_valid_hann, ph_hann, 0))[mask_valid_hann]
            
        if mask_valid_hamm.any():
            ph_hamm_u[mask_valid_hamm] = unwrap2d(np.where(mask_valid_hamm, ph_hamm, 0))[mask_valid_hamm]
    else:
        ph_hann_u = ph_hann.copy()
        ph_hamm_u = ph_hamm.copy()
        ph_hann_u[~mask_valid_hann] = np.nan
        ph_hamm_u[~mask_valid_hamm] = np.nan

    # Detrending
    if not args.no_detrend:
        logging.info("Detrending phase...")
        ph_hann_flat = np.full_like(ph_hann_u, np.nan)
        ph_hamm_flat = np.full_like(ph_hamm_u, np.nan)
        
        # Helper to detrend masked array
        def apply_detrend(ph_in, mask_in):
            if not mask_in.any():
                return ph_in, None
            
            # We need to pass the full array to detrend_plane, but it fits a plane to the whole thing.
            # Ideally we only fit the plane to the valid pixels.
            # The existing detrend_plane function fits to ALL pixels. 
            # Let's do a masked fit here inline or modify the helper.
            # Modified approach: fit only valid pixels.
            
            ny, nx = ph_in.shape
            Y, X = np.mgrid[0:ny, 0:nx]
            valid = mask_in & np.isfinite(ph_in)
            
            if not valid.any():
                return ph_in, None

            # Solve A*c = z for valid points
            A = np.c_[X[valid], Y[valid], np.ones(valid.sum())]
            z = ph_in[valid]
            c, *_ = np.linalg.lstsq(A, z, rcond=None)
            
            # Construct plane for whole image
            plane = c[0]*X + c[1]*Y + c[2]
            return ph_in - plane, c

        ph_hann_flat, c_hann = apply_detrend(ph_hann_u, mask_valid_hann)
        if c_hann is not None:
            meta["plane_hann_abc"] = list(map(float, c_hann))
            
        ph_hamm_flat, c_hamm = apply_detrend(ph_hamm_u, mask_valid_hamm)
        if c_hamm is not None:
            meta["plane_hamming_abc"] = list(map(float, c_hamm))
            
        # Re-mask NaNs
        ph_hann_flat[~mask_valid_hann] = np.nan
        ph_hamm_flat[~mask_valid_hamm] = np.nan
    else:
        ph_hann_flat = ph_hann_u.copy()
        ph_hamm_flat = ph_hamm_u.copy()

    # --- 7. Save Results ---
    
    # Amplitudes: scale to 99th percentile for better contrast
    for name, amp in [("hann", amp_hann), ("hamming", amp_hamm)]:
        p99 = np.percentile(amp, 99.0)
        save_png(os.path.join(outdir, f"amp_{name}.png"), amp, vmin=0, vmax=p99,
                 title=f"Amplitude ({name.title()})", pixel_size=args.dx, unit="A", colorbar_label="Amplitude")

    # Wrapped Phases
    save_png(os.path.join(outdir, "phase_hann_wrapped.png"), ph_hann, cmap="twilight",
             title="Wrapped Phase (Hann)", pixel_size=args.dx, unit="A", colorbar_label="Phase (rad)")
    save_png(os.path.join(outdir, "phase_hamming_wrapped.png"), ph_hamm, cmap="twilight",
             title="Wrapped Phase (Hamming)", pixel_size=args.dx, unit="A", colorbar_label="Phase (rad)")

    # Flattened Phases: center around zero and clip
    def viz_phase(phi, clip=np.pi):
        # Center the median of the valid pixels to 0
        m = np.nanmedian(phi)
        if np.isnan(m): m = 0
        return np.clip(phi - m, -clip, clip)

    save_png(os.path.join(outdir, "phase_hann_flat.png"), 
             viz_phase(ph_hann_flat), cmap="twilight", vmin=-np.pi, vmax=np.pi,
             title="Flat Phase (Hann)", pixel_size=args.dx, unit="A", colorbar_label="Phase (rad)")
    save_png(os.path.join(outdir, "phase_hamming_flat.png"), 
             viz_phase(ph_hamm_flat), cmap="twilight", vmin=-np.pi, vmax=np.pi,
             title="Flat Phase (Hamming)", pixel_size=args.dx, unit="A", colorbar_label="Phase (rad)")

    # Save NPZ
    np.savez_compressed(os.path.join(outdir, "results_arrays.npz"),
                        amp_hann=amp_hann, phase_hann_wrapped=ph_hann, phase_hann_flat=ph_hann_flat,
                        amp_hamming=amp_hamm, phase_hamming_wrapped=ph_hamm, phase_hamming_flat=ph_hamm_flat,
                        mask_hann=mask_hann, mask_hamming=mask_hamm,
                        mask_valid_hann=mask_valid_hann, mask_valid_hamm=mask_valid_hamm,
                        iy0=int(iy0), ix0=int(ix0))

    # Save HDF5
    with h5py.File(os.path.join(outdir, "results.h5"), "w") as f:
        g1 = f.create_group("hann")
        g1.create_dataset("amplitude", data=amp_hann, compression="gzip")
        g1.create_dataset("phase_wrapped", data=ph_hann, compression="gzip")
        g1.create_dataset("phase_flat", data=ph_hann_flat, compression="gzip")
        g1.create_dataset("mask_kspace", data=mask_hann, compression="gzip")
        
        g2 = f.create_group("hamming")
        g2.create_dataset("amplitude", data=amp_hamm, compression="gzip")
        g2.create_dataset("phase_wrapped", data=ph_hamm, compression="gzip")
        g2.create_dataset("phase_flat", data=ph_hamm_flat, compression="gzip")
        g2.create_dataset("mask_kspace", data=mask_hamm, compression="gzip")
        
        f.attrs["meta"] = json.dumps(meta)

    logging.info(f"Success! Peak at (iy,ix)=({iy0},{ix0}) on padded grid {Ny2}x{Nx2}")
    logging.info(f"Results saved to: {outdir}/")

if __name__ == "__main__":
    main()
