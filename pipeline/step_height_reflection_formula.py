import math

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from scipy.signal import find_peaks

def relativistic_lambda_A(E_keV):
    """Calculates the relativistic electron wavelength in Angstroms."""
    h = 6.62607015e-34
    e = 1.602176634e-19
    m0 = 9.10938356e-31
    c = 299792458.0
    V = E_keV * 1e3 * e
    p = (2 * m0 * V * (1 + V / (2 * m0 * c * c))) ** 0.5
    return (h / p) * 1e10

def find_dominant_levels(data, bins=50):
    """Finds the two most dominant levels in the data using a histogram."""
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Find peaks in the histogram
    peaks, properties = find_peaks(hist, height=np.max(hist)*0.1, distance=5)
    
    if len(peaks) < 2:
        # Fallback: just use min and max percentiles if peaks aren't clear
        print("Warning: Could not find two distinct peaks. Using percentiles.")
        return np.percentile(data, 10), np.percentile(data, 90)
    
    # Sort peaks by height (prominence)
    sorted_indices = np.argsort(hist[peaks])[::-1]
    top_peaks = peaks[sorted_indices][:2]
    
    level1 = bin_centers[top_peaks[0]]
    level2 = bin_centers[top_peaks[1]]
    
    return min(level1, level2), max(level1, level2)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Measure step height from reconstructed phase.")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory containing results_arrays.npz")
    parser.add_argument("--meta", type=str, required=True, help="Path to meta.json")
    parser.add_argument("--output", type=str, default="step_height_measurement_mesa.png", help="Output PNG filename")
    
    args = parser.parse_args()

    NPZ_PATH = os.path.join(args.results_dir, "results_arrays.npz")
    META_PATH = args.meta

    # Load Data
    print(f"Loading results from {NPZ_PATH}...")
    try:
        data = np.load(NPZ_PATH)
        phase_flat = data['phase_hann_flat']
    except FileNotFoundError:
        print(f"Error: Could not find {NPZ_PATH}. Please run the filter script first.")
        return

    # Load Meta
    print(f"Loading metadata from {META_PATH}...")
    with open(META_PATH, 'r') as f:
        meta = json.load(f)
    
    E_keV = float(meta.get('energy_keV', 200.0))
    # Prefer the explicit Bragg angle from meta; fall back to alpha_deg for the
    # legacy Si(100) slabs that stored geometry as a single tilt angle.
    if 'target_theta_B_mrad' in meta:
        theta_rad = float(meta['target_theta_B_mrad']) * 1e-3
        alpha_deg = math.degrees(theta_rad)
    else:
        alpha_deg = float(meta.get('alpha_deg', 0.0))
        theta_rad = np.radians(alpha_deg)

    # Direction cosine between g_hat and surface normal. Defaults to 1.0
    # (symmetric Osakabe case, g parallel to n_surf); for asymmetric reflections
    # like Si(2,-2,0) off a (1,-1,1) facet it is ~0.8165. Without this factor
    # the height read-out is high by 1/(g_dot_n).
    g_dot_n = float(meta.get('g_dot_n_surf', 1.0))

    lam = relativistic_lambda_A(E_keV)

    print(f"Physics Parameters:")
    print(f"  Energy = {E_keV} keV")
    print(f"  Lambda = {lam:.5f} A")
    print(f"  Theta_B = {alpha_deg:.4f} deg ({theta_rad*1e3:.3f} mrad)")
    print(f"  g_hat . n_surf = {g_dot_n:.4f}")

    # --- Analysis ---
    ny, nx = phase_flat.shape
    
    # Automatic Step Detection
    # We look for the profile (row or column) with the highest variance (contrast)
    # This assumes the step creates a bimodal distribution (high variance)
    
    # Scan horizontal profiles (along X, varying Y)
    # We use a sliding window of width 20 to average noise
    width = 10
    
    best_var_x = -1
    best_y = -1
    
    # Sample every 5th row to save time
    for y in range(width, ny-width, 5):
        prof = np.nanmean(phase_flat[y-width:y+width, :], axis=0)
        prof = prof[np.isfinite(prof)]
        if len(prof) > 10:
            var = np.var(prof)
            if var > best_var_x:
                best_var_x = var
                best_y = y
                
    # Scan vertical profiles (along Y, varying X)
    best_var_y = -1
    best_x = -1
    for x in range(width, nx-width, 5):
        prof = np.nanmean(phase_flat[:, x-width:x+width], axis=1)
        prof = prof[np.isfinite(prof)]
        if len(prof) > 10:
            var = np.var(prof)
            if var > best_var_y:
                best_var_y = var
                best_x = x
    
    print(f"\nAutomatic Detection:")
    print(f"  Max Variance Horizontal (at Y={best_y}): {best_var_x:.4f}")
    print(f"  Max Variance Vertical   (at X={best_x}): {best_var_y:.4f}")
    
    # Decide which direction has the step
    if best_var_x > best_var_y:
        print("  -> Detected Step running Horizontally (profile along X)")
        direction = "horizontal"
        profile = np.nanmean(phase_flat[best_y-width:best_y+width, :], axis=0)
        loc_str = f"Y={best_y}"
    else:
        print("  -> Detected Step running Vertically (profile along Y)")
        direction = "vertical"
        profile = np.nanmean(phase_flat[:, best_x-width:best_x+width], axis=1)
        loc_str = f"X={best_x}"
        
    profile = profile[np.isfinite(profile)]
    
    if len(profile) == 0:
        print("Error: Best profile is empty.")
        return

    # Identify levels
    level_low, level_high = find_dominant_levels(profile, bins=60)
    delta_phi = level_high - level_low
    # Projected Osakabe formula: h = Δφ λ / (4π sin θ_B (g_hat · n_surf))
    # The g·n factor accounts for asymmetric reflections (it is 1 only when
    # g is parallel to the surface normal).
    denom = 4 * np.pi * np.sin(theta_rad) * g_dot_n
    h_A = delta_phi * lam / denom if denom != 0.0 else float('nan')

    # Theoretical step height: prefer meta['step_height_A'], else d_111 of Si.
    a_Si = float(meta.get('a_A', 5.4309))
    h_theo = float(meta.get('step_height_A') or (a_Si / math.sqrt(3.0)))
    dphi_theo = 4 * np.pi * h_theo * np.sin(theta_rad) * g_dot_n / lam

    print(f"\nMeasurements (Auto-Found at {loc_str}):")
    print(f"  Level Low : {level_low:.3f} rad")
    print(f"  Level High: {level_high:.3f} rad")
    print(f"  Delta Phi : {delta_phi:.3f} rad")
    print(f"  Calculated Step Height: {h_A:.3f} A")
    print(f"  Theoretical Step (meta): {h_theo:.3f} A")
    print(f"  Theoretical Delta Phi: {dphi_theo:.3f} rad")
    
    # --- Plotting ---
    plt.figure(figsize=(10, 8))
    
    # Subplot 1: Profile
    plt.subplot(2, 1, 1)
    plt.plot(profile, label=f'{direction.title()} Profile at {loc_str}', color='blue')
    plt.axhline(level_high, color='red', linestyle='--', label=f'High: {level_high:.1f}')
    plt.axhline(level_low, color='green', linestyle='--', label=f'Low: {level_low:.1f}')
    plt.title(f"Phase Profile ({direction.title()} at {loc_str})\nHeight: {h_A:.2f} A")
    plt.ylabel("Phase (rad)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Histogram
    plt.subplot(2, 1, 2)
    plt.hist(profile, bins=60, color='gray', alpha=0.7, label='Phase Dist')
    plt.axvline(level_high, color='red', linestyle='--')
    plt.axvline(level_low, color='green', linestyle='--')
    plt.title("Phase Histogram")
    plt.xlabel("Phase (rad)")
    plt.ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"\nSaved plot to {args.output}")

if __name__ == "__main__":
    main()
