#!/usr/bin/env python3
"""
Si[110] cleave-edge slab generator for reflection electron holography.

Geometry (slab frame):
    z = beam direction        = [110]/sqrt(2)        (length per period: a/sqrt(2) = 3.840 A)
    x = surface normal        = [1, -1, 1]/sqrt(3)   (length per period: a*sqrt(3) = 9.408 A)
    y = step-normal-in-plane  = [1, -1, -2]/sqrt(6)  (length per period: a*sqrt(3/2) = 6.652 A)

The (1,-1,1) facet is the "top" of the slab; the step edge runs parallel to the
beam (z) and the surface height jumps by d_{111} = a/sqrt(3) ≈ 3.136 A as one
crosses y = L_y/2. This is the simulation analogue of the cleaved-edge {111}
facet seen in transmission with B = [110].

Primary target reflection: Si(2,-2,0). At 200 keV, theta_B = 6.53 mrad.
The (2,-2,0) reciprocal vector projects into the (k_x, k_y) plane at azimuth
arctan(1/sqrt(2)) ≈ 35.264 deg from +k_x. This azimuth is written into
meta.json so the tilt runner can place the beam tilt along the correct
in-plane direction.

Output:
    <outdir>/si110_<tag>.xyz          (Prismatic 6-col format: Z x y z occ sigma)
    <outdir>/si110_<tag>.cif          (ASE CIF for inspection)
    <outdir>/meta.json                (physics + tilt sweep params for the pipeline)
"""

import argparse
import json
import math
import os

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import write


# ----------------------- physics helpers (reused from si100_slab_generator) -----------------------

def relativistic_lambda_A(keV: float) -> float:
    h = 4.135667696e-15
    c = 2.99792458e8
    m0c2 = 510.99895e3
    E = keV * 1e3
    lam_m = h * c / math.sqrt(E * (E + 2.0 * m0c2))
    return lam_m * 1e10


def bragg_angle_mrad(keV: float, a0_A: float, hkl) -> float:
    lam = relativistic_lambda_A(keV)
    h, k, l = hkl
    d_hkl = a0_A / math.sqrt(h * h + k * k + l * l)
    s = lam / (2.0 * d_hkl)
    if s > 1.0:
        return 0.0
    return math.asin(s) * 1000.0


# ----------------------- prismatic .xyz writer -----------------------

def write_prismatic_xyz(path: str, atoms: Atoms, comment: str, occ: float = 1.0, sigma: float = 0.076):
    Lx, Ly, Lz = atoms.cell.lengths()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(comment.strip() + "\n")
        # Match the 10-decimal precision used for the atom rows so that the
        # cell dimensions read back exactly. With %.6f, Ly can be truncated by
        # ~6e-8 A, which is enough to flip atoms across step boundaries when
        # the staircase y-edges fall near atomic rows.
        f.write(f"{Lx:.10f} {Ly:.10f} {Lz:.10f}\n")
        Zs = atoms.get_atomic_numbers()
        for Z, (x, y, z) in zip(Zs, atoms.positions):
            # 10 decimals: well within float64 precision and avoids %.6f rounding
            # flipping atoms across terrace boundaries when y sits a fraction of
            # a femtometre below Ly/N.
            f.write(f"{int(Z)} {x:.10f} {y:.10f} {z:.10f} {occ:.3f} {sigma:.3f}\n")
        f.write("-1")


# ----------------------- slab-frame geometry -----------------------

def slab_rotation_matrix() -> np.ndarray:
    """
    Rotation that takes crystal-frame positions to slab-frame positions.
    Rows are the slab basis vectors expressed in crystal coords:
        row 0 = x_hat = [1,-1, 1] / sqrt(3)   (surface normal)
        row 1 = y_hat = [1,-1,-2] / sqrt(6)   (step-normal in plane)
        row 2 = z_hat = [1, 1, 0] / sqrt(2)   (beam direction)
    Right-handed: x_hat x y_hat = z_hat.
    """
    e_x = np.array([1, -1,  1], dtype=float) / math.sqrt(3)
    e_y = np.array([1, -1, -2], dtype=float) / math.sqrt(6)
    e_z = np.array([1,  1,  0], dtype=float) / math.sqrt(2)
    return np.array([e_x, e_y, e_z])


def target_g_azimuth_deg(hkl) -> float:
    """
    In-plane azimuth (in the slab frame's (k_x, k_y) plane) of the reciprocal
    vector g = (h, k, l) for B = [110]. Returns angle from +k_x in degrees.
    """
    g = np.array(hkl, dtype=float)
    R = slab_rotation_matrix()
    g_slab = R @ g
    return math.degrees(math.atan2(g_slab[1], g_slab[0]))


def g_dot_n_surf(hkl) -> float:
    """
    Direction cosine between g_hat and the surface normal n_hat = [1,-1,1]/sqrt(3).
    Used for the projected step-height formula:
        Delta_phi = (4 pi / lambda) sin(theta_B) (g_hat . n_hat) h_step
    """
    g = np.array(hkl, dtype=float)
    n = np.array([1, -1, 1], dtype=float) / math.sqrt(3)
    return abs(float(np.dot(g, n) / np.linalg.norm(g)))


# ----------------------- slab build -----------------------

def build_si110_step_slab(a_A: float,
                          n_x_si: int, n_y: int, n_z_si: int,
                          x_vac_A: float, z_vac_A: float,
                          terrace_heights_A):
    """
    Build the cleaved-Si slab in the slab frame and return (atoms, meta_dims).

    n_x_si: layers of x-period (a*sqrt(3)) of Si along surface-normal direction.
    n_y:    layers of y-period (a*sqrt(3/2)) of Si filling the whole y-extent
            (PBC, no vacuum buffer in y).
    n_z_si: layers of z-period (a/sqrt(2)) of Si along the beam direction.
    x_vac:  vacuum thickness on each side in x (above the surface AND below the
            back face, so the slab is x-symmetric in vacuum).
    z_vac:  vacuum thickness on each side in z (front and back of the slab).

    terrace_heights_A: list of "shave-depth" heights, one per terrace, in
        Angstroms. Length = number of terraces N. The y-extent is divided
        into N equal-width bins; terrace i has its surface at
        x_top_max - terrace_heights_A[i]. terrace_heights_A == [0.0] gives
        a perfectly flat reference slab.
    """
    period_x = a_A * math.sqrt(3.0)     # 9.408 A
    period_y = a_A * math.sqrt(1.5)     # 6.652 A
    period_z = a_A / math.sqrt(2.0)     # 3.840 A

    Lx_si = n_x_si * period_x
    Ly    = n_y    * period_y
    Lz_si = n_z_si * period_z
    Lx    = Lx_si + 2.0 * x_vac_A
    Lz    = Lz_si + 2.0 * z_vac_A

    # Build a cubic bulk supercell big enough to cover the rotated slab box
    # in any orientation, then trim. Diagonal of slab box ~ sqrt(Lx^2+Ly^2+Lz^2),
    # so n_cubic ~ ceil(diagonal / a) + 2 (margin for rotation skew).
    L_diag = math.sqrt(Lx * Lx + Ly * Ly + Lz * Lz)
    n_cubic = int(math.ceil(L_diag / a_A)) + 2

    big = bulk("Si", "diamond", a=a_A, cubic=True).repeat((n_cubic, n_cubic, n_cubic))

    # Center crystal at origin
    big_center = big.cell.sum(axis=0) / 2.0
    big.positions -= big_center

    # Rotate to slab frame (positions @ R.T  is the same as (R @ p).T column-wise)
    R = slab_rotation_matrix()
    big.positions = big.positions @ R.T

    pos = big.positions
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]

    # Slab box: x in [-Lx/2, +Lx/2], y in [-Ly/2, +Ly/2], z in [-Lz/2, +Lz/2].
    # Si along [1,-1,1] has a *bilayer* structure: pairs of (1,-1,1) atomic
    # planes separated by 0.78 A (narrow gap inside a bilayer), with 2.35 A
    # between bilayers (wide gap). A clean d_111 step removes one whole bilayer
    # at a time, so each terrace cutoff must lie in a WIDE gap. The cubic-to-
    # slab rotation shifts atomic planes by a phase that depends on where the
    # crystal origin landed after centering, so we cannot snap on a fixed
    # half-d_111 grid -- we have to empirically locate the wide gap.
    d_111 = a_A / math.sqrt(3.0)  # 3.1355 A

    # Restrict to the intended slab region (y, z inside the box; x within the
    # intended Si band ±0.5*d_111 margin) so the plane-finding statistics are
    # not contaminated by the surrounding cubic supercell that we trim away.
    in_y_pre = (y > -Ly / 2.0) & (y < Ly / 2.0)
    in_z_pre = (z > -Lz_si / 2.0) & (z < Lz_si / 2.0)
    in_x_pre = (x > -Lx_si / 2.0 - d_111) & (x < +Lx_si / 2.0 + d_111)
    in_slab = in_y_pre & in_z_pre & in_x_pre
    x_in_box = x[in_slab]

    # The two (1,-1,1) sub-planes of each bilayer give two peaks per period
    # when we fold x mod d_111. The 0.78 A narrow gap sits between them; the
    # 2.35 A wide gap is the complement. Find the midpoint of the wide gap.
    fold = np.mod(x_in_box, d_111)
    # Histogram (200 bins per d_111 -> 0.016 A resolution) and pick the two
    # tallest peaks; the wide-gap midline is opposite the narrow gap midpoint.
    hist, edges = np.histogram(fold, bins=200, range=(0.0, d_111))
    centres = 0.5 * (edges[:-1] + edges[1:])
    # Sort bins by population; the two leading bins should be the two sub-planes.
    order = np.argsort(hist)[::-1]
    peak1, peak2 = sorted([centres[order[0]], centres[order[1]]])
    # Narrow gap centre = midpoint of the two peaks; wide gap is on the other
    # side of the d_111 ring.
    narrow_mid = 0.5 * (peak1 + peak2)
    wide_mid = (narrow_mid + 0.5 * d_111) % d_111  # midpoint of wide gap, mod d_111

    # Anchor x_top_high at the wide-gap midline above the topmost atom in the
    # slab region. Find that atomic plane, then add (wide_gap / 2) so the
    # cutoff sits cleanly above the topmost bilayer.
    si_lateral = in_slab
    x_top_atom = float(x[si_lateral].max())
    # The next wide-gap midline above x_top_atom:
    offset = (wide_mid - x_top_atom) % d_111
    if offset < 1e-6:
        offset += d_111
    x_top_high = x_top_atom + offset

    # Match the back cutoff to the same wide-gap structure so termination is
    # symmetric front-to-back.
    x_back_atom = float(x[si_lateral].min())
    offset_back = (x_back_atom - wide_mid) % d_111
    if offset_back < 1e-6:
        offset_back += d_111
    x_back = x_back_atom - offset_back

    # Bin atoms by y into N terraces of equal width; assign surface height per bin.
    heights = np.asarray(terrace_heights_A, dtype=float)
    N = int(len(heights))
    y_norm = (y + Ly / 2.0) / Ly                                 # in [0, 1)
    terrace_idx = np.clip(np.floor(y_norm * N).astype(int), 0, N - 1)
    x_top_arr = x_top_high - heights[terrace_idx]

    in_x = (x > x_back) & (x < x_top_arr)
    in_y = (y > -Ly / 2.0) & (y < Ly / 2.0)
    in_z = (z > -Lz_si / 2.0) & (z < Lz_si / 2.0)
    keep = in_x & in_y & in_z

    del big[~keep]

    # Now shift to put the supercell origin at (0,0,0) instead of (-L/2,-L/2,-L/2).
    big.positions += np.array([Lx / 2.0, Ly / 2.0, Lz / 2.0])

    # Set the explicit supercell.
    big.set_cell(np.diag([Lx, Ly, Lz]), scale_atoms=False)
    big.set_pbc((True, True, True))

    return big, dict(
        Lx_A=Lx, Ly_A=Ly, Lz_A=Lz,
        Lx_si_A=Lx_si, Lz_si_A=Lz_si,
        x_vacuum_A=x_vac_A, z_vacuum_A=z_vac_A,
        period_x_A=period_x, period_y_A=period_y, period_z_A=period_z,
        x_back_A=x_back + Lx / 2.0,
        x_top_high_A=x_top_high + Lx / 2.0,
        terrace_x_top_A=[float(x_top_high - h + Lx / 2.0) for h in heights],
        d_111_A=d_111,
    )


# ----------------------- main -----------------------

def main():
    ap = argparse.ArgumentParser(
        description="Si[110] cleave-edge slab generator. Beam = [110]; surface normal = [1,-1,1]; step parallel to beam."
    )
    ap.add_argument("--energy-keV", type=float, default=200.0)
    ap.add_argument("--a-A", type=float, default=5.4309, help="Si lattice parameter (A)")

    # supercell layer counts
    ap.add_argument("--n-x-si", type=int, default=9,
                    help="Layers of x-period (a*sqrt(3) = 9.408 A) of Si along surface normal. "
                         "Default 9 -> 84.67 A Si thickness.")
    ap.add_argument("--n-y", type=int, default=12,
                    help="Layers of y-period (a*sqrt(3/2) = 6.652 A). PBC in y, no vacuum buffer. "
                         "Default 12 -> 79.83 A.")
    ap.add_argument("--n-z-si", type=int, default=36,
                    help="Layers of z-period (a/sqrt(2) = 3.840 A) of Si along beam. "
                         "Default 36 -> 138.24 A.")

    # vacuum buffers
    ap.add_argument("--x-vac-A", type=float, default=10.0,
                    help="Vacuum buffer in x on each side (above surface AND below back face). Default 10 A.")
    ap.add_argument("--z-vac-A", type=float, default=30.0,
                    help="Vacuum buffer in z on each side (entrance and exit). Default 30 A.")

    # topography test pattern on the (1,-1,1) facet
    # Heights are "shave-depths" below the maximum surface; integer multiples
    # of d_111 keep clean (1,-1,1) terminations.
    ap.add_argument("--create-step", action="store_true",
                    help="If set, build the test pattern (staircase by default). "
                         "Without this flag the slab is flat (reference).")
    ap.add_argument("--n-terraces", type=int, default=4,
                    help="Number of terraces across y for the staircase. Default 4. "
                         "Each terrace is L_y/N wide (~20 A for default geometry).")
    ap.add_argument("--terrace-heights-bilayers", type=int, nargs="+", default=None,
                    help="Surface depths in units of d_111, one per terrace. "
                         "Length must equal --n-terraces. Default = 0,1,2,...,N-1 "
                         "(monotonic ascending staircase). For Si(2,-2,0) this "
                         "yields Delta_phi steps of 0, 2pi/3, 4pi/3, 6pi/3 -> wraps.")
    ap.add_argument("--step-height-A", type=float, default=None,
                    help="Override the d_111 bilayer unit (A). Default = a/sqrt(3) = 3.1355 A.")

    # target reflection
    ap.add_argument("--target-hkl", type=int, nargs=3, default=[2, -2, 0],
                    help="Target reflection (h k l) for tilt azimuth and pixel-size advisory. Default (2 -2 0).")

    # tilt sweep
    ap.add_argument("--tilt-start", type=float, default=0.0, help="Tilt sweep start (mrad)")
    ap.add_argument("--tilt-end",   type=float, default=11.0, help="Tilt sweep end (mrad)")
    ap.add_argument("--tilt-step",  type=float, default=0.15, help="Tilt sweep step (mrad)")
    ap.add_argument("--discrete-tilts", type=float, nargs="*",
                    help="Explicit list of tilt angles (mrad). Overrides start/end/step.")

    # pixel size
    ap.add_argument("--advisory-px-A", type=float, default=0.5,
                    help="Advisory pixel size (A) written to meta.json. Default 0.5 A.")

    # thermal
    ap.add_argument("--thermal-sigma-A", type=float, default=0.076,
                    help="Frozen-phonon Debye-Waller-like sigma per atom (A). Default 0.076.")
    ap.add_argument("--enable-thermal", action="store_true")
    ap.add_argument("--num-configs", type=int, default=1)

    # output
    ap.add_argument("--outdir", default="outputs/si110_cleave")

    args = ap.parse_args()

    a = args.a_A
    E = args.energy_keV
    hkl = tuple(args.target_hkl)
    d_111 = a / math.sqrt(3.0)
    bilayer_A = args.step_height_A if args.step_height_A is not None else d_111

    theta_b_mrad = bragg_angle_mrad(E, a, hkl)
    d_hkl_A = a / math.sqrt(sum(int(i) * int(i) for i in hkl))
    az_deg = target_g_azimuth_deg(hkl)
    g_dot_n = g_dot_n_surf(hkl)

    # Resolve the per-terrace height list
    if args.create_step:
        n_t = max(1, int(args.n_terraces))
        if args.terrace_heights_bilayers is not None:
            if len(args.terrace_heights_bilayers) != n_t:
                raise SystemExit(
                    f"--terrace-heights-bilayers has {len(args.terrace_heights_bilayers)} entries "
                    f"but --n-terraces is {n_t}; they must match."
                )
            terrace_units = list(args.terrace_heights_bilayers)
        else:
            terrace_units = list(range(n_t))   # 0, 1, 2, ..., N-1
        terrace_heights_A = [u * bilayer_A for u in terrace_units]
    else:
        terrace_units = [0]
        terrace_heights_A = [0.0]

    tag = "object" if args.create_step else "reference"
    out = os.path.join(args.outdir, f"si110_{tag}")
    os.makedirs(out, exist_ok=True)

    print(f"-> Building Si[110] {tag} (cleave-edge geometry) | E = {E:.1f} keV")
    print(f"   target (hkl) = {hkl}: d = {d_hkl_A:.4f} A, theta_B = {theta_b_mrad:.3f} mrad, "
          f"|g_hat . n_surf| = {g_dot_n:.4f}, azimuth = {az_deg:.3f} deg")
    if args.create_step:
        print(f"   terraces:     {len(terrace_units)}, heights (bilayers) = {terrace_units}, "
              f"bilayer = {bilayer_A:.4f} A")

    atoms, dims = build_si110_step_slab(
        a_A=a,
        n_x_si=args.n_x_si, n_y=args.n_y, n_z_si=args.n_z_si,
        x_vac_A=args.x_vac_A, z_vac_A=args.z_vac_A,
        terrace_heights_A=terrace_heights_A,
    )

    print(f"   supercell (A): Lx = {dims['Lx_A']:.3f}, Ly = {dims['Ly_A']:.3f}, Lz = {dims['Lz_A']:.3f}")
    print(f"   Si region:    Lx_si = {dims['Lx_si_A']:.3f}, Lz_si = {dims['Lz_si_A']:.3f}")
    if args.create_step:
        terrace_width = dims['Ly_A'] / len(terrace_units)
        print(f"   each terrace: {terrace_width:.2f} A wide in y, step edges run along beam (z)")
    print(f"   atoms:        {len(atoms)}")

    # write outputs
    comment = f"Si[110] {tag}; cleave-edge slab; beam=[110]; n_surf=[1,-1,1]; E={E:.1f} keV"
    write_prismatic_xyz(os.path.join(out, f"si110_{tag}.xyz"), atoms, comment,
                        occ=1.0, sigma=args.thermal_sigma_A)
    write(os.path.join(out, f"si110_{tag}.cif"), atoms)

    # meta.json
    meta = dict(
        material="Silicon",
        structure="Diamond Cubic; cleave-edge slab; beam = [110]",
        a_A=a,
        a0_A=a,
        energy_keV=E,

        # geometry: who is who
        zone_axis=[1, 1, 0],
        beam_direction_hkl=[1, 1, 0],
        surface_normal_hkl=[1, -1, 1],
        step_edge_direction_hkl=[1, 1, 0],         # step edge parallel to beam
        step_normal_in_plane_hkl=[1, -1, -2],

        # target reflection
        target_hkl=list(hkl),
        target_d_A=d_hkl_A,
        target_theta_B_mrad=theta_b_mrad,
        g_dot_n_surf=g_dot_n,
        tilt_azimuth_deg=az_deg,

        # topography test pattern
        step_hkl=[1, 1, 1] if args.create_step else None,
        bilayer_height_A=bilayer_A,
        step_height_A=bilayer_A if args.create_step else 0.0,  # legacy single-step field
        n_terraces=len(terrace_units),
        terrace_heights_bilayers=terrace_units,
        terrace_heights_A=terrace_heights_A,

        # supercell geometry (all in slab frame)
        Lx_A=dims["Lx_A"], Ly_A=dims["Ly_A"], Lz_A=dims["Lz_A"],
        Lx_si_A=dims["Lx_si_A"], Lz_si_A=dims["Lz_si_A"],
        x_vacuum_A=dims["x_vacuum_A"], z_vacuum_A=dims["z_vacuum_A"],
        period_x_A=dims["period_x_A"], period_y_A=dims["period_y_A"], period_z_A=dims["period_z_A"],
        box_A=[dims["Lx_A"], dims["Ly_A"], dims["Lz_A"]],

        # alpha_deg retained for downstream compatibility (some scripts read it)
        alpha_deg=theta_b_mrad * 180.0 / math.pi / 1000.0,

        advisory_pixel_size_A=args.advisory_px_A,
        thermal_sigma_A=args.thermal_sigma_A,

        # thermal
        enable_thermal_effects=args.enable_thermal,
        num_frozen_phonon_configs_per_subset=args.num_configs,
        num_subsets=1,
    )

    if args.discrete_tilts is not None:
        meta["tilt_angles_mrad"] = list(args.discrete_tilts)
        print(f"   tilt series: discrete angles {args.discrete_tilts} mrad")
    else:
        meta["tilt_start_mrad"] = args.tilt_start
        meta["tilt_end_mrad"] = args.tilt_end
        meta["tilt_step_mrad"] = args.tilt_step
        n_tilts = int(round((args.tilt_end - args.tilt_start) / args.tilt_step)) + 1
        print(f"   tilt series: {args.tilt_start} -> {args.tilt_end} mrad, step {args.tilt_step} mrad ({n_tilts} angles)")

    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"   wrote: {out}/si110_{tag}.xyz")
    print(f"   wrote: {out}/si110_{tag}.cif")
    print(f"   wrote: {out}/meta.json")


if __name__ == "__main__":
    main()
