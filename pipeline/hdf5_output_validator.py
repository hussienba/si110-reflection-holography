#!/usr/bin/env python3
import h5py

fname = "/Users/hussienballouk/Desktop/Research - 2025/Stage 1 - Replicate Osakabe's paper and simulations. /#2 - Electron holography simulation. /outputs/run_manual/tilt_0000mrad/hrtem_sim_wavefunction_output_of_subset_0.h5"

with h5py.File(fname, 'r') as f:
    print('Top-level keys:', list(f.keys()))
    if 'data' in f:
        print('Data keys:', list(f['data'].keys()))
    if 'data/image_wavefunctions' in f:
        print('Wavefunction shape:', f['data/image_wavefunctions'].shape)
        print('Wavefunction dtype:', f['data/image_wavefunctions'].dtype)
