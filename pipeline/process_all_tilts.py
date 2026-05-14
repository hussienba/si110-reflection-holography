#!/usr/bin/env python3
import os
import glob
import json
import subprocess
import sys

def main():
    # Base directory containing the tilt subdirectories
    base_dir = os.path.abspath("../outputs/job_12198643")
    
    # Path to the filter script
    filter_script = os.path.abspath("specular_666_filter.py")
    
    if not os.path.exists(base_dir):
        print(f"Error: Base directory not found: {base_dir}")
        return

    # Find all tilt directories
    tilt_dirs = sorted(glob.glob(os.path.join(base_dir, "tilt_*")))
    
    if not tilt_dirs:
        print(f"No tilt directories found in {base_dir}")
        return

    print(f"Found {len(tilt_dirs)} tilt directories.")

    for tilt_dir in tilt_dirs:
        print(f"\nProcessing: {tilt_dir}")
        
        # 1. Locate the input HDF5 file
        # Based on exploration, it is 'hrtem_sim_wavefunction_output_of_subset_0.h5'
        input_file = os.path.join(tilt_dir, "hrtem_sim_wavefunction_output_of_subset_0.h5")
        
        if not os.path.exists(input_file):
            print(f"  [SKIP] Input file not found: {input_file}")
            continue
            
        # 2. Get pixel size from meta.json
        meta_file = os.path.join(tilt_dir, "meta.json")
        dx = 0.5 # Default
        dy = 0.5
        
        if os.path.exists(meta_file):
            try:
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                    if "advisory_pixel_size_A" in meta:
                        val = float(meta["advisory_pixel_size_A"])
                        dx = val
                        dy = val
                        print(f"  [INFO] Found pixel size in meta.json: {dx:.6f} A/px")
            except Exception as e:
                print(f"  [WARN] Could not read meta.json: {e}")
        else:
            print("  [WARN] meta.json not found, using default pixel size 0.5 A/px")

        # 3. Construct command
        # We will save the output in a 'filtered' subdirectory within the tilt directory
        # The script takes --out-prefix. 
        # If we want output in `tilt_dir/filtered/`, we can set prefix to `tilt_dir/filtered/result`
        
        output_dir = os.path.join(tilt_dir, "filtered")
        os.makedirs(output_dir, exist_ok=True)
        out_prefix = os.path.join(output_dir, "specular_666")
        
        cmd = [
            sys.executable,
            filter_script,
            input_file,
            "--out-prefix", out_prefix,
            "--dx", str(dx),
            "--dy", str(dy)
        ]
        
        # 4. Run the command
        print(f"  [EXEC] Running filter script...")
        try:
            subprocess.run(cmd, check=True)
            print("  [DONE] Success.")
        except subprocess.CalledProcessError as e:
            print(f"  [FAIL] Error running script: {e}")

if __name__ == "__main__":
    main()
