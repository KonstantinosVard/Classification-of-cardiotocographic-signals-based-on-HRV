import os
import glob
import numpy as np
import shutil

# Input directory containing .dat and .hea files
INPUT_DIR = "../data/all_data"

# Output directories
THRESHOLDS = [0.05, 0.1, 0.15, 0.2]
OUTPUT_DIRS = [f"../data/zero_quantiles/{t}_data" for t in THRESHOLDS]


def main():
    # Create folders if they don't exist
    for out_dir in OUTPUT_DIRS:
        os.makedirs(out_dir, exist_ok=True)

    # Constants
    fhr_gain = 100
    fhr_baseline = 0

    # Loop over all .dat files
    for dat_file in glob.glob(os.path.join(INPUT_DIR, "*.dat")):
        record_id = os.path.splitext(os.path.basename(dat_file))[0]
        hea_file = os.path.join(INPUT_DIR, f"{record_id}.hea")

        # Read .dat binary data
        data = np.fromfile(dat_file, dtype=np.int16)

        # Extract signals
        fhr_signal = (data[0::2] - fhr_baseline) / fhr_gain

        # Compute percentage of zero values in fhr_signal
        zero_ratio = np.mean(fhr_signal == 0)

        # Determine which thresholds to include
        for t in THRESHOLDS:
            if zero_ratio <= t:
                # Copy both files to the corresponding output directory
                shutil.copy(dat_file, f"../data/zero_quantiles/{t}_data/")
                shutil.copy(hea_file, f"../data/zero_quantiles/{t}_data/")

        print(f"Processed {record_id}: zero_ratio={zero_ratio:.3f}")

    print("Processing complete.")

if __name__ == "__main__":
    main()
