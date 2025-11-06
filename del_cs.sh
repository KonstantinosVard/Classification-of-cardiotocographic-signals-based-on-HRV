#!/bin/bash
# Directory containing the data
DIR="data/all_data"

# Loop over all .hea files in the directory
for hea_file in "$DIR"/*.hea; do
    # Skip if no .hea files exist
    [ -e "$hea_file" ] || continue

    # Check if file contains "#Deliv. type 2"
    if grep -q "#Deliv. type  2" "$hea_file"; then
        # Derive matching .dat file (same base name)
        base_name="${hea_file%.hea}"
        dat_file="${base_name}.dat"

        echo "Deleting: $hea_file and $dat_file"

        # Delete both .hea and .dat files if they exist
        rm -f "$hea_file"
        rm -f "$dat_file"
    fi
done

echo "Cleanup completed"
