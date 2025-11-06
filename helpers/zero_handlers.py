import numpy as np


# Drop all zeros
def drop_zeros(signal):
    """Remove regions of more than 'cutoff' consecutive zeros."""
    mask = np.where(signal!=0.)
    return signal[mask]


# Linear Interpolation method
def interpolate_zeros(signal):
    """Linearly interpolate 0 values in a 1D array."""
    x = np.arange(len(signal))
    y = signal.copy()
    nonzero_idx = y != 0
    if nonzero_idx.sum() == 0 or nonzero_idx.all():
        return y
    return np.interp(x, x[nonzero_idx], y[nonzero_idx])


# find regions with more than cutoff number of zeros
# used only on drop_long_zero_regions function
def find_long_zero_runs(signal, cutoff=10):
    """Find continuous zero regions longer than 'cutoff'."""
    zeros = np.where(signal == 0)[0]
    if len(zeros) == 0:
        return []
    drop = []
    start = zeros[0]
    length = 1
    for i in range(1, len(zeros)):
        if zeros[i] == zeros[i - 1] + 1:
            length += 1
        else:
            if length > cutoff:
                drop.append([start, zeros[i - 1]])
            start = zeros[i]
            length = 1
    if length > cutoff:
        drop.append([start, zeros[-1]])
    return drop

# Drop regions with more than cutoff amount of zeros
# used alongside other methods
def drop_long_zero_regions(signal, cutoff=10):
    """Remove regions of more than 'cutoff' consecutive zeros."""
    drop_ranges = find_long_zero_runs(signal, cutoff)
    mask = np.ones(len(signal), dtype=bool)
    for start, end in drop_ranges:
        mask[start:end + 1] = False
    return signal[mask]

# Fill zero values using a bi-directional weighted average
def weighted_average_zipper(signal, n=10):
    """
    Fill zero values using a bi-directional weighted average "zipper".

    1.  Pass LTR: Fill zeros based on a weighted average of the last 'n' values.
        -   Weights are [n, n-1, ..., 1], favoring the most recent value.
        -   Filled-in values are used for subsequent calculations.
        -   Does not fill initial zeros (before the first non-zero value).
    2.  Pass RTL: Fill zeros based on a weighted average of the next 'n' values.
        -   Weights are [n, n-1, ..., 1], favoring the nearest future value.
        -   Filled-in values are used.
        -   Does not fill trailing zeros (after the last non-zero value).
    3.  Combine:
        -   For initial zeros: Use the RTL pass value.
        -   For trailing zeros: Use the LTR pass value.
        -   For internal zeros: Use the average of the LTR and RTL values.
    """
    if n <= 0:
        raise ValueError("Window size 'n' must be positive.")
        
    filled_ltr = signal.copy()
    filled_rtl = signal.copy()
    final_signal = signal.copy()
    
    non_zero_indices = np.where(signal != 0)[0]
    if len(non_zero_indices) == 0:
        return signal  # All zeros, nothing to do
        
    first_non_zero = non_zero_indices[0]
    last_non_zero = non_zero_indices[-1]
    
    zero_indices = np.where(signal == 0)[0]
    
    # --- 1. Left-to-Right Pass ---
    # Weights: [n, n-1, ..., 1]
    weights = np.arange(n, 0, -1).astype(float)
    
    # Only process zeros *after* the first non-zero value
    for i in range(first_non_zero + 1, len(signal)):
        if filled_ltr[i] == 0:  # Only fill original zeros
            # Get window [i-n, ..., i-2, i-1]
            window = filled_ltr[max(0, i - n) : i]
            
            # Get values in reverse order: [val_at_i-1, val_at_i-2, ...]
            values_to_use = window[::-1]
            
            # Get corresponding weights: [weight_for_i-1, weight_for_i-2, ...]
            weights_to_use = weights[:len(values_to_use)]
            
            weight_sum = np.sum(weights_to_use)
            if weight_sum > 0:
                filled_ltr[i] = np.dot(values_to_use, weights_to_use) / weight_sum

    # --- 2. Right-to-Left Pass ---
    # Weights are the same: [n, n-1, ..., 1]
    
    # Only process zeros *before* the last non-zero value
    for i in range(last_non_zero - 1, -1, -1):
        if filled_rtl[i] == 0: # Only fill original zeros
            # Get window [i+1, i+2, ..., i+n]
            window = filled_rtl[i + 1 : min(len(signal), i + n + 1)]
            
            # Values are in order: [val_at_i+1, val_at_i+2, ...]
            values_to_use = window
            
            # Corresponding weights: [weight_for_i+1, weight_for_i+2, ...]
            weights_to_use = weights[:len(values_to_use)]

            weight_sum = np.sum(weights_to_use)
            if weight_sum > 0:
                filled_rtl[i] = np.dot(values_to_use, weights_to_use) / weight_sum

    # --- 3. Combine ---
    for i in zero_indices:
        val_l = filled_ltr[i]
        val_r = filled_rtl[i]
        
        is_start_gap = i < first_non_zero
        is_end_gap = i > last_non_zero
        
        if is_start_gap:
            final_signal[i] = val_r  # Use only RTL pass
        elif is_end_gap:
            final_signal[i] = val_l  # Use only LTR pass
        else:
            # Internal gap.
            # Since FHR > 0, 0 means "not computed".
            if val_l != 0 and val_r != 0:
                final_signal[i] = (val_l + val_r) / 2.0
            elif val_l != 0:
                final_signal[i] = val_l # RTL pass failed (gap too big), use LTR
            elif val_r != 0:
                final_signal[i] = val_r # LTR pass failed (gap too big), use RTL
            else:
                final_signal[i] = 0 # Both failed, gap is isolated and larger than n
                
    return final_signal