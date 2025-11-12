import numpy as np
import torch
from torch.utils.data import Dataset

class InpaintingDataset(Dataset):
    def __init__(self, segments, mean_mask_percentage=0.02, 
                 std_mask_percentage=0.005, min_clip=0.05, max_clip=0.15,
                 fhr_mean=None, fhr_std=None, uc_mean=None, uc_std=None):
        """
        Dataset that randomly masks portions of signals for inpainting training.
        """
        self.segments = segments
        self.mean_mask_percentage = mean_mask_percentage
        self.std_mask_percentage = std_mask_percentage
        self.min_clip = min_clip
        self.max_clip = max_clip
        
        # Calculate normalization stats from all segments if not provided
        if fhr_mean is None:
            all_fhr = np.concatenate([s[0] for s in segments])
            all_uc = np.concatenate([s[1] for s in segments])
            self.fhr_mean, self.fhr_std = all_fhr.mean(), all_fhr.std()
            self.uc_mean, self.uc_std = all_uc.mean(), all_uc.std()
        else:
            self.fhr_mean, self.fhr_std = fhr_mean, fhr_std
            self.uc_mean, self.uc_std = uc_mean, uc_std
        
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        # Get original segment and its length
        original = self.segments[idx].copy()
        length = original.shape[1]
        
        # Normalize both channels
        normalized = original.copy()
        normalized[0] = (original[0] - self.fhr_mean) / self.fhr_std
        normalized[1] = (original[1] - self.uc_mean) / self.uc_std
        
        # Start with normalized data for corruption
        corrupted = normalized.copy()
        
        # Decide how much of the signal to mask (total %)
        total_mask_pct = np.random.uniform(self.min_clip, self.max_clip)
        target_samples = int(length * total_mask_pct)
        
        # Track which samples are available for masking
        available = np.ones(length, dtype=bool)
        masked_count = 0
        
        # Keep adding random mask blocks until we hit our target
        for _ in range(1000):  # Safety limit
            if masked_count >= target_samples:
                break
            
            # Sample size for this mask block (with some randomness)
            block_pct = np.random.normal(self.mean_mask_percentage, self.std_mask_percentage)
            block_pct = np.clip(block_pct, 0.001, 0.05)  # Keep it reasonable
            
            block_size = max(1, int(length * block_pct))
            block_size = min(block_size, target_samples - masked_count)  # Don't overshoot
            
            if block_size == 0:
                break
            
            # Find spots where we can place this block
            if length > block_size:
                valid_starts = np.where(available[:-block_size])[0]
                
                if len(valid_starts) > 0:
                    # Pick random start and mask that block
                    start = np.random.choice(valid_starts)
                    available[start:start + block_size] = False
                    masked_count += block_size
                else:
                    break  # No space left
        
        # Zero out the masked regions in the FHR channel
        corrupted[0, ~available] = 0
        
        # Return as tensors: (input with masks, clean target)
        target = torch.from_numpy(normalized[0]).float().unsqueeze(0)
        input_tensor = torch.from_numpy(corrupted).float()
        
        return input_tensor, target


class WindowDataset(Dataset):
    '''
    Extract Windows from Signals
    '''
    def __init__(self, signals, window_size, stride):
        self.windows = []
        for signal_data in signals:
            fhr = signal_data["fhr"]
            uc = signal_data["uc"]
            seq_len = len(fhr)
            for i in range(0, seq_len - window_size + 1, stride):
                window = np.stack([fhr[i:i+window_size], uc[i:i+window_size]], axis=0)
                self.windows.append(window)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.windows[idx])