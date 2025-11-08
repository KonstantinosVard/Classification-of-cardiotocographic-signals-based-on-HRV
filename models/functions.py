import torch
from models.definitions import ContextualUNet1D
from helpers.zero_handlers import drop_long_zero_regions, find_long_zero_runs
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from helpers.dataset import WindowDataset
from torch.utils.data import DataLoader

def load_best_ae_and_stats(model_path, device):
    """Load trained ContextualUNet1D and normalization stats."""
    print(f"Loading model from {model_path}...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "results/autoencoder/grid_search/best_inpainting_model.pth"

    model = ContextualUNet1D(
        in_channels=2, 
        out_channels=1, 
        kernel_size=3, 
        base_channels=32, 
        depth=2, 
        use_intermediate_skips=True
    )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    stats = {
        'fhr_mean': checkpoint['fhr_mean'],
        'fhr_std': checkpoint['fhr_std'],
        'uc_mean': checkpoint['uc_mean'],
        'uc_std': checkpoint['uc_std']
    }

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print("Model loaded successfully.")
    print(f"Stats: FHR μ={stats['fhr_mean']:.2f}, σ={stats['fhr_std']:.2f} | UC μ={stats['uc_mean']:.2f}, σ={stats['uc_std']:.2f}")
    return model, stats


def preprocess_segment(test_segment, stats, device):
    """Normalize and prepare a raw [2, N] segment for model inference."""
    original_fhr = test_segment[0].copy()
    original_uc = test_segment[1].copy()
    is_gappy = (original_fhr == 0)

    norm_fhr = (original_fhr - stats['fhr_mean']) / stats['fhr_std']
    norm_uc = (original_uc - stats['uc_mean']) / stats['uc_std']
    norm_fhr[is_gappy] = 0

    input_numpy = np.stack([norm_fhr, norm_uc], axis=0)
    input_tensor = torch.from_numpy(input_numpy).float().unsqueeze(0).to(device)
    return input_tensor, original_fhr


def postprocess_output(predicted_normalized_fhr, stats):
    """De-normalize the model output back to BPM."""
    output_numpy = predicted_normalized_fhr.squeeze().cpu().numpy()
    return (output_numpy * stats['fhr_std']) + stats['fhr_mean']


def autoencoder_inpaint(signal_fhr, signal_uc, model, stats, device):
    """Use the trained autoencoder to reconstruct missing FHR values."""
    divisor = 4  # For depth=2 UNet (must be divisible by 4)

    signal = np.stack([signal_fhr, signal_uc], axis=0)
    input_tensor, _ = preprocess_segment(signal, stats, device)

    # Pad to ensure divisibility by 4
    original_length = input_tensor.shape[2]
    padding_needed = (divisor - (original_length % divisor)) % divisor
    if padding_needed > 0:
        input_tensor = F.pad(input_tensor, (0, padding_needed))

    with torch.no_grad():
        predicted_normalized_fhr = model(input_tensor)

    predicted_normalized_fhr = predicted_normalized_fhr[..., :original_length]
    reconstructed_fhr = postprocess_output(predicted_normalized_fhr, stats)
    return reconstructed_fhr


def drop_long_then_autoencoder(signal_fhr, signal_uc, model, stats, device, cutoff=10):
    """
    1. Drop long zero runs (> cutoff)
    2. Use autoencoder to inpaint remaining short gaps.
    """
    # Step 1: Drop long zero regions
    trimmed_fhr = drop_long_zero_regions(signal_fhr, cutoff=cutoff)

    # Step 2: Adjust UC to match the trimmed FHR length

    mask = np.ones_like(signal_fhr, dtype=bool)
    for start, end in find_long_zero_runs(signal_fhr, cutoff):
        mask[start:end + 1] = False
    trimmed_uc = signal_uc[mask]

    # Step 3: Run autoencoder on trimmed signal
    reconstructed = autoencoder_inpaint(trimmed_fhr, trimmed_uc, model, stats, device)
    return reconstructed


def ae_loss(recon, x):
    """Standard AE loss: reconstruction (MSE)."""
    recon_loss = F.mse_loss(recon, x, reduction='mean')
    return recon_loss

def train(loader, model, optimizer, epochs, device):
    """Train AE on windows from normal signals."""
    model.train()
    print(f"Training on {len(loader.dataset)} windows...")
    
    for epoch in range(epochs):
        total_loss = 0
        
        for windows in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            windows = windows.to(device)
            
            # Forward pass (simpler output)
            recon, z = model(windows)
            loss = ae_loss(recon, windows)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f"  Reconstruction Loss: {avg_loss:.4f}")

def get_signal_recon_error(model, signal_data, window_size, stride, batch_size, device):
    """
    Calculates the average reconstruction error (MSE) for an entire signal.
    """
    model.eval()
    
    dataset = WindowDataset([signal_data], window_size, stride)
    if len(dataset) == 0:
        return 0.0  # Return 0 error if signal is empty
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    total_loss = 0
    with torch.no_grad():
        for windows in loader:
            windows = windows.to(device)
            recon, _ = model(windows) 
            loss = ae_loss(recon, windows)
            total_loss += loss.item()
            
    # Return the average loss per batch for this signal
    return total_loss / len(loader)