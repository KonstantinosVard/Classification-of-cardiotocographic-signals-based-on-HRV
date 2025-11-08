import torch
import torch.nn as nn
import torch.nn.functional as F

# General building block for the ContextualUNet1D 
# and the Generator (no longer with us)
class DoubleConv(nn.Module):
    """Two convolution blocks: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU"""
    
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding='same', bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding='same', bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

# AutoEncoder with skip connections with purpose to reconstruct zeroed signal
class ContextualUNet1D(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, kernel_size=7, 
                 base_channels=16, depth=2, use_intermediate_skips=True):
        super().__init__()
        
        self.depth = depth
        self.use_intermediate_skips = use_intermediate_skips
        
        # Initial convolution (before downsampling)
        self.inc = DoubleConv(in_channels, base_channels, kernel_size=kernel_size)
        
        # Downsampling blocks (encoder)
        self.downs = nn.ModuleList()
        for i in range(depth):
            ch_in = base_channels * (2**i)
            ch_out = base_channels * (2**(i+1))
            self.downs.append(
                nn.Sequential(
                    nn.MaxPool1d(2),
                    DoubleConv(ch_in, ch_out, kernel_size=kernel_size)
                )
            )
        
        # Upsampling blocks (decoder)
        self.ups = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            ch_in = base_channels * (2**(i+1))
            ch_out = base_channels * (2**i)
            
            # Upsampling layer
            self.ups.append(nn.ConvTranspose1d(ch_in, ch_out, kernel_size=2, stride=2))
            
            # Convolution after concatenation
            # Last block always uses skip connection, others depend on the flag
            is_final_block = (i == 0)
            conv_ch_in = ch_out * 2 if (self.use_intermediate_skips or is_final_block) else ch_out
            self.ups.append(DoubleConv(conv_ch_in, ch_out, kernel_size=kernel_size))
        
        # Final 1x1 convolution to get desired output channels
        self.outc = nn.Conv1d(base_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Store encoder outputs for skip connections
        skips = []
        
        # Encoder: progressively downsample
        x = self.inc(x)
        skips.append(x)
        
        for down in self.downs:
            x = down(x)
            skips.append(x)
        
        # Decoder: progressively upsample
        # Start from bottleneck (last encoder output)
        x = skips.pop()
        
        # Process upsampling layers in pairs (upsample + conv)
        for i in range(0, len(self.ups), 2):
            upsample = self.ups[i]
            conv = self.ups[i + 1]
            
            # Upsample
            x = upsample(x)
            
            # Get corresponding encoder feature
            skip = skips.pop()
            
            # Concatenate skip connection (always for final block, conditionally for others)
            is_final_block = (len(skips) == 0)
            if self.use_intermediate_skips or is_final_block:
                x = torch.cat([skip, x], dim=1)
            
            # Process concatenated features
            x = conv(x)

        return self.outc(x)     
        

class WindowAE(nn.Module):
    """
    Simple Autoencoder for fixed-size windows (10 timeframes * 2 channels).
    """
    def __init__(self, window_size=10, in_channels=2, latent_dim=4):
        super().__init__()
        self.window_size = window_size
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
        input_size = in_channels * window_size  # 2*10 = 20
        
        # Encoder: Compress window to latent space
        self.encoder = nn.Sequential(
            nn.Flatten(),  # (batch, 2, 10) -> (batch, 20)
            nn.Linear(input_size, 12),
            nn.ReLU(),
            nn.Linear(12, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim) # Direct to latent vector z
        )
        
        # Decoder: Reconstruct window from latent
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 12),
            nn.ReLU(),
            nn.Linear(12, input_size)
        )
        
    def encode(self, x):
        """Encode window to a single latent vector z."""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent to reconstructed window."""
        h = self.decoder(z)
        return h.view(-1, self.in_channels, self.window_size)
    
    def forward(self, x):
        """Full forward pass: encode -> decode."""
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z