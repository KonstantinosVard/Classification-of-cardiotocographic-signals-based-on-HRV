import torch
import torch.nn as nn

# General building block for the ContextualUNet1D and the Generator
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