import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual block for deep feature extraction"""
    
    def __init__(self, n_features: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_features, n_features, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_features, n_features, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_features)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class CNN_SuperResolution(nn.Module):
    """
    CNN-based super-resolution model for marine gravity fields
    Architecture inspired by SRResNet, adapted for geophysical grid data
    
    Justification: CNNs with residual learning excel at learning the mapping
    from coarse to fine spatial patterns, making them ideal for gravity
    field super-resolution where local spatial context is critical
    """
    
    def __init__(self, n_input_channels: int = 4, 
                 n_output_channels: int = 1,
                 n_features: int = 64,
                 n_residual_blocks: int = 16,
                 upscale_factor: int = 4):
        super(CNN_SuperResolution, self).__init__()
        
        self.upscale_factor = upscale_factor
        
        # Initial feature extraction
        self.initial_conv = nn.Sequential(
            nn.Conv2d(n_input_channels, n_features, kernel_size=9, padding=4),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks for deep feature learning
        residual_blocks = []
        for _ in range(n_residual_blocks):
            residual_blocks.append(ResidualBlock(n_features))
        self.residual_blocks = nn.Sequential(*residual_blocks)
        
        # Second convolution after residual blocks
        self.second_conv = nn.Sequential(
            nn.Conv2d(n_features, n_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_features)
        )
        
        # Upsampling (pixel shuffle for 4x super-resolution)
        self.upsampling = nn.Sequential(
            nn.Conv2d(n_features, n_features * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.ReLU(inplace=True)
        )
        
        # Output convolution
        self.output_conv = nn.Conv2d(n_features, n_output_channels, kernel_size=9, padding=4)
        
    def forward(self, x):
        # Initial feature extraction
        initial_features = self.initial_conv(x)
        
        # Deep residual learning
        residual_features = self.residual_blocks(initial_features)
        
        # Skip connection
        final_features = self.second_conv(residual_features) + initial_features
        
        # Upsampling
        upsampled_features = self.upsampling(final_features)
        
        # Output
        output = self.output_conv(upsampled_features)
        
        return output
    
    def get_feature_maps(self, x):
        """Extract intermediate feature maps for visualization"""
        features = {}
        
        features['input'] = x
        features['initial'] = self.initial_conv(x)
        features['residual'] = self.residual_blocks(features['initial'])
        features['upsampled'] = self.upsampling(features['residual'])
        features['output'] = self.output_conv(features['upsampled'])
        
        return features


class UNet_SuperResolution(nn.Module):
    """
    U-Net architecture for gravity super-resolution
    Alternative to pure CNN, better for preserving spatial structure
    
    Justification: U-Net's skip connections preserve fine-scale details
    that are critical for coastal gravity features
    """
    
    def __init__(self, n_input_channels: int = 4,
                 n_output_channels: int = 1,
                 initial_filters: int = 64,
                 depth: int = 4):
        super(UNet_SuperResolution, self).__init__()
        
        self.depth = depth
        self.down_path = nn.ModuleList()
        self.up_path = nn.ModuleList()
        
        # Encoder (downsampling path)
        in_channels = n_input_channels
        for i in range(depth):
            out_channels = initial_filters * (2 ** i)
            self.down_path.append(
                self._double_conv(in_channels, out_channels)
            )
            in_channels = out_channels
        
        # Bridge
        bridge_channels = initial_filters * (2 ** depth)
        self.bridge = self._double_conv(in_channels, bridge_channels)
        
        # Decoder (upsampling path)
        for i in range(depth, 0, -1):
            in_channels = initial_filters * (2 ** i)
            out_channels = initial_filters * (2 ** (i - 1))
            self.up_path.append(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            )
            self.up_path.append(
                self._double_conv(in_channels, out_channels)
            )
        
        # Output
        self.output_conv = nn.Conv2d(initial_filters, n_output_channels, kernel_size=1)
        
    def _double_conv(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for down in self.down_path:
            x = down(x)
            skip_connections.append(x)
            x = nn.MaxPool2d(2)(x)
        
        # Bridge
        x = self.bridge(x)
        
        # Decoder
        for i, up in enumerate(self.up_path):
            if isinstance(up, nn.ConvTranspose2d):
                x = up(x)
                skip = skip_connections[-(i//2 + 1)]
                # Handle size mismatches
                if x.shape != skip.shape:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat([skip, x], dim=1)
            else:
                x = up(x)
        
        return self.output_conv(x)