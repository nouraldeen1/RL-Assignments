import torch
import torch.nn as nn

class CarRacingEncoder(nn.Module):
    """
    CNN Encoder to process 96x96 images from CarRacing.
    Output: A flattened vector (feature representation) of size `feature_dim`.
    """
    def __init__(self, input_channels=3, feature_dim=256):
        super().__init__()

        # Build a simple conv stack that reduces 96x96 -> 4x4 spatially
        # Final conv output: (256, 4, 4) -> flattened 256*4*4 = 4096
        self.net = nn.Sequential(
            # (3, 96, 96) -> (32, 47, 47)
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),

            # (32, 47, 47) -> (64, 22, 22)
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            # (64, 22, 22) -> (128, 10, 10)
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),

            # (128, 10, 10) -> (256, 4, 4)
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Flatten(),  # -> (256*4*4 = 4096)
        )

        self.proj = nn.Sequential(
            nn.Linear(256 * 4 * 4, feature_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: float tensor in shape (B, C, H, W) with values in [0,255] or [0,1].
        Returns: (B, feature_dim)
        """
        # If input looks like uint8 image, cast and normalize
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            # if values appear large (>1), normalize
            if x.max() > 1.0:
                x = x.float() / 255.0

        feats = self.net(x)
        out = self.proj(feats)
        return out
