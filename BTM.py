import torch
import torch.nn as nn
import torch.nn.functional as F

class TotalVariationDenoising(nn.Module):
    """Total Variation Denoising module."""
    def __init__(self, weight=0.1):
        super(TotalVariationDenoising, self).__init__()
        self.weight = weight

    def forward(self, x):
        # Calculate the total variation
        diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
        diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]
        tv_h = torch.pow(diff_h, 2).mean(dim=(1, 2, 3), keepdim=True)
        tv_w = torch.pow(diff_w, 2).mean(dim=(1, 2, 3), keepdim=True)
        # Apply the total variation denoising
        return x + self.weight * (tv_h + tv_w)

class BilateralTotalVariation(nn.Module):
    """Bilateral Total Variation module using downsampled guidance maps."""
    def __init__(self, weight=0.1):
        super(BilateralTotalVariation, self).__init__()
        self.weight = weight

    def forward(self, x, guidance):
        # Downsample the guidance map to reduce computational load
        guidance_down = F.interpolate(guidance, scale_factor=0.5, mode='bilinear', align_corners=False)
        weight_h = torch.exp(-torch.abs(guidance_down[:, :, 1:, :] - guidance_down[:, :, :-1, :]))
        weight_w = torch.exp(-torch.abs(guidance_down[:, :, :, 1:] - guidance_down[:, :, :, :-1]))
        # Upsample weights to match the dimensions of the difference results
        weight_h_up = F.interpolate(weight_h, size=(x.size(2)-1, x.size(3)), mode='bilinear', align_corners=False)
        weight_w_up = F.interpolate(weight_w, size=(x.size(2), x.size(3)-1), mode='bilinear', align_corners=False)
        # Compute the bilateral total variation
        diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
        diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]
        btv_h = torch.pow(diff_h, 2) * weight_h_up
        btv_w = torch.pow(diff_w, 2) * weight_w_up
        btv = btv_h.mean(dim=(1, 2, 3), keepdim=True) + btv_w.mean(dim=(1, 2, 3), keepdim=True)
        return x + self.weight * btv

class supixel(nn.Module):
    """Module to fuse features using Total Variation Denoising and Bilateral Total Variation."""
    def __init__(self, in_channels):
        super(supixel, self).__init__()
        self.tvd = TotalVariationDenoising()
        self.btv = BilateralTotalVariation()
        self.conv = nn.Conv2d(in_channels * 2, in_channels, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, rgb, depth):
        # Apply total variation denoising and bilateral total variation
        rgb = self.tvd(rgb)
        depth = self.btv(depth, rgb)
        # Fuse features
        fused = self.conv(torch.cat([rgb, depth], dim=1))
        return self.relu(fused)

# Example usage
if __name__ == "__main__":
    in_channels = 320  # Assuming RGB and depth both have 320 channels
    rgb = torch.randn(2, in_channels, 30, 40)
    depth = torch.randn(2, in_channels, 30, 40)
    module = supixel(in_channels)
    result = module(rgb, depth)
    print(result.shape)  # Expected output: torch.Size([2, 320, 30, 40])