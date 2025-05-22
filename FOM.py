import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureNormalization(nn.Module):
    def __init__(self):
        super(FeatureNormalization, self).__init__()

    def forward(self, x):
        return F.normalize(x, p=2, dim=1)

class GaussianMixtureLayer(nn.Module):
    def __init__(self, in_channels, num_mixtures=3):  # 减少混合成分数量
        super(GaussianMixtureLayer, self).__init__()
        self.num_mixtures = num_mixtures
        self.mean = nn.Parameter(torch.randn(num_mixtures, in_channels))
        self.covariance = nn.Parameter(torch.eye(in_channels).repeat(num_mixtures, 1, 1))  # 使用对角协方差矩阵

    def forward(self, x):
        B, C, H, W = x.size()
        x_flat = x.view(B, C, -1).permute(0, 2, 1)
        mean_expanded = self.mean.unsqueeze(0).unsqueeze(2)
        covariance_expanded = self.covariance.unsqueeze(0)
        x_expanded = x_flat.unsqueeze(1)
        diff = x_expanded - mean_expanded
        covariance_inverse = torch.inverse(covariance_expanded + 1e-6 * torch.eye(C, device=x.device).unsqueeze(0))
        exponent = -0.5 * torch.sum(torch.matmul(diff, covariance_inverse) * diff, dim=-1)
        responsibilities = F.softmax(exponent, dim=1).sum(dim=1, keepdim=True).view(B, 1, H, W)

        return responsibilities

class AffineTransform(nn.Module):
    def __init__(self):
        super(AffineTransform, self).__init__()
        self.affine_params = nn.Parameter(torch.eye(2, 3).unsqueeze(0), requires_grad=True)

    def forward(self, x):
        B, C, H, W = x.size()
        grid = F.affine_grid(self.affine_params.repeat(B, 1, 1), x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)

class FinalProcessingModule1(nn.Module):
    def __init__(self, in_channels, num_mixtures=3):  # 减少混合成分数量
        super(FinalProcessingModule1, self).__init__()
        self.rgb_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)  # 减少输出通道数
        self.depth_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.feature_norm = FeatureNormalization()
        self.gmm = GaussianMixtureLayer(in_channels // 2, num_mixtures)
        self.affine_transform = AffineTransform()
        self.refinement = nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, padding=1)  # 使用更少的通道

    def forward(self, rgb_input, depth_input):
        rgb_features = self.feature_norm(self.rgb_conv(rgb_input))
        depth_features = self.feature_norm(self.depth_conv(depth_input))
        combined_features = rgb_features + depth_features

        responsibilities = self.gmm(combined_features)
        weighted_features = combined_features * responsibilities

        transformed_features = self.affine_transform(weighted_features)
        refined_features = self.refinement(transformed_features)

        return refined_features

# Test the FinalProcessingModule
if __name__ == "__main__":
    model = FinalProcessingModule1(in_channels=512, num_mixtures=3)
    rgb_input = torch.randn(2, 512, 15, 20)
    depth_input = torch.randn(2, 512, 15, 20)
    output = model(rgb_input, depth_input)
    print("FinalProcessingModule output shape:", output.shape)
    from mydesignmodel.yzy_model.FindTheBestDec.model.FLOP import CalParams

    CalParams(model, rgb_input, depth_input)
    print("FinalProcessingModule output shape:", output.shape)