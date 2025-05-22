import torch
import torch.nn as nn
import torch.nn.functional as F


# Scharr 曲率计算
def scharr_curvature(x):
    channels = x.shape[1]
    scharr_x = torch.tensor([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    scharr_y = torch.tensor([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    scharr_x = scharr_x.expand(channels, 1, 3, 3)
    scharr_y = scharr_y.expand(channels, 1, 3, 3)
    grad_x = F.conv2d(x, scharr_x, groups=channels, padding=1)
    grad_y = F.conv2d(x, scharr_y, groups=channels, padding=1)
    curvature = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    return curvature


# CBR模块
class CBR(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.bn(self.conv(x)))
        return x


# 残差卷积单元
class ResidualConv(nn.Module):
    def __init__(self, channels):
        super(ResidualConv, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


# 细化特征模块
class FeatureRefinement(nn.Module):
    def __init__(self, channels):
        super(FeatureRefinement, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


# 主融合模块
class Unifie(nn.Module):
    def __init__(self, channels):
        super(Unifie, self).__init__()
        self.cbr = CBR(channels, channels)
        self.residual_conv = ResidualConv(channels)
        self.curvature_attention = FeatureRefinement(channels)
        self.feature_refinement = FeatureRefinement(channels)
        self.fusion_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, rgb_input, depth_input):
        rgb_features = self.cbr(rgb_input)
        depth_features = self.cbr(depth_input)

        # 分支1: 残差卷积
        rgb_branch1 = self.residual_conv(rgb_features)
        depth_branch1 = self.residual_conv(depth_features)

        # 分支2: 曲率引导
        rgb_curvature = scharr_curvature(rgb_features)
        depth_curvature = scharr_curvature(depth_features)

        rgb_branch2 = self.curvature_attention(rgb_features + rgb_curvature)
        depth_branch2 = self.curvature_attention(depth_features + depth_curvature)

        # 确保分支1和分支2在余弦相似度计算前尺寸匹配
        if rgb_branch1.size() != rgb_branch2.size():
            rgb_branch1 = F.interpolate(rgb_branch1, size=rgb_branch2.shape[2:], mode='bilinear', align_corners=True)
        if depth_branch1.size() != depth_branch2.size():
            depth_branch1 = F.interpolate(depth_branch1, size=depth_branch2.shape[2:], mode='bilinear',
                                          align_corners=True)

        # 相似度计算与加权融合
        sim_rgb = F.cosine_similarity(rgb_branch1, rgb_branch2, dim=1).unsqueeze(1)
        sim_depth = F.cosine_similarity(depth_branch1, depth_branch2, dim=1).unsqueeze(1)
        rgb_branch3 = rgb_branch1 * sim_rgb
        depth_branch3 = depth_branch1 * sim_depth

        # 分支4：进一步细化
        rgb_branch4 = self.feature_refinement(rgb_branch2)
        depth_branch4 = self.feature_refinement(depth_branch2)

        # 确保分支4和分支3尺寸一致，进行最终融合
        if rgb_branch3.size() != rgb_branch4.size():
            rgb_branch4 = F.interpolate(rgb_branch4, size=rgb_branch3.shape[2:], mode='bilinear', align_corners=True)
        if depth_branch3.size() != depth_branch4.size():
            depth_branch4 = F.interpolate(depth_branch4, size=depth_branch3.shape[2:], mode='bilinear',
                                          align_corners=True)

        combined_feature_rgb = rgb_branch3 + rgb_branch4
        combined_feature_depth = depth_branch3 + depth_branch4
        combined_feature = combined_feature_rgb + combined_feature_depth

        # 最终融合
        fused_feature = self.fusion_conv(combined_feature)

        return fused_feature


# 测试
if __name__ == "__main__":
    batch_size, channels, height, width = 2, 128, 60, 80
    rgb_input = torch.randn(batch_size, channels, height, width)
    depth_input = torch.randn(batch_size, channels, height, width)

    model = Unifie(channels)
    output = model(rgb_input, depth_input)

    print("输出特征形状:", output.shape)