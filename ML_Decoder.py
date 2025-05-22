import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

# 定义Deformable Block，用于形变处理
class DeformableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DeformableBlock, self).__init__()
        self.offset_conv = nn.Conv2d(in_channels, 18, kernel_size=kernel_size, padding=padding)
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        offset = self.offset_conv(x)  # 学习形变的偏移量
        x = self.deform_conv(x, offset)  # 可变形卷积
        x = self.bn(x)
        return self.relu(x)

# 反向残差块
class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=6):
        super(InvertedResidualBlock, self).__init__()
        self.expand = nn.Conv2d(in_channels, in_channels * expansion_factor, kernel_size=1)
        self.depthwise = nn.Conv2d(in_channels * expansion_factor, in_channels * expansion_factor,
                                   kernel_size=3, padding=1, groups=in_channels * expansion_factor)
        self.reduce = nn.Conv2d(in_channels * expansion_factor, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels * expansion_factor)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.expand(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.depthwise(x)
        x = self.reduce(x)
        x = self.bn2(x)
        return x + residual  # 残差连接

# 定义解码器，结合Deformable Conv和反向残差块
class CombinedDecoder(nn.Module):
    def __init__(self):
        super(CombinedDecoder, self).__init__()
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deform_block3 = DeformableBlock(512, 320)
        self.inv_res_block3 = InvertedResidualBlock(320, 320)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deform_block2 = DeformableBlock(320, 128)
        self.inv_res_block2 = InvertedResidualBlock(128, 128)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deform_block1 = DeformableBlock(128, 64)
        self.inv_res_block1 = InvertedResidualBlock(64, 64)

        self.final_conv = nn.Conv2d(64, 41, kernel_size=3, padding=1)

    def forward(self, c3, c2, c1, c0):
        # 第三层特征处理
        x = self.up3(c3)
        x = self.deform_block3(x)
        x = self.inv_res_block3(x + c2)  # 残差连接c2

        # 第二层特征处理
        x = self.up2(x)
        x = self.deform_block2(x)
        x = self.inv_res_block2(x + c1)  # 残差连接c1

        # 第一层特征处理
        x = self.up1(x)
        x = self.deform_block1(x)
        x = self.inv_res_block1(x + c0)  # 残差连接c0

        # 最终输出
        out = self.final_conv(x)
        return F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)

# 测试代码
if __name__ == '__main__':
    model = CombinedDecoder()
    c3 = torch.randn(2, 512, 15, 20)
    c2 = torch.randn(2, 320, 30, 40)
    c1 = torch.randn(2, 128, 60, 80)
    c0 = torch.randn(2, 64, 120, 160)

    output = model(c3, c2, c1, c0)
    print("输出形状:", output.shape)