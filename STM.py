import torch
import torch.nn as nn


class Split(nn.Module):
    def __init__(self, dim, reduction=4, lambda_x=0.5, lambda_y=0.5):
        super(Split, self).__init__()
        self.dim = dim
        self.reduction = reduction
        self.lambda_x = lambda_x  # x 方向的 lambda
        self.lambda_y = lambda_y  # y 方向的 lambda

        # MLP 加权模块
        self.channel_weight_mlp = self._build_channel_weight_mlp(dim, reduction)

        # 全局平均池化和最大池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

    def _build_channel_weight_mlp(self, dim, reduction):
        # 生成信道加权系数的 MLP
        return nn.Sequential(
            nn.Linear(dim * 2, dim // reduction),  # 信道缩减
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, 2),  # 输出两个加权
            nn.Sigmoid()
        )

    def split_xy_directions(self, features):
        B, C, H, W = features.shape

        # 动态分割高度和宽度，确保匹配
        h_split_1 = H // 2  # 分割高度为 7
        w_split_1 = W // 2  # 分割宽度为 10

        # x 方向分割
        x_splits = [features[:, :, :, :w_split_1], features[:, :, :, w_split_1:]]
        # print(f"x_splits: {[x.shape for x in x_splits]}")  # 测试 x_splits 的输出形状

        # y 方向分割
        y_splits = [features[:, :, :h_split_1, :], features[:, :, h_split_1:, :]]
        # print(f"y_splits: {[y.shape for y in y_splits]}")  # 测试 y_splits 的输出形状

        return x_splits, y_splits

    def forward(self, rgb, depth):
        B, C, H, W = rgb.shape
        # print(f"输入形状: {rgb.shape}, {depth.shape}")

        # 1. x 和 y 方向分割
        merged_input = rgb + depth  # 先将 RGB 和深度图相加
        x_splits, y_splits = self.split_xy_directions(merged_input)

        # 2. 对 x 方向池化操作
        x_avg_pooled = [self.global_avg_pool(split) for split in x_splits]  # 保持形状
        x_max_pooled = [self.global_max_pool(split) for split in x_splits]
        x_pooled = [avg + max for avg, max in zip(x_avg_pooled, x_max_pooled)]  # 保持形状 [B, C, 1, 1]

        # 3. 对 y 方向池化操作
        y_avg_pooled = [self.global_avg_pool(split) for split in y_splits]  # 保持形状
        y_max_pooled = [self.global_max_pool(split) for split in y_splits]
        y_pooled = [avg + max for avg, max in zip(y_avg_pooled, y_max_pooled)]  # 保持形状 [B, C, 1, 1]

        x_combined = x_pooled[0] + x_pooled[1]
        y_combined = y_pooled[0] + y_pooled[1]

        x_combined = x_combined.view(B,C)
        y_combined = y_combined.view(B,C)

        # 4. MLP 生成加权
        # avg_pooled_input = self.global_avg_pool(merged_input).view(B, C)
        # max_pooled_input = self.global_max_pool(merged_input).view(B, C)
        # # pooled_input = torch.cat([x_pooled[0],x_pooled[1], y_pooled[0],y_pooled[1]], dim=1)
        # pooled_input = torch.cat([avg_pooled_input, max_pooled_input], dim=1)
        pooled_input = torch.cat([x_combined, y_combined], dim=1)
        channel_weights = self.channel_weight_mlp(pooled_input)  # 生成两个加权系数 [B, 2]

        # 5. 将加权系数与 x 和 y 方向池化结果相乘，确保扩展形状匹配
        channel_weights_rgb_local_x = channel_weights[:, 0].view(B, 1, 1, 1).expand(-1, C, H,
                                                                                    W // 2)  # 保持与 x_splits 尺寸匹配

        # 针对 y_splits 中的每个块单独生成权重
        channel_weights_rgb_local_y_1 = channel_weights[:, 1].view(B, 1, 1, 1).expand(-1, C, y_splits[0].shape[2],
                                                                                      W)  # 针对 y_splits[0] 的加权
        channel_weights_rgb_local_y_2 = channel_weights[:, 1].view(B, 1, 1, 1).expand(-1, C, y_splits[1].shape[2],
                                                                                      W)  # 针对 y_splits[1] 的加权

        # 对 x 和 y 方向分割块应用加权
        weighted_x_splits = [split * channel_weights_rgb_local_x for split in x_splits]
        weighted_y_splits = [y_splits[0] * channel_weights_rgb_local_y_1,
                             y_splits[1] * channel_weights_rgb_local_y_2]  # 分开处理 y_splits

        # 6. 恢复 x 和 y 方向的宽度和高度
        fused_x = torch.cat(weighted_x_splits, dim=3)  # 恢复 x 方向的宽度 [B, C, H, W]
        fused_y = torch.cat(weighted_y_splits, dim=2)  # 恢复 y 方向的高度 [B, C, H, W]

        # print(f"fused_x shape: {fused_x.shape}")
        # print(f"fused_y shape: {fused_y.shape}")

        # 7. 最终输出融合，加入 lambda 调整
        final_output = self.lambda_x * fused_x + self.lambda_y * fused_y  # 根据 lambda 调整融合

        return final_output


# 测试完整模块
def main():
    B, C, H, W = 2, 320, 30, 40  # 输入 RGB 和深度图的形状 (H=15)
    rgb = torch.randn(B, C, H, W)  # 模拟 RGB 输入
    depth = torch.randn(B, C, H, W)  # 模拟深度输入

    # 实例化模块并运行
    fusion_module = Split(dim=C, lambda_x=0.6, lambda_y=0.4)  # 调整 lambda 参数
    output = fusion_module(rgb, depth)

    # 输出最终融合结果的形状
    print(f"最终融合输出形状: {output.shape}")


# 运行测试
main()