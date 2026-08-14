import torch
import torch.nn as nn

# ------------------------------
# GAT模型架构初版（任务交付版本）
# 注释：本项目计划基于PyTorch‑Geometric的GATConv实现图注意力网络
# ------------------------------
class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.heads = heads

        # 计划使用：from torch_geometric.nn import GATConv
        # self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        # self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index):
        # x = self.conv1(x, edge_index).relu()
        # x = self.conv2(x, edge_index)
        # return x
        raise NotImplementedError("PyTorch‑Geometric环境待安装，网络骨架已定义")